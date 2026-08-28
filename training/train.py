"""
Train the Foresight world model.

Runs on a free Colab T4 GPU (or CPU for the synthetic smoke test). With no
--data it trains on synthetic attack-progression traffic in ~seconds, so you
can confirm the whole pipeline works BEFORE pointing it at CIC-IDS-2018.

Colab quickstart:
    !git clone <repo> && cd sentinel-foresight && pip install -q torch scikit-learn
    !python training/train.py                                   # synthetic smoke test
    !python training/train.py --data "data/cicids/*.csv" --epochs 30 --out /content/drive/MyDrive/foresight_ckpt

The checkpoint is portable (model weights + scaler + feature list + config),
so the offline Streamlit demo loads it on your laptop with no retraining.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

# Make the repo importable whether run from root or elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from foresight.data.state import flows_to_state_windows, make_sequences, STATE_FEATURES
from foresight.data.synth import make_synthetic_flows
from foresight.model.world_model import (
    WorldModel, WorldModelConfig, WorldModelLoss, infiltration_prob, INFILTRATION_STAGE, N_STAGES,
)


def build_dataset(args):
    """Load + window each capture file one at a time, discarding the raw
    flow DataFrame before loading the next.

    A CIC-IDS-2018 day CSV carries ~80 raw CICFlowMeter columns even though
    only ~14 are used; holding all 10 days' raw DataFrames in memory at
    once (the previous approach) multiplies peak RAM ~10x over processing
    one at a time and is what got the run OOM-killed with no traceback on
    a free-tier Colab instance. Only the small windowed float32 arrays
    (state vectors, not raw flows) are kept per file.

    Days are windowed independently, not concatenated into one timeline:
    that would force the model to "learn" a transition across the overnight
    gap between Friday's traffic and Monday's, which never happened, and
    (with fill_gaps) blows up window count to one per elapsed second across
    the whole multi-day span instead of per second of actual capture.
    """
    import gc
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6")
    except OSError:
        _libc = None  # non-glibc platform (e.g. macOS) — gc.collect() alone still runs

    per_file_X, per_file_y = [], []
    if args.data:
        from foresight.data.cicids import load_cicids_csv
        files = sorted(glob.glob(args.data))
        if not files:
            raise SystemExit(f"no files match {args.data!r}")
        print(f"loading + windowing {len(files)} CIC-IDS CSV(s)…")
        for f in files:
            df = load_cicids_csv(f, nrows=args.nrows)
            X, y, _ts = flows_to_state_windows(df, window_seconds=args.window_seconds)
            print(f"  {f}: {len(df):,} flows -> {len(X):,} windows")
            per_file_X.append(X.values.astype(np.float32))
            per_file_y.append(y)
            del df, X
            gc.collect()
            # gc.collect() frees Python objects, but glibc's malloc often keeps
            # the underlying pages rather than returning them to the OS — over
            # 10 files that's what compounded into an OOM kill. malloc_trim(0)
            # forces the freed arenas back, so RSS actually drops between files.
            if _libc is not None:
                _libc.malloc_trim(0)
    else:
        print("no --data given → synthetic attack-progression traffic (smoke test)")
        df = make_synthetic_flows(seed=args.seed, minutes=8.0)
        X, y, _ts = flows_to_state_windows(df, window_seconds=args.window_seconds)
        per_file_X.append(X.values.astype(np.float32))
        per_file_y.append(y)

    n_windows = sum(len(x) for x in per_file_X)
    print(f"state windows: {n_windows}  features: {per_file_X[0].shape[1]}")

    # Temporal split: earlier files (days) are train, later ones val — no
    # leakage, and no split boundary falls inside a single day's sequence.
    split_file = max(1, int(round(0.7 * len(per_file_X))))
    split_file = min(split_file, len(per_file_X) - 1) if len(per_file_X) > 1 else len(per_file_X)

    Xv_train = np.concatenate(per_file_X[:split_file], axis=0) if split_file else per_file_X[0][:0]
    scaler = StandardScaler().fit(Xv_train)                # fit on TRAIN files only (no leakage)

    def sequences_for(files_slice):
        xs, yn, ys = [], [], []
        for X, y in files_slice:
            Xs = scaler.transform(X).astype(np.float32)
            xseq, ynext, ystage = make_sequences(Xs, y, window=args.window)
            if len(xseq):
                xs.append(xseq); yn.append(ynext); ys.append(ystage)
        if not xs:
            F = Xv_train.shape[1]
            return (np.empty((0, args.window, F), np.float32),
                    np.empty((0, F), np.float32), np.empty((0,), int))
        return np.concatenate(xs), np.concatenate(yn), np.concatenate(ys)

    paired = list(zip(per_file_X, per_file_y))
    Xseq_tr, Ynext_tr, Ystage_tr = sequences_for(paired[:split_file])
    Xseq_va, Ynext_va, Ystage_va = sequences_for(paired[split_file:])
    if len(Xseq_tr) < 8:
        raise SystemExit("too few training sequences — lower --window or --window-seconds")
    print(f"sequences: train {len(Xseq_tr)}  val {len(Xseq_va)}")

    def ds(xseq, ynext, ystage):
        return TensorDataset(torch.tensor(xseq), torch.tensor(ynext),
                             torch.tensor(ystage, dtype=torch.long))

    # Inverse-frequency class weights from the TRAIN split only (no val
    # leakage). CIC-IDS-2018 is overwhelmingly benign, so unweighted CE
    # lets the model default to predicting the majority stage — that's why
    # the first real training run got 96% infiltration precision but only
    # 12% recall (confident, but barely ever fires). Clamp to keep rare
    # classes from dominating the loss and destabilising training.
    # Sized to N_STAGES (the model's fixed stage_head output), not just
    # whatever stages happen to appear in this training split.
    counts = np.bincount(Ystage_tr, minlength=N_STAGES)[:N_STAGES].astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for a stage absent from train
    weights = len(Ystage_tr) / (N_STAGES * counts)
    weights = np.clip(weights, 0.1, 20.0).astype(np.float32)
    print(f"class weights (stage 0..{N_STAGES-1}): {weights.round(2).tolist()}")

    return (ds(Xseq_tr, Ynext_tr, Ystage_tr), ds(Xseq_va, Ynext_va, Ystage_va),
            scaler, torch.tensor(weights))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_sum = n = 0.0
    stage_correct = 0
    tp = fp = fn = 0
    for xb, yn, ys in loader:
        xb, yn, ys = xb.to(device), yn.to(device), ys.to(device)
        pred_state, stage_logits, _ = model(xb)
        mse_sum += torch.nn.functional.mse_loss(pred_state, yn, reduction="sum").item()
        n += yn.numel()
        stage_correct += (stage_logits.argmax(-1) == ys).sum().item()
        # binary infiltration detection (stage >= INITIAL_ACCESS)
        pred_inf = infiltration_prob(stage_logits) >= 0.5
        true_inf = ys >= INFILTRATION_STAGE
        tp += int((pred_inf & true_inf).sum()); fp += int((pred_inf & ~true_inf).sum())
        fn += int((~pred_inf & true_inf).sum())
    acc = stage_correct / len(loader.dataset)
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"dyn_mse": mse_sum / n, "stage_acc": acc, "infil_f1": f1, "precision": prec, "recall": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="", help="glob of CIC-IDS CSVs; omit for synthetic")
    ap.add_argument("--nrows", type=int, default=None, help="cap rows per CSV (fit free Colab)")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--window-seconds", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10, help="rollout horizon (stored in ckpt)")
    ap.add_argument("--encoder", choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-stage", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, val_ds, scaler, class_weights = build_dataset(args)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    cfg = WorldModelConfig(n_features=len(STATE_FEATURES), window=args.window, encoder=args.encoder)
    model = WorldModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = WorldModelLoss(lambda_stage=args.lambda_stage, class_weights=class_weights.to(device))
    print(f"model: {cfg.encoder}  params: {sum(p.numel() for p in model.parameters()):,}")

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); running = 0.0
        for xb, yn, ys in train_dl:
            xb, yn, ys = xb.to(device), yn.to(device), ys.to(device)
            pred_state, stage_logits, _ = model(xb)
            loss, parts = loss_fn(pred_state, stage_logits, yn, ys)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(xb)
        m = evaluate(model, val_dl, device)
        # infil_F1 is highly unstable epoch-to-epoch on this imbalanced task
        # (observed swinging from 0.01 to 0.40 within one run) — training
        # for a fixed epoch count and keeping only the LAST epoch means the
        # saved checkpoint is whatever that epoch's noise happened to land
        # on, not the model's actual best performance. Track and keep the
        # best-val-F1 epoch's weights instead.
        is_best = m["infil_f1"] > best_f1
        if is_best:
            best_f1 = m["infil_f1"]; best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:2d}  loss {running/len(train_ds):.4f}  "
              f"val: dyn_mse {m['dyn_mse']:.4f}  stage_acc {m['stage_acc']:.3f}  "
              f"infil_F1 {m['infil_f1']:.3f}{'  *best*' if is_best else ''}  ({time.time()-t0:.1f}s)")

    model.load_state_dict(best_state)
    print(f"\nrestored best checkpoint from epoch {best_epoch} (infil_F1 {best_f1:.3f})")

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "world_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": vars(cfg),
        "feature_names": STATE_FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "window": args.window, "k": args.k, "window_seconds": args.window_seconds,
        "best_epoch": best_epoch,
    }, ckpt)
    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ saved checkpoint → {ckpt}")
    print(f"✓ best-epoch val metrics: {evaluate(model, val_dl, device)}")


if __name__ == "__main__":
    main()
