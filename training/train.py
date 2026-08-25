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
    WorldModel, WorldModelConfig, WorldModelLoss, infiltration_prob, INFILTRATION_STAGE,
)


def load_flows(args):
    if args.data:
        import pandas as pd
        from foresight.data.cicids import load_cicids_csv
        files = sorted(glob.glob(args.data))
        if not files:
            raise SystemExit(f"no files match {args.data!r}")
        print(f"loading {len(files)} CIC-IDS CSV(s)…")
        df = pd.concat([load_cicids_csv(f, nrows=args.nrows) for f in files], ignore_index=True)
        print(f"  {len(df):,} flows, labels: {sorted(df.label.unique())[:8]}")
        return df
    print("no --data given → synthetic attack-progression traffic (smoke test)")
    return make_synthetic_flows(seed=args.seed, minutes=8.0)


def build_dataset(args):
    df = load_flows(args)
    X, y, ts = flows_to_state_windows(df, window_seconds=args.window_seconds)
    print(f"state windows: {len(X)}  features: {X.shape[1]}")

    Xv = X.values.astype(np.float32)
    split_w = max(args.window + 1, int(0.7 * len(Xv)))     # temporal split point (windows)

    scaler = StandardScaler().fit(Xv[:split_w])            # fit on TRAIN only (no leakage)
    Xs = scaler.transform(Xv).astype(np.float32)

    Xseq, Ynext, Ystage = make_sequences(Xs, y, window=args.window)
    if len(Xseq) < 8:
        raise SystemExit("too few sequences — lower --window or --window-seconds")

    split_seq = max(1, split_w - args.window)              # align seq split to the window split
    tr = slice(0, split_seq); va = slice(split_seq, None)
    print(f"sequences: {len(Xseq)}  train: {split_seq}  val: {len(Xseq) - split_seq}")

    def ds(sl):
        return TensorDataset(torch.tensor(Xseq[sl]), torch.tensor(Ynext[sl]),
                             torch.tensor(Ystage[sl], dtype=torch.long))
    return ds(tr), ds(va), scaler


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

    train_ds, val_ds, scaler = build_dataset(args)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    cfg = WorldModelConfig(n_features=len(STATE_FEATURES), window=args.window, encoder=args.encoder)
    model = WorldModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = WorldModelLoss(lambda_stage=args.lambda_stage)
    print(f"model: {cfg.encoder}  params: {sum(p.numel() for p in model.parameters()):,}")

    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); running = 0.0
        for xb, yn, ys in train_dl:
            xb, yn, ys = xb.to(device), yn.to(device), ys.to(device)
            pred_state, stage_logits, _ = model(xb)
            loss, parts = loss_fn(pred_state, stage_logits, yn, ys)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(xb)
        m = evaluate(model, val_dl, device)
        print(f"ep {ep:2d}  loss {running/len(train_ds):.4f}  "
              f"val: dyn_mse {m['dyn_mse']:.4f}  stage_acc {m['stage_acc']:.3f}  "
              f"infil_F1 {m['infil_f1']:.3f}  ({time.time()-t0:.1f}s)")

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "world_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": vars(cfg),
        "feature_names": STATE_FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "window": args.window, "k": args.k, "window_seconds": args.window_seconds,
    }, ckpt)
    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n✓ saved checkpoint → {ckpt}")
    print(f"✓ final val metrics: {evaluate(model, val_dl, device)}")


if __name__ == "__main__":
    main()
