"""
Benchmark the Foresight world model against baselines it must beat.

Required by SIH PS 26153 (logistic-regression comparison), but the more
important question this answers is scientific rather than procedural:

  1. DYNAMICS — does the model actually learn P(S_t+1 | S_t..S_t-w), or is it
     just echoing the current state? The "persistence" baseline predicts
     next_state = last observed state. A world model that cannot beat
     persistence has not learned transition dynamics, and calling it a world
     model would be an overclaim. This is the headline check.

  2. DETECTION — does the temporal transformer beat a linear model on the
     SAME features, and does either beat trivially predicting the majority
     class? On a dataset this imbalanced, a majority-class predictor already
     scores ~76% accuracy, so accuracy alone proves nothing.

Everything reuses training.train.build_dataset, so the temporal split and the
train-only scaler fit are identical to training by construction — a
re-implementation here could silently drift and invalidate the comparison.

Run (on the machine that has the CSVs):
    python3 benchmark/baseline.py --data "data/cicids/*.csv" --nrows 400000 \
        --ckpt checkpoints/world_model_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.linear_model import LogisticRegression

from foresight.data.state import STATE_FEATURES
from foresight.model.world_model import (
    WorldModel, WorldModelConfig, infiltration_prob, INFILTRATION_STAGE,
)
from training.train import build_dataset


def binary_scores(pred: np.ndarray, true: np.ndarray) -> dict:
    """Precision/recall/F1/FPR for the infiltration (stage >= INITIAL_ACCESS) call."""
    pred, true = pred.astype(bool), true.astype(bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    # These must match the values the checkpoint was trained with, or the
    # comparison is meaningless — they define the val split being scored.
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--nrows", type=int, default=400000)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--window-seconds", type=float, default=1.0)
    ap.add_argument("--ckpt", type=str, default="checkpoints/world_model_best.pt")
    ap.add_argument("--out", type=str, default="benchmark/results.json")
    ap.add_argument("--seed", type=int, default=0)
    # build_dataset reads these but they don't affect the split:
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, scaler, _weights = build_dataset(args)
    Xtr, Ntr, Str = (t.numpy() for t in train_ds.tensors)
    Xva, Nva, Sva = (t.numpy() for t in val_ds.tensors)
    print(f"\ntrain {len(Xtr):,} sequences   val {len(Xva):,} sequences")

    true_inf = Sva >= INFILTRATION_STAGE
    print(f"val infiltration prevalence: {true_inf.mean():.1%} "
          f"({int(true_inf.sum()):,} of {len(true_inf):,} windows)")

    results = {}

    # ── 1. DYNAMICS: world model vs. persistence ────────────────────
    # Persistence = "the network next second looks like it does now". This is
    # a genuinely strong baseline for slow-moving traffic, which is exactly
    # why it's the honest bar for a dynamics model to clear.
    persist_pred = Xva[:, -1, :]                     # last observed state in each window
    persist_mse = float(((persist_pred - Nva) ** 2).mean())
    results["dynamics_persistence_mse"] = persist_mse
    print(f"\n[dynamics] persistence baseline MSE : {persist_mse:.4f}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = WorldModelConfig(**ckpt["config"])
    model = WorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Sanity: a checkpoint trained on different features/window can't be
    # scored against this split, and silently doing so would produce
    # confident nonsense.
    if ckpt.get("feature_names") != STATE_FEATURES:
        raise SystemExit("checkpoint feature list differs from current STATE_FEATURES")
    if ckpt.get("window") != args.window:
        raise SystemExit(f"checkpoint window {ckpt.get('window')} != --window {args.window}")

    preds, stage_logits = [], []
    with torch.no_grad():
        for i in range(0, len(Xva), 4096):
            xb = torch.tensor(Xva[i:i + 4096]).to(device)
            p, s, _ = model(xb)
            preds.append(p.cpu().numpy()); stage_logits.append(s.cpu())
    model_pred = np.concatenate(preds)
    logits = torch.cat(stage_logits)

    model_mse = float(((model_pred - Nva) ** 2).mean())
    results["dynamics_world_model_mse"] = model_mse
    gain = (persist_mse - model_mse) / persist_mse * 100
    print(f"[dynamics] world model MSE          : {model_mse:.4f}"
          f"   ({gain:+.1f}% vs persistence)")
    results["dynamics_improvement_pct"] = gain

    # ── 2. DETECTION: majority / logistic regression / world model ──
    maj = np.zeros_like(true_inf)                    # always predict "benign"
    results["detect_majority"] = binary_scores(maj, true_inf)

    # Logistic regression gets the SAME input the transformer gets: the full
    # flattened window, already scaled by the same train-fit scaler. Anything
    # less would be a straw-man baseline.
    ytr = (Str >= INFILTRATION_STAGE).astype(int)
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(Xtr.reshape(len(Xtr), -1), ytr)
    lr_pred = lr.predict(Xva.reshape(len(Xva), -1)).astype(bool)
    results["detect_logreg"] = binary_scores(lr_pred, true_inf)
    print(f"\n[detect] logistic regression fitted in {time.time()-t0:.1f}s")

    wm_pred = (infiltration_prob(logits).numpy() >= 0.5)
    results["detect_world_model"] = binary_scores(wm_pred, true_inf)

    rows = [("majority (always benign)", results["detect_majority"]),
            ("logistic regression", results["detect_logreg"]),
            ("world model", results["detect_world_model"])]
    print(f"\n{'model':<26} {'precision':>9} {'recall':>8} {'F1':>7} {'FPR':>8}")
    print("-" * 62)
    for name, m in rows:
        print(f"{name:<26} {m['precision']:>9.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>7.3f} {m['fpr']:>8.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ wrote {args.out}")


if __name__ == "__main__":
    main()
