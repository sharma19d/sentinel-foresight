"""
K-step forward simulation — the world model's defining capability.

Given the current window of observed network states, we don't just classify the
present: we roll the learned dynamics forward K steps, feeding each predicted
state back in, to forecast where the network is heading. At every predicted
step we read off the infiltration probability and the predicted MITRE stage,
producing the "infiltration probability timeline" the PS 26153 demo shows.

Rollouts are batched internally: scoring a whole capture means one rollout per
window position, and doing those one-at-a-time is hundreds of thousands of
single-sample forward passes — far too slow for an interactive demo on CPU.
`rollout()` keeps the simple single-window API and delegates to the batched path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from foresight.model.world_model import WorldModel, infiltration_prob
from foresight.mitre import AttackStage


@dataclass
class Forecast:
    infiltration_prob: np.ndarray   # [K] probability at each future step
    stages: np.ndarray              # [K] predicted AttackStage (int) per step
    states: np.ndarray              # [K, F] predicted future state vectors
    horizon_prob: float             # max infiltration prob across the horizon
    predicted_stage: int            # worst (max) predicted stage across horizon


@torch.no_grad()
def rollout_batch(
    model: WorldModel, seed_windows: torch.Tensor, k_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll many windows forward at once.

    seed_windows: [B, window, F] of recent observed states (already scaled).
    Returns (probs [B, K], stages [B, K], states [B, K, F]).
    """
    model.eval()
    device = next(model.parameters()).device
    win = seed_windows.to(device, dtype=torch.float32).clone()

    probs, stages, states = [], [], []
    for _ in range(k_steps):
        next_state, stage_logits, _ = model(win)              # [B,F], [B,S]
        probs.append(infiltration_prob(stage_logits).cpu().numpy())
        stages.append(stage_logits.argmax(dim=-1).cpu().numpy())
        states.append(next_state.cpu().numpy())
        # slide each window forward by one, appending its own prediction
        win = torch.cat([win[:, 1:], next_state.unsqueeze(1)], dim=1)

    return (np.stack(probs, axis=1), np.stack(stages, axis=1),
            np.stack(states, axis=1))


def rollout(model: WorldModel, seed_window: torch.Tensor, k_steps: int = 10) -> Forecast:
    """
    seed_window: [window, F] tensor of the most recent observed states (scaled).
    Rolls K steps ahead by feeding predictions back into the input window.
    """
    x = torch.as_tensor(seed_window)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    probs, stages, states = rollout_batch(model, x, k_steps=k_steps)
    probs, stages, states = probs[0], stages[0], states[0]
    return Forecast(
        infiltration_prob=probs.astype(float),
        stages=stages.astype(int),
        states=states.astype(float),
        horizon_prob=float(probs.max()) if len(probs) else 0.0,
        predicted_stage=int(stages.max()) if len(stages) else int(AttackStage.BENIGN),
    )


def sliding_windows(X_scaled: np.ndarray, window: int) -> np.ndarray:
    """[T, F] -> [T-window+1, window, F] as a view (no copy)."""
    if len(X_scaled) < window:
        return np.empty((0, window, X_scaled.shape[1]), dtype=X_scaled.dtype)
    return np.lib.stride_tricks.sliding_window_view(
        X_scaled, window, axis=0
    ).transpose(0, 2, 1)


def forecast_series(
    model: WorldModel,
    X_scaled: np.ndarray,
    window: int,
    k_steps: int = 10,
    batch_size: int = 512,
    stride: int = 1,
):
    """
    Slide over a whole capture and, at each position t, forecast K steps ahead —
    producing the timeline of "infiltration probability predicted at time t".

    `stride` subsamples window positions (stride=5 scores every 5th window),
    which is how an interactive demo stays responsive on a long capture without
    changing what any individual forecast means.

    Returns (idx, horizon_prob, predicted_stage), aligned so idx[i] is the index
    of the last *observed* window feeding forecast i.
    """
    seeds = sliding_windows(X_scaled, window)[::stride]
    if not len(seeds):
        return np.empty(0, int), np.empty(0, float), np.empty(0, int)

    hprob, hstage = [], []
    for i in range(0, len(seeds), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(seeds[i:i + batch_size]))
        probs, stages, _ = rollout_batch(model, chunk, k_steps=k_steps)
        hprob.append(probs.max(axis=1))          # worst case across the horizon
        hstage.append(stages.max(axis=1))

    idx = np.arange(len(seeds)) * stride + (window - 1)
    return idx, np.concatenate(hprob), np.concatenate(hstage)
