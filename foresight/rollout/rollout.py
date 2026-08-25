"""
K-step forward simulation — the world model's defining capability.

Given the current window of observed network states, we don't just classify the
present: we roll the learned dynamics forward K steps, feeding each predicted
state back in, to forecast where the network is heading. At every predicted
step we read off the infiltration probability and the predicted MITRE stage,
producing the "infiltration probability timeline" the PS 26153 demo shows.
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
def rollout(model: WorldModel, seed_window: torch.Tensor, k_steps: int = 10) -> Forecast:
    """
    seed_window: [window, F] tensor of the most recent observed states (scaled).
    Rolls K steps ahead by feeding predictions back into the input window.
    """
    model.eval()
    device = next(model.parameters()).device
    window = seed_window.to(device).clone()

    probs, stages, states = [], [], []
    for _ in range(k_steps):
        next_state, stage_logits, _ = model(window.unsqueeze(0))   # [1,F], [1,S]
        p = float(infiltration_prob(stage_logits).item())
        stage = int(stage_logits.argmax(dim=-1).item())
        probs.append(p); stages.append(stage); states.append(next_state.squeeze(0).cpu().numpy())
        # slide the window forward with the predicted state
        window = torch.cat([window[1:], next_state], dim=0)

    probs = np.asarray(probs, dtype=float)
    stages = np.asarray(stages, dtype=int)
    return Forecast(
        infiltration_prob=probs,
        stages=stages,
        states=np.asarray(states, dtype=float),
        horizon_prob=float(probs.max()) if len(probs) else 0.0,
        predicted_stage=int(stages.max()) if len(stages) else int(AttackStage.BENIGN),
    )


@torch.no_grad()
def forecast_series(model: WorldModel, X_scaled: np.ndarray, window: int, k_steps: int = 10):
    """
    Slide over a whole capture and, at each position t, forecast K steps ahead —
    producing the timeline of "infiltration probability predicted at time t".
    Returns arrays aligned to t = window .. len(X): (times_idx, horizon_prob, predicted_stage).
    """
    device = next(model.parameters()).device
    idx, hprob, hstage = [], [], []
    for t in range(window, len(X_scaled) + 1):
        seed = torch.tensor(X_scaled[t - window:t], dtype=torch.float32, device=device)
        fc = rollout(model, seed, k_steps=k_steps)
        idx.append(t - 1); hprob.append(fc.horizon_prob); hstage.append(fc.predicted_stage)
    return np.asarray(idx), np.asarray(hprob), np.asarray(hstage)
