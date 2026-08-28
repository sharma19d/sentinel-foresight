"""
The world model:  learns  P(S_t+1 | S_t..S_t-w)  — transition dynamics, not labels.

This is the core deliverable of SIH PS 26153. Given a window of network-state
vectors, it predicts the NEXT state vector (dynamics), and reads off the
kill-chain stage of that predicted future. Because it predicts the full next
state, it supports K-step forward simulation (see foresight.rollout): roll the
prediction forward and watch whether the trajectory converges to infiltration
BEFORE the attacker completes the kill chain.

Two encoders are supported:
  * 'transformer' (default) — temporal self-attention; the attention weights are
    returned for explainability ("which past windows drove this forecast").
  * 'lstm' — a recurrent baseline; explainability then comes from SHAP/gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from foresight.mitre import AttackStage

N_STAGES = len(AttackStage)
INFILTRATION_STAGE = int(AttackStage.INITIAL_ACCESS)  # stages >= this = compromised


@dataclass
class WorldModelConfig:
    n_features: int = 22
    window: int = 16
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dim_ff: int = 192
    dropout: float = 0.1
    n_stages: int = N_STAGES
    encoder: str = "transformer"   # 'transformer' | 'lstm'


class _EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer that returns its attention weights."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.ff(self.n2(x))
        return x, w                        # w: [B, T, T] attention over the window


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)

        if cfg.encoder == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, cfg.window, cfg.d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.layers = nn.ModuleList(
                _EncoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            )
            self.norm = nn.LayerNorm(cfg.d_model)
        elif cfg.encoder == "lstm":
            self.lstm = nn.LSTM(cfg.d_model, cfg.d_model, num_layers=cfg.n_layers,
                                batch_first=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        else:
            raise ValueError(f"unknown encoder {cfg.encoder!r}")

        # Two heads on the final-timestep representation:
        self.dynamics_head = nn.Linear(cfg.d_model, cfg.n_features)   # predict S_t+1
        self.stage_head = nn.Linear(cfg.d_model, cfg.n_stages)        # its kill-chain stage

    def forward(self, x):
        """
        x: [B, window, F].  Returns:
          next_state  [B, F]        — predicted S_t+1
          stage_logits[B, n_stages] — kill-chain stage of the predicted state
          attn        [B, L, T, T] or None — per-layer temporal attention
        """
        h = self.input_proj(x)
        attn = None
        if self.cfg.encoder == "transformer":
            h = h + self.pos[:, : x.size(1)]
            ws = []
            for layer in self.layers:
                h, w = layer(h)
                ws.append(w)
            h = self.norm(h)
            attn = torch.stack(ws, dim=1)          # [B, L, T, T]
            last = h[:, -1]
        else:
            out, _ = self.lstm(h)
            last = out[:, -1]

        return self.dynamics_head(last), self.stage_head(last), attn


def infiltration_prob(stage_logits: torch.Tensor) -> torch.Tensor:
    """
    P(network is at/after Initial Access) — the infiltration probability.
    Sums softmax mass over kill-chain stages >= INITIAL_ACCESS.  [B] in [0,1].
    """
    p = torch.softmax(stage_logits, dim=-1)
    return p[..., INFILTRATION_STAGE:].sum(dim=-1)


class WorldModelLoss(nn.Module):
    """
    Joint objective:
      * dynamics — MSE between predicted and true next state (the world-model core)
      * stage    — cross-entropy on the next kill-chain stage
    lambda_stage balances them.
    """

    def __init__(self, lambda_stage: float = 1.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.lambda_stage = lambda_stage

    def forward(self, pred_state, pred_stage_logits, true_state, true_stage):
        dyn = self.mse(pred_state, true_state)
        stg = self.ce(pred_stage_logits, true_stage)
        return dyn + self.lambda_stage * stg, {"dynamics": dyn.item(), "stage": stg.item()}
