"""
Why did the model forecast this? — attribution for the infiltration call.

PS 26153 requires the forecast to be explainable, not just accurate. Two
complementary views, because they answer different questions:

  * WHICH FEATURES drove it — integrated gradients over the 22 state features
    ("the SYN/ACK ratio and port-scan score pushed this toward infiltration").
  * WHICH MOMENTS drove it — the transformer's temporal attention over the
    input window ("the model is reacting to what happened 2-3 seconds ago").

Integrated gradients (Sundararajan et al., 2017) is used rather than SHAP as
the default: it needs no extra dependency, is deterministic, and satisfies a
completeness axiom we can actually check at runtime — the attributions must
sum to F(input) - F(baseline). `Explanation.completeness_error` reports that
residual, so a caller can tell a trustworthy attribution from a broken one
instead of taking the numbers on faith. `shap_values()` is offered separately
for callers that specifically want SHAP and have the package installed.

The baseline is the all-zeros state *in scaled space*, which is the dataset
mean — i.e. "an average, unremarkable network". Attributions therefore read
as "what about this traffic differs from normal, and which way did it push".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from foresight.data.state import STATE_FEATURES
from foresight.mitre import AttackStage
from foresight.model.world_model import WorldModel, infiltration_prob


@dataclass
class Explanation:
    infiltration_prob: float          # the model's call being explained
    predicted_stage: int              # argmax kill-chain stage
    feature_attribution: np.ndarray   # [F] signed contribution per state feature
    timestep_attribution: np.ndarray  # [T] signed contribution per window position
    attention: np.ndarray | None      # [T] attention the final position paid to each step
    feature_names: list[str] = field(default_factory=lambda: list(STATE_FEATURES))
    completeness_error: float = 0.0   # |sum(attr) - (F(x) - F(baseline))|; small == trustworthy

    def top_features(self, n: int = 5) -> list[tuple[str, float]]:
        """The n features that moved the forecast most, by absolute contribution."""
        order = np.argsort(-np.abs(self.feature_attribution))[:n]
        return [(self.feature_names[i], float(self.feature_attribution[i])) for i in order]

    def summary(self, n: int = 3) -> str:
        """One-line plain-English rationale, for the demo UI and for analysts."""
        stage = AttackStage(self.predicted_stage).name.replace("_", " ").title()
        drivers = ", ".join(
            f"{name} ({'+' if v >= 0 else '−'}{abs(v):.2f})"
            for name, v in self.top_features(n)
        )
        peak = int(np.argmax(np.abs(self.timestep_attribution))) if len(self.timestep_attribution) else 0
        lag = len(self.timestep_attribution) - peak
        return (f"{self.infiltration_prob:.0%} infiltration risk, stage «{stage}». "
                f"Driven by {drivers}; most influenced by traffic ~{lag} window(s) ago.")


def _as_batch(window: torch.Tensor | np.ndarray, device) -> torch.Tensor:
    x = torch.as_tensor(window, dtype=torch.float32, device=device)
    return x.unsqueeze(0) if x.dim() == 2 else x       # [T,F] -> [1,T,F]


def _infiltration_scalar(model: WorldModel, x: torch.Tensor) -> torch.Tensor:
    _next_state, logits, _attn = model(x)
    return infiltration_prob(logits).sum()


def integrated_gradients(
    model: WorldModel, window: torch.Tensor | np.ndarray, steps: int = 64,
) -> tuple[np.ndarray, float]:
    """
    Attribute the infiltration probability across the [T, F] input window.

    Returns (attributions [T, F], completeness_error). Gradients are required
    here, so this deliberately does not run under torch.no_grad().
    """
    device = next(model.parameters()).device
    model.eval()
    x = _as_batch(window, device)
    baseline = torch.zeros_like(x)                     # scaled space -> dataset mean

    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        score = _infiltration_scalar(model, point)
        (grad,) = torch.autograd.grad(score, point)
        total_grad += grad
    attr = ((x - baseline) * (total_grad / steps)).squeeze(0).detach().cpu().numpy()

    # Completeness axiom: attributions should sum to the score difference.
    # A large residual means the attribution is not faithful (too few steps,
    # or a sharply non-smooth region) and should not be trusted as-is.
    with torch.no_grad():
        delta = float(_infiltration_scalar(model, x) - _infiltration_scalar(model, baseline))
    return attr, abs(float(attr.sum()) - delta)


@torch.no_grad()
def temporal_attention(model: WorldModel, window: torch.Tensor | np.ndarray) -> np.ndarray | None:
    """
    How much attention the final window position paid to each earlier position,
    averaged across layers. None for the LSTM encoder, which has no attention —
    callers fall back to timestep_attribution, which is encoder-agnostic.
    """
    device = next(model.parameters()).device
    model.eval()
    _next_state, _logits, attn = model(_as_batch(window, device))
    if attn is None:
        return None
    return attn[:, :, -1, :].mean(dim=1).squeeze(0).cpu().numpy()      # [B,L,T,T] -> [T]


def explain(
    model: WorldModel, window: torch.Tensor | np.ndarray, steps: int = 64,
) -> Explanation:
    """Full explanation for one window of observed (scaled) network states."""
    device = next(model.parameters()).device
    model.eval()
    x = _as_batch(window, device)

    with torch.no_grad():
        _next_state, logits, _attn = model(x)
        prob = float(infiltration_prob(logits).item())
        stage = int(logits.argmax(dim=-1).item())

    attr, comp_err = integrated_gradients(model, window, steps=steps)
    return Explanation(
        infiltration_prob=prob,
        predicted_stage=stage,
        feature_attribution=attr.sum(axis=0),      # [T,F] -> per-feature
        timestep_attribution=attr.sum(axis=1),     # [T,F] -> per-timestep
        attention=temporal_attention(model, window),
        completeness_error=comp_err,
    )


def shap_values(model: WorldModel, window, background, nsamples: int = 100):
    """
    SHAP attributions, for callers that specifically want them.

    Kept optional and separate: shap is a heavy dependency, and its gradient
    explainers are stochastic, so `explain()` above stays the reproducible
    default. `background` should be a [N, T, F] sample of typical windows.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is not installed — `pip install shap`, or use explain()") from e

    device = next(model.parameters()).device
    model.eval()

    class _InfiltrationHead(torch.nn.Module):
        """shap explains a scalar output; expose infiltration prob as that scalar."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            _s, logits, _a = self.inner(x)
            return infiltration_prob(logits).unsqueeze(-1)

    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(_InfiltrationHead(model), bg)
    return explainer.shap_values(_as_batch(window, device), nsamples=nsamples)
