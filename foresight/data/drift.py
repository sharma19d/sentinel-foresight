"""
Out-of-distribution guard for inference inputs.

A model asked to score traffic unlike anything it was trained on will still
return a confident number, and that number is meaningless. For a security
tool this failure mode is worse than useless — it is actively dangerous,
because "100% infiltration risk" reads identically whether the model is
certain or merely off the edge of its map.

This was not hypothetical: the project's own synthetic test-fixture traffic
sits ~53σ from the CIC-IDS-2018 training distribution on syn_ack_ratio, and
the trained model duly reported ~100% risk on *every* window of it, benign
ones included. This module exists so that shows up as a warning instead of a
convincing-looking chart.

The check is deliberately simple and assumption-light: how far, in training
standard deviations, does each feature of the input sit from the training
mean? The scaler statistics travel inside the checkpoint, so no training data
is needed at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foresight.data.state import STATE_FEATURES

# A scaled feature is "extreme" past this many training standard deviations.
# 6σ is far outside anything the model saw often enough to have learned, while
# still tolerating the genuinely heavy tails in network traffic (a real DDoS
# window legitimately sits several σ out on packet counts).
SIGMA_LIMIT = 6.0


@dataclass
class DriftReport:
    in_distribution: bool
    mean_abs_z: float                  # average |z| across features and windows
    worst_feature: str
    worst_z: float
    extreme_features: list[tuple[str, float]]   # (name, mean |z|) past SIGMA_LIMIT

    def message(self) -> str:
        if self.in_distribution:
            return (f"Input looks consistent with the training distribution "
                    f"(mean |z| {self.mean_abs_z:.1f}σ).")
        offenders = ", ".join(f"{n} ({z:.0f}σ)" for n, z in self.extreme_features[:4])
        return (f"⚠ This traffic is far outside the model's training distribution "
                f"— {offenders}. Forecasts on it are not trustworthy: the model is "
                f"extrapolating, and will tend to saturate at extreme values "
                f"regardless of what the traffic actually is.")


def check_drift(
    X_scaled: np.ndarray, sigma_limit: float = SIGMA_LIMIT,
) -> DriftReport:
    """
    X_scaled: [T, F] states already scaled with the CHECKPOINT's scaler
    (i.e. z-scores against the training mean/std). Returns a DriftReport.
    """
    if X_scaled.size == 0:
        return DriftReport(True, 0.0, "", 0.0, [])

    z = np.abs(np.asarray(X_scaled, dtype=np.float64))
    per_feature = z.mean(axis=0)                       # [F]
    worst_i = int(np.argmax(per_feature))
    names = STATE_FEATURES[:len(per_feature)]

    extreme = [(names[i], float(per_feature[i]))
               for i in np.argsort(-per_feature)
               if per_feature[i] > sigma_limit]

    return DriftReport(
        in_distribution=not extreme,
        mean_abs_z=float(per_feature.mean()),
        worst_feature=names[worst_i],
        worst_z=float(per_feature[worst_i]),
        extreme_features=extreme,
    )
