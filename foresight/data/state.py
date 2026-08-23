"""
Network-state representation  S_t  (SIH PS 26153, representation "A").

The world model learns P(S_t+1 | S_t), so *everything* hinges on what S_t is.
Here S_t is a fixed-length, ordered feature vector describing ALL traffic in a
short time window [t, t+Δ): flag distributions, port activity, timing, volume.
A sequence of these vectors is what the model learns transition dynamics over.

This module is dataset-agnostic: it operates on a "canonical flow DataFrame"
(one row per flow, standard columns) that the dataset loaders produce, so the
same state definition serves CIC-IDS-2018, CTU-13, and PCAP-derived flows.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from foresight.mitre import AttackStage

# ── Canonical flow columns every loader must produce ────────────────
CANONICAL_COLUMNS = [
    "ts",                      # flow start, epoch seconds (float)
    "src_ip", "dst_ip",        # may be "" if the dataset dropped them
    "dst_port", "protocol",    # protocol: 6=TCP, 17=UDP
    "duration",                # seconds
    "bytes", "pkts",
    "syn", "ack", "fin", "rst", "psh", "urg",   # per-flow flag counts
    "iat_mean",                # mean inter-arrival time within the flow (s)
    "label", "stage",          # raw dataset label + mapped AttackStage (int)
]

# ── The state vector: ordered, fixed. Order is frozen — the model and the
#    explainability layer index features by this order, so never reorder it;
#    only append. ───────────────────────────────────────────────────
STATE_FEATURES = [
    "n_flows",
    "tot_bytes", "tot_pkts",
    "mean_duration", "std_duration",
    "mean_bytes_per_flow", "mean_pkts_per_flow",
    "syn", "ack", "fin", "rst", "psh", "urg",
    "syn_ack_ratio",           # SYN flood / half-open signature
    "n_unique_dst_ports",
    "port_scan_score",         # unique dst ports / flows — recon signature
    "n_unique_src", "n_unique_dst",   # fan-in / fan-out (0 if no IPs)
    "mean_iat", "std_iat",
    "frac_tcp", "frac_udp",
]
N_FEATURES = len(STATE_FEATURES)


def _aggregate_state(g: pd.DataFrame) -> list[float]:
    """Collapse all flows in one time window into the ordered state vector."""
    n = len(g)
    if n == 0:
        return [0.0] * N_FEATURES
    syn, ack = float(g["syn"].sum()), float(g["ack"].sum())
    uniq_ports = int(g["dst_port"].nunique())
    proto = g["protocol"]
    feats = {
        "n_flows": float(n),
        "tot_bytes": float(g["bytes"].sum()),
        "tot_pkts": float(g["pkts"].sum()),
        "mean_duration": float(g["duration"].mean()),
        "std_duration": float(g["duration"].std(ddof=0) or 0.0),
        "mean_bytes_per_flow": float(g["bytes"].mean()),
        "mean_pkts_per_flow": float(g["pkts"].mean()),
        "syn": syn, "ack": ack,
        "fin": float(g["fin"].sum()), "rst": float(g["rst"].sum()),
        "psh": float(g["psh"].sum()), "urg": float(g["urg"].sum()),
        "syn_ack_ratio": syn / (ack + 1.0),
        "n_unique_dst_ports": float(uniq_ports),
        "port_scan_score": uniq_ports / (n + 1.0),
        "n_unique_src": float(g["src_ip"].nunique()) if g["src_ip"].any() else 0.0,
        "n_unique_dst": float(g["dst_ip"].nunique()) if g["dst_ip"].any() else 0.0,
        "mean_iat": float(g["iat_mean"].mean()),
        "std_iat": float(g["iat_mean"].std(ddof=0) or 0.0),
        "frac_tcp": float((proto == 6).mean()),
        "frac_udp": float((proto == 17).mean()),
    }
    return [feats[k] for k in STATE_FEATURES]


def flows_to_state_windows(
    df: pd.DataFrame,
    window_seconds: float = 1.0,
    fill_gaps: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Turn a canonical flow DataFrame into a time-ordered sequence of state vectors.

    Returns
    -------
    X        : DataFrame [T, N_FEATURES]  — one state vector per window, in time order
    y_stage  : int array [T]              — the window's kill-chain stage (max over its flows)
    win_ts   : float array [T]            — window start time (epoch seconds)

    fill_gaps makes windows contiguous (empty windows become an all-zero
    "quiet network" state) so the model learns transitions over continuous time
    rather than jumping across gaps.
    """
    if df.empty:
        return pd.DataFrame(columns=STATE_FEATURES), np.array([], int), np.array([], float)

    df = df.sort_values("ts")
    t0 = float(df["ts"].min())
    win_idx = ((df["ts"] - t0) // window_seconds).astype(int)
    df = df.assign(_win=win_idx)

    grouped = {w: g for w, g in df.groupby("_win")}
    max_win = int(df["_win"].max())
    win_range = range(0, max_win + 1) if fill_gaps else sorted(grouped)

    rows, stages, times = [], [], []
    for w in win_range:
        g = grouped.get(w)
        if g is None:
            rows.append([0.0] * N_FEATURES)
            stages.append(int(AttackStage.BENIGN))
        else:
            rows.append(_aggregate_state(g))
            stages.append(int(g["stage"].max()))
        times.append(t0 + w * window_seconds)

    X = pd.DataFrame(rows, columns=STATE_FEATURES)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X, np.asarray(stages, dtype=int), np.asarray(times, dtype=float)


def make_sequences(
    X: np.ndarray, y_stage: np.ndarray, window: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised world-model training pairs from a state series.

    For each t: input = states[t-window : t]  →  targets:
        next_state = states[t]      (what the dynamics model predicts)
        next_stage = y_stage[t]     (kill-chain stage of that next state)

    Returns (Xseq [M, window, F], Ynext [M, F], Ystage [M]).
    """
    xs, yn, ys = [], [], []
    for t in range(window, len(X)):
        xs.append(X[t - window:t])
        yn.append(X[t])
        ys.append(y_stage[t])
    if not xs:
        F = X.shape[1] if X.ndim == 2 else N_FEATURES
        return (np.empty((0, window, F)), np.empty((0, F)), np.empty((0,), int))
    return np.asarray(xs, np.float32), np.asarray(yn, np.float32), np.asarray(ys, int)
