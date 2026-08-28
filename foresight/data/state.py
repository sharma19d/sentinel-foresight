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

    Implementation note: this is vectorised over pandas groupby aggregates
    (not a per-window Python loop) — with window_seconds=1.0 a single day's
    capture produces on the order of 10^5 windows, and a Python-level loop
    at that scale is what made early runs of this look hung.
    """
    if df.empty:
        return pd.DataFrame(columns=STATE_FEATURES), np.array([], int), np.array([], float)

    df = df.sort_values("ts").copy()
    t0 = float(df["ts"].min())
    df["_win"] = ((df["ts"] - t0) // window_seconds).astype(int)
    df["_is_tcp"] = (df["protocol"] == 6).astype(float)
    df["_is_udp"] = (df["protocol"] == 17).astype(float)

    g = df.groupby("_win")
    sums = g[["bytes", "pkts", "syn", "ack", "fin", "rst", "psh", "urg"]].sum()
    means = g[["duration", "bytes", "pkts", "iat_mean", "_is_tcp", "_is_udp"]].mean()
    stds = g[["duration", "iat_mean"]].std(ddof=0).fillna(0.0)
    n_flows = g.size().astype(float)
    n_ports = g["dst_port"].nunique().astype(float)
    stage_max = g["stage"].max()

    # IP columns are "" for every row when the dataset doesn't carry IPs at
    # all (true for the public CIC-IDS-2018 CSVs); treat that dataset-wide
    # rather than re-checking non-emptiness per window (matches practice —
    # a loader either has IPs for every flow or none).
    n_src = g["src_ip"].nunique().astype(float) if not (df["src_ip"] == "").all() else pd.Series(0.0, index=sums.index)
    n_dst = g["dst_ip"].nunique().astype(float) if not (df["dst_ip"] == "").all() else pd.Series(0.0, index=sums.index)

    agg = pd.DataFrame({
        "n_flows": n_flows,
        "tot_bytes": sums["bytes"], "tot_pkts": sums["pkts"],
        "mean_duration": means["duration"], "std_duration": stds["duration"],
        "mean_bytes_per_flow": means["bytes"], "mean_pkts_per_flow": means["pkts"],
        "syn": sums["syn"], "ack": sums["ack"], "fin": sums["fin"], "rst": sums["rst"],
        "psh": sums["psh"], "urg": sums["urg"],
        "n_unique_dst_ports": n_ports,
        "n_unique_src": n_src, "n_unique_dst": n_dst,
        "mean_iat": means["iat_mean"], "std_iat": stds["iat_mean"],
        "frac_tcp": means["_is_tcp"], "frac_udp": means["_is_udp"],
    })
    agg["syn_ack_ratio"] = agg["syn"] / (agg["ack"] + 1.0)
    agg["port_scan_score"] = agg["n_unique_dst_ports"] / (agg["n_flows"] + 1.0)

    max_win = int(df["_win"].max())
    if fill_gaps and max_win > 20_000_000:
        # A handful of malformed timestamps (seen in real CIC-IDS-2018 CSVs —
        # e.g. one row stamped ~1970 instead of the file's actual date) can
        # inflate the apparent time span from hours to decades. fill_gaps
        # would then try to allocate one row per elapsed second across that
        # whole span — fail loudly here instead of silently exhausting RAM.
        raise ValueError(
            f"flows_to_state_windows: {max_win + 1:,} windows requested "
            f"(window_seconds={window_seconds}) — this almost always means "
            f"the input has an outlier timestamp inflating the time span; "
            f"check df['ts'].describe() before calling this."
        )
    full_range = range(0, max_win + 1) if fill_gaps else sorted(agg.index)

    X = agg.reindex(full_range, fill_value=0.0)[STATE_FEATURES]
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    stages = stage_max.reindex(full_range, fill_value=int(AttackStage.BENIGN)).astype(int).to_numpy()
    times = t0 + np.asarray(full_range, dtype=float) * window_seconds
    return X.reset_index(drop=True), stages, times


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
