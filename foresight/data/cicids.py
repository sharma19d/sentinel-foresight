"""
CIC-IDS-2018 (CSE-CIC-IDS2018) CSV loader → canonical flow DataFrame.

The dataset's CSVs come from CICFlowMeter and have well-known quirks: column
names with stray spaces/casing, Inf/NaN in rate columns, and (in most public
copies) no source/dest IP. This loader normalises all of that into the canonical
schema `state.py` expects, and maps the per-flow Label to a kill-chain stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.mitre import stage_from_label

# Map canonical field -> the possible CICFlowMeter column names (lowercased,
# stripped) it can appear under. First match wins.
_COLMAP = {
    "ts":       ["timestamp"],
    "dst_port": ["dst port", "destination port"],
    "protocol": ["protocol"],
    "duration": ["flow duration"],
    "tot_fwd":  ["tot fwd pkts", "total fwd packets"],
    "tot_bwd":  ["tot bwd pkts", "total backward packets"],
    "len_fwd":  ["totlen fwd pkts", "total length of fwd packets"],
    "len_bwd":  ["totlen bwd pkts", "total length of bwd packets"],
    "iat_mean": ["flow iat mean"],
    "syn":      ["syn flag cnt", "syn flag count"],
    "ack":      ["ack flag cnt", "ack flag count"],
    "fin":      ["fin flag cnt", "fin flag count"],
    "rst":      ["rst flag cnt", "rst flag count"],
    "psh":      ["psh flag cnt", "psh flag count"],
    "urg":      ["urg flag cnt", "urg flag count"],
    "src_ip":   ["src ip", "source ip"],
    "dst_ip":   ["dst ip", "destination ip"],
    "label":    ["label"],
}


def _resolve(cols: list[str]) -> dict[str, str]:
    """Map canonical name -> actual column name present in this CSV."""
    norm = {c.strip().lower(): c for c in cols}
    found = {}
    for canon, variants in _COLMAP.items():
        for v in variants:
            if v in norm:
                found[canon] = norm[v]
                break
    return found


def load_cicids_csv(
    path: str, nrows: int | None = None, require_label: bool = True,
) -> pd.DataFrame:
    """
    Load one CIC-IDS-2018 CSV into the canonical flow schema.

    Duration is reported by CICFlowMeter in microseconds → converted to seconds.
    Missing IP columns are filled with "" (state.py degrades gracefully).

    Reads only the ~14 columns this loader actually uses, via `usecols` —
    CICFlowMeter CSVs carry ~80 columns, and reading all of them (as an
    earlier version did) inflates per-file memory ~6x for no benefit; across
    10 files that compounds (freed pandas/glibc memory doesn't fully return
    to the OS between files) until the process gets OOM-killed.

    require_label=False allows loading an UNLABELLED capture: real traffic
    being scored at inference time has no ground-truth Label column, so the
    demo path must not require one. Labels are then blank and every stage is
    BENIGN — placeholders that inference ignores, never to be read as truth.
    """
    header_cols = list(pd.read_csv(path, nrows=0).columns)
    col = _resolve(header_cols)
    if "label" not in col and require_label:
        raise ValueError(f"{path}: not a recognised CIC-IDS CSV (no Label column)")

    raw = pd.read_csv(path, nrows=nrows, low_memory=False, usecols=list(col.values()))

    # Several public CIC-IDS-2018 CSVs were produced by concatenating capture
    # chunks and carry the header row repeated mid-file (Label == "Label").
    # Left in, every numeric column of those rows coerces to 0 and the stage
    # mapper reads "Label" as BENIGN — i.e. they become fabricated all-zero
    # benign flows in both training and evaluation.
    if "label" in col:
        header_rows = raw[col["label"]].astype(str).str.strip() == col["label"]
        if header_rows.any():
            print(f"{path}: dropping {int(header_rows.sum())} repeated header row(s)")
            raw = raw[~header_rows].reset_index(drop=True)

    def num(canon: str, default=0.0) -> pd.Series:
        if canon in col:
            return pd.to_numeric(raw[col[canon]], errors="coerce").fillna(default)
        return pd.Series(default, index=raw.index, dtype=float)

    if "ts" in col:
        # Force nanosecond resolution before the int64 cast. pandas >= 2.0
        # infers the unit from the data and may return datetime64[us] or
        # [s]; .astype("int64") then yields microseconds or seconds, and
        # dividing by 1e9 silently produced timestamps 1e3 (or 1e9) times
        # too small — collapsing a multi-hour capture into a few seconds,
        # which in turn made every state window aggregate thousands of flows.
        # The bug is version-dependent: correct on pandas with ns default,
        # wrong on newer installs, so it must not be left implicit.
        ts_epoch = (
            pd.to_datetime(raw[col["ts"]], errors="coerce", dayfirst=True)
            .astype("datetime64[ns]").astype("int64") / 1e9
        )
    else:
        # No Timestamp column — use row order as pseudo-time (CICFlowMeter output
        # is roughly time-ordered; the world model learns transition order, which
        # is preserved). One tick per flow; set --window-seconds to flows-per-window.
        print(f"{path}: no Timestamp column -> using row order as pseudo-time")
        ts_epoch = pd.Series(range(len(raw)), dtype=float)
    duration_us = num("duration")

    out = pd.DataFrame({
        "ts": ts_epoch,                                    # epoch seconds (or row-order ticks)
        "src_ip": raw[col["src_ip"]].astype(str) if "src_ip" in col else "",
        "dst_ip": raw[col["dst_ip"]].astype(str) if "dst_ip" in col else "",
        "dst_port": num("dst_port").astype(int),
        "protocol": num("protocol").astype(int),
        "duration": duration_us / 1e6,                    # µs → s
        "bytes": num("len_fwd") + num("len_bwd"),
        "pkts": num("tot_fwd") + num("tot_bwd"),
        "syn": num("syn"), "ack": num("ack"), "fin": num("fin"),
        "rst": num("rst"), "psh": num("psh"), "urg": num("urg"),
        "iat_mean": (num("iat_mean") / 1e6),              # µs → s
        "label": raw[col["label"]].astype(str) if "label" in col else "",
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ts"]).reset_index(drop=True)

    # Some public CIC-IDS-2018 CSVs carry a handful of rows with a garbage
    # Timestamp (e.g. epoch ~1970 instead of the file's actual 2018 date).
    # A single such row makes the file's apparent time span ~1.5 BILLION
    # seconds instead of a few hours; flows_to_state_windows()'s fill_gaps
    # then tries to allocate one zero-row per elapsed second across that
    # entire span, which OOMs the process. Drop rows whose timestamp is far
    # from the file's median before any windowing sees them.
    if len(out) and "ts" in col:
        median_ts = out["ts"].median()
        keep = (out["ts"] - median_ts).abs() <= 2 * 86400  # within 2 days of the file's typical timestamp
        n_bad = int((~keep).sum())
        if n_bad:
            print(f"{path}: dropping {n_bad} row(s) with a corrupted Timestamp far outside the file's date")
            out = out[keep].reset_index(drop=True)

    out["stage"] = out["label"].map(stage_from_label).astype(int)
    return out
