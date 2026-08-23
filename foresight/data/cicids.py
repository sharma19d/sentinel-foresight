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


def load_cicids_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load one CIC-IDS-2018 CSV into the canonical flow schema.

    Duration is reported by CICFlowMeter in microseconds → converted to seconds.
    Missing IP columns are filled with "" (state.py degrades gracefully).
    """
    raw = pd.read_csv(path, nrows=nrows, low_memory=False)
    col = _resolve(list(raw.columns))
    if "label" not in col or "ts" not in col:
        raise ValueError(f"{path}: not a recognised CIC-IDS CSV (missing Label/Timestamp)")

    def num(canon: str, default=0.0) -> pd.Series:
        if canon in col:
            return pd.to_numeric(raw[col[canon]], errors="coerce").fillna(default)
        return pd.Series(default, index=raw.index, dtype=float)

    ts = pd.to_datetime(raw[col["ts"]], errors="coerce")
    duration_us = num("duration")

    out = pd.DataFrame({
        "ts": ts.view("int64") / 1e9,                     # epoch seconds
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
        "label": raw[col["label"]].astype(str),
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ts"]).reset_index(drop=True)
    out["stage"] = out["label"].map(stage_from_label).astype(int)
    return out
