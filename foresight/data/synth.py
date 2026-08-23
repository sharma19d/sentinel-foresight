"""
Synthetic attack-progression generator.

Fabricates a canonical flow DataFrame with a realistic kill-chain over time:
quiet benign traffic → a reconnaissance port-scan burst → an SSH brute-force →
C2 beaconing. Lets us validate the whole state/sequence pipeline (and later,
smoke-test the model) BEFORE the real CIC-IDS-2018 download is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_flows(seed: int = 0, minutes: float = 6.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0

    def emit(n, label, stage, *, ports, syn=0, ack=1, proto=6, dur=0.05, byts=800, pkts=6, dst="10.0.0.5"):
        nonlocal t
        for _ in range(n):
            t += float(rng.exponential(0.05))
            rows.append(dict(
                ts=t, src_ip="10.0.0.9", dst_ip=dst,
                dst_port=int(rng.choice(ports)), protocol=proto,
                duration=abs(rng.normal(dur, dur / 3)),
                bytes=abs(rng.normal(byts, byts / 3)), pkts=max(1, int(rng.normal(pkts, 2))),
                syn=syn, ack=ack, fin=0, rst=0, psh=int(rng.integers(0, 2)), urg=0,
                iat_mean=abs(rng.normal(0.02, 0.01)),
                label=label, stage=stage,
            ))

    # 0: benign background
    emit(400, "Benign", 0, ports=[80, 443, 53], syn=1, ack=3, byts=1500, pkts=12)
    # 1: reconnaissance — many unique ports, SYN-heavy, tiny flows
    emit(300, "PortScan", 1, ports=list(range(1, 1024)), syn=1, ack=0, dur=0.005, byts=60, pkts=1)
    # 2: initial access — brute force one port, repeated
    emit(200, "SSH-Bruteforce", 2, ports=[22], syn=1, ack=1, byts=400, pkts=8)
    # 3: C2 — periodic beacon to one port
    emit(150, "Bot", 4, ports=[8080], syn=1, ack=1, byts=300, pkts=5)
    # trailing benign
    emit(150, "Benign", 0, ports=[80, 443], syn=1, ack=3, byts=1500, pkts=12)

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df
