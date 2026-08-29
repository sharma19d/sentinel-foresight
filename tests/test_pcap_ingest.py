"""
Regression tests for PCAP ingest.

Run:  .venv/bin/python tests/test_pcap_ingest.py

No pytest dependency — plain asserts, so this runs anywhere the demo runs.
Both tests below guard failures that actually happened in this project:
unit-conversion asymmetry between ingest paths, and features that look
plausible but mean something different from what the model was trained on.
"""
import os
import sys
import datetime as dt
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight.data.cicids import load_cicids_csv
from foresight.data.pcap import load_pcap


def build_pcap(path: str) -> None:
    """A capture whose every field is known by construction."""
    from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

    pkts, t0 = [], 1700000000.0

    def add(p, t):
        p.time = t
        pkts.append(p)

    a_ip, a_pt, b_ip, b_pt = "10.0.0.5", 44444, "10.0.0.9", 80
    fwd = lambda **k: Ether() / IP(src=a_ip, dst=b_ip) / TCP(sport=a_pt, dport=b_pt, **k)
    bwd = lambda **k: Ether() / IP(src=b_ip, dst=a_ip) / TCP(sport=b_pt, dport=a_pt, **k)

    add(fwd(flags="S"), t0 + 0.0)
    add(bwd(flags="SA"), t0 + 0.1)
    add(fwd(flags="A"), t0 + 0.2)
    add(fwd(flags="PA") / Raw(b"x" * 100), t0 + 0.3)
    add(bwd(flags="A"), t0 + 0.4)
    add(fwd(flags="FA"), t0 + 0.5)
    add(bwd(flags="FA"), t0 + 0.6)

    udp = Ether() / IP(src=a_ip, dst="10.0.0.255") / UDP(sport=5353, dport=5353)
    add(udp / Raw(b"y" * 50), t0 + 1.0)
    add(udp / Raw(b"y" * 50), t0 + 1.4)

    # Same 5-tuple as the first flow but past the timeout -> must be its own row.
    add(fwd(flags="S"), t0 + 400.0)
    # ICMP: no ports, and frac_tcp/frac_udp only know 6 and 17 -> must be ignored.
    add(Ether() / IP(src=a_ip, dst=b_ip, proto=1), t0 + 2.0)

    wrpcap(path, pkts)


def test_flow_assembly(pcap_path: str) -> pd.DataFrame:
    df = load_pcap(pcap_path)
    assert len(df) == 3, f"expected 3 flows (TCP, UDP, post-timeout TCP), got {len(df)}"

    tcp = df[df.protocol == 6].iloc[0]
    assert tcp.pkts == 7, f"packet count {tcp.pkts}"
    # CICFlowMeter's TotLen counts L4 PAYLOAD; summing frame length here would
    # inflate this several-fold on ACK-heavy traffic and shift the feature
    # out of the distribution the model was trained on.
    assert tcp.bytes == 100, f"payload bytes {tcp.bytes} (must exclude headers)"
    assert tcp.syn == 2, f"syn {tcp.syn}"      # SYN + SYN/ACK
    assert tcp.ack == 6, f"ack {tcp.ack}"
    assert tcp.fin == 2, f"fin {tcp.fin}"
    assert tcp.psh == 1, f"psh {tcp.psh}"
    assert abs(tcp.duration - 0.6) < 1e-6, f"duration {tcp.duration}"
    assert abs(tcp.iat_mean - 0.1) < 1e-6, f"iat_mean {tcp.iat_mean}"

    udp = df[df.protocol == 17].iloc[0]
    assert udp.pkts == 2 and udp.bytes == 100, f"udp {udp.pkts} {udp.bytes}"
    assert abs(udp.iat_mean - 0.4) < 1e-6, f"udp iat {udp.iat_mean}"

    assert (df.stage == 0).all() and (df.label == "").all(), "unlabelled placeholders"
    assert df.ts.is_monotonic_increasing, "rows must be time-ordered"
    print("✓ flow assembly: counts, payload accounting, flags, IAT, timeout split")
    return df


def test_csv_pcap_equivalence(df: pd.DataFrame) -> None:
    """The two ingest paths must agree on identical traffic.

    Guards unit asymmetry: CICFlowMeter reports duration and IAT in
    MICROSECONDS while PCAP timestamps are seconds. A regression on either
    side shows up here as a 1e6 discrepancy rather than as a silently
    mis-scaled feature at inference time.
    """
    rows = [{
        "Timestamp": dt.datetime.fromtimestamp(r.ts, dt.timezone.utc).strftime("%d/%m/%Y %H:%M:%S"),
        "Dst Port": int(r.dst_port), "Protocol": int(r.protocol),
        "Flow Duration": r.duration * 1e6,
        "Tot Fwd Pkts": r.pkts, "Tot Bwd Pkts": 0,
        "TotLen Fwd Pkts": r.bytes, "TotLen Bwd Pkts": 0,
        "Flow IAT Mean": r.iat_mean * 1e6,
        "FIN Flag Cnt": r.fin, "SYN Flag Cnt": r.syn, "RST Flag Cnt": r.rst,
        "PSH Flag Cnt": r.psh, "ACK Flag Cnt": r.ack, "URG Flag Cnt": r.urg,
        "Label": "Benign",
    } for _, r in df.iterrows()]

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        pd.DataFrame(rows).to_csv(fh.name, index=False)
        csv_path = fh.name
    try:
        c = load_cicids_csv(csv_path)
    finally:
        os.unlink(csv_path)

    for col in ["dst_port", "protocol", "duration", "bytes", "pkts",
                "syn", "ack", "fin", "rst", "psh", "urg", "iat_mean"]:
        a, b = df[col].to_numpy(float), c[col].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9), f"{col}: pcap {a} != csv {b}"
    print("✓ equivalence: PCAP and CSV ingest agree on identical traffic")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.pcap")
        build_pcap(p)
        test_csv_pcap_equivalence(test_flow_assembly(p))
    print("\nALL PCAP TESTS PASSED")
