"""
PCAP → canonical flow DataFrame.

PS 26153 asks for a demo accepting "PCAP or CSV". The CSV path (cicids.py)
consumes CICFlowMeter output; this module reassembles the same flow-level
features directly from packets, so a raw capture can be scored by the same
trained model.

**The hard constraint is distributional, not functional.** The model was
trained on CICFlowMeter-derived features, so these must mean the *same thing*
CICFlowMeter means or the model sees out-of-distribution input and produces
confident nonsense (exactly the failure `foresight/data/drift.py` was written
to catch). Choices below therefore follow CICFlowMeter's actual behaviour
rather than what might seem more natural:

  * A flow is **bidirectional**, keyed on the unordered 5-tuple; the first
    packet seen fixes the forward direction.
  * `bytes` sums **L4 payload** bytes, not frame or IP-total length —
    CICFlowMeter's TotLen Fwd/Bwd Pkts are payload sums, so counting headers
    would inflate this feature several-fold on ACK-heavy traffic.
  * `pkts` counts **all** packets including pure ACKs with no payload.
  * Flag counts are per-flow packet counts with that TCP flag set, summed over
    both directions (matching SYN/ACK/... Flag Cnt).
  * `iat_mean` is the mean gap between consecutive packets in the flow,
    either direction (Flow IAT Mean), in seconds.
  * Flows expire after `flow_timeout` seconds of inactivity (CICFlowMeter's
    default is 120 s) and TCP flows also close on RST, or on FIN seen from
    both sides. A long-lived connection therefore becomes several flow rows,
    as it does in the training data.

Even so, this is a reimplementation, not CICFlowMeter itself: verify with
`check_drift()` before trusting a forecast built on PCAP input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from foresight.mitre import AttackStage

DEFAULT_FLOW_TIMEOUT = 120.0        # seconds; CICFlowMeter default
_TCP, _UDP = 6, 17

# Standard TCP flag bits (RFC 793). Duplicated from
# foresight.features.traffic_models.TCPFlags rather than imported: that module
# pulls in pydantic, and the demo/inference path should not grow a dependency
# for six fixed constants. These bit values cannot drift.
_FIN, _SYN, _RST, _PSH, _ACK, _URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


@dataclass
class _Flow:
    """Accumulator for one bidirectional flow."""
    ts: float                        # first packet time
    last: float                      # most recent packet time
    src_ip: str                      # forward direction = first packet seen
    dst_ip: str
    dst_port: int
    protocol: int
    pkts: int = 0
    payload: int = 0
    times: list = field(default_factory=list)
    syn: int = 0
    ack: int = 0
    fin: int = 0
    rst: int = 0
    psh: int = 0
    urg: int = 0
    fin_fwd: bool = False
    fin_bwd: bool = False
    closed: bool = False

    def row(self) -> dict:
        # Flow IAT = gaps between consecutive packets, both directions.
        # A single-packet flow has no gap; 0.0 matches CICFlowMeter, which
        # reports 0 rather than NaN for these.
        if len(self.times) > 1:
            gaps = np.diff(np.asarray(self.times))
            iat_mean = float(gaps.mean())
        else:
            iat_mean = 0.0
        return {
            "ts": self.ts,
            "src_ip": self.src_ip, "dst_ip": self.dst_ip,
            "dst_port": int(self.dst_port), "protocol": int(self.protocol),
            "duration": float(self.last - self.ts),
            "bytes": float(self.payload), "pkts": float(self.pkts),
            "syn": float(self.syn), "ack": float(self.ack), "fin": float(self.fin),
            "rst": float(self.rst), "psh": float(self.psh), "urg": float(self.urg),
            "iat_mean": iat_mean,
            "label": "",
        }


def _key(a_ip, a_port, b_ip, b_port, proto):
    """Direction-independent flow key, so both halves land in one flow."""
    return (proto, *sorted(((a_ip, a_port), (b_ip, b_port))))


def load_pcap(
    path: str,
    max_packets: int | None = None,
    flow_timeout: float = DEFAULT_FLOW_TIMEOUT,
) -> pd.DataFrame:
    """
    Read a .pcap/.pcapng into the canonical flow schema `state.py` expects.

    Streams packets rather than loading the file into memory (scapy's rdpcap
    would hold an entire capture at once, and captures are routinely GBs).
    `max_packets` caps work for interactive use.

    Returns a DataFrame with CANONICAL_COLUMNS. `label` is "" and `stage` is
    BENIGN for every row: a raw capture carries no ground truth, and these are
    placeholders for schema compatibility — never read them as truth.
    """
    try:
        from scapy.all import IP, IPv6, TCP, UDP, PcapReader
    except ImportError as e:                                    # pragma: no cover
        raise ImportError(
            "scapy is required to read PCAP files — `pip install scapy`"
        ) from e

    flows: dict[tuple, _Flow] = {}
    done: list[dict] = []
    n_seen = n_ip = 0

    with PcapReader(path) as reader:
        for pkt in reader:
            n_seen += 1
            if max_packets and n_seen > max_packets:
                break

            ip = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if ip is None:
                continue                                        # ARP, pure L2, etc.

            if pkt.haslayer(TCP):
                l4, proto = pkt.getlayer(TCP), _TCP
            elif pkt.haslayer(UDP):
                l4, proto = pkt.getlayer(UDP), _UDP
            else:
                continue        # ICMP and friends: no ports, and the model's
                                # frac_tcp/frac_udp features only know 6 and 17
            n_ip += 1

            t = float(pkt.time)
            sport, dport = int(l4.sport), int(l4.dport)
            src, dst = str(ip.src), str(ip.dst)
            k = _key(src, sport, dst, dport, proto)

            f = flows.get(k)
            # Expire on inactivity, and never merge across a closed flow —
            # otherwise one long-lived connection becomes a single giant row
            # unlike anything in the training distribution.
            if f is not None and (f.closed or t - f.last > flow_timeout):
                done.append(f.row())
                f = None
            if f is None:
                f = _Flow(ts=t, last=t, src_ip=src, dst_ip=dst,
                          dst_port=dport, protocol=proto)
                flows[k] = f

            forward = (src == f.src_ip and dport == f.dst_port)
            f.last = max(f.last, t)
            f.pkts += 1
            f.times.append(t)
            # L4 payload only — see the module docstring.
            f.payload += len(l4.payload)

            if proto == _TCP:
                flags = int(l4.flags)
                if flags & _SYN: f.syn += 1
                if flags & _ACK: f.ack += 1
                if flags & _PSH: f.psh += 1
                if flags & _URG: f.urg += 1
                if flags & _RST:
                    f.rst += 1
                    f.closed = True
                if flags & _FIN:
                    f.fin += 1
                    if forward:
                        f.fin_fwd = True
                    else:
                        f.fin_bwd = True
                    if f.fin_fwd and f.fin_bwd:
                        f.closed = True

    done.extend(f.row() for f in flows.values())
    if not done:
        raise ValueError(
            f"{path}: no TCP/UDP-over-IP flows found "
            f"({n_seen} packets read, {n_ip} usable)"
        )

    out = pd.DataFrame(done).sort_values("ts").reset_index(drop=True)
    out["stage"] = int(AttackStage.BENIGN)
    return out
