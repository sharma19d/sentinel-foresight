# ─────────────────────────────────────────────────────────────────────────
# Ported from SENTINEL: sentinel/core/capture/models.py  (unchanged)
# Flow-level vocabulary: TCP flags, protocol, flow schema (PS 26153 flow features).
# ─────────────────────────────────────────────────────────────────────────
"""
SENTINEL Layer 1 — Data Models

Pydantic models defining all data structures flowing through the
Network Capture & Packet Engine pipeline.

All timestamps are UTC epoch floats. All IPs are strings.
All MAC addresses are colon-separated lowercase hex (aa:bb:cc:dd:ee:ff).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ── Enums ────────────────────────────────────────────────────────────

class FlowStatus(str, Enum):
    """Lifecycle state of a tracked network flow."""
    ACTIVE = "active"
    FINISHED = "finished"
    TIMED_OUT = "timed_out"


class FlowEventType(str, Enum):
    """Type of flow lifecycle event published to Redis."""
    NEW_FLOW = "new_flow"
    FLOW_UPDATE = "flow_update"
    FLOW_END = "flow_end"


class L4Protocol(str, Enum):
    """Layer-4 transport protocol."""
    TCP = "TCP"
    UDP = "UDP"
    ICMP = "ICMP"
    IGMP = "IGMP"
    OTHER = "OTHER"

    @classmethod
    def from_ip_proto(cls, proto_num: int) -> "L4Protocol":
        """Convert IP protocol number to enum member."""
        _MAP = {1: cls.ICMP, 2: cls.IGMP, 6: cls.TCP, 17: cls.UDP}
        return _MAP.get(proto_num, cls.OTHER)


# ── Per-Packet Metadata ─────────────────────────────────────────────

class PacketMeta(BaseModel):
    """
    Metadata extracted from a single captured packet.
    This is the raw unit produced by the sniffer before flow aggregation.
    """
    timestamp: float = Field(
        ..., description="Epoch timestamp when the packet was captured"
    )

    # Layer 2
    src_mac: str = Field(
        ..., description="Source MAC address (aa:bb:cc:dd:ee:ff)"
    )
    dst_mac: str = Field(
        ..., description="Destination MAC address"
    )

    # Layer 3
    src_ip: str = Field(
        ..., description="Source IP address"
    )
    dst_ip: str = Field(
        ..., description="Destination IP address"
    )
    ip_version: int = Field(
        default=4, description="IP version (4 or 6)"
    )
    ttl: int = Field(
        default=0, description="Time-to-live / hop limit"
    )

    # Layer 4
    protocol: L4Protocol = Field(
        default=L4Protocol.OTHER, description="Transport protocol"
    )
    src_port: Optional[int] = Field(
        default=None, description="Source port (TCP/UDP only)"
    )
    dst_port: Optional[int] = Field(
        default=None, description="Destination port (TCP/UDP only)"
    )
    tcp_flags: Optional[int] = Field(
        default=None, description="TCP flags bitmask (TCP only)"
    )

    # Payload
    length: int = Field(
        ..., description="Total packet length in bytes"
    )
    payload_size: int = Field(
        default=0, description="Application-layer payload size in bytes"
    )

    # Layer 7 hints (populated by metadata_extractor before flow tracking)
    dns_query: Optional[str] = Field(
        default=None, description="DNS query name if DNS packet"
    )
    dns_answers: Optional[list[str]] = Field(
        default=None, description="DNS answer records"
    )
    http_host: Optional[str] = Field(
        default=None, description="HTTP Host header value"
    )
    http_user_agent: Optional[str] = Field(
        default=None, description="HTTP User-Agent header value"
    )
    http_method: Optional[str] = Field(
        default=None, description="HTTP request method (GET, POST, etc.)"
    )
    http_uri: Optional[str] = Field(
        default=None, description="HTTP request URI path"
    )
    tls_sni: Optional[str] = Field(
        default=None, description="TLS Server Name Indication hostname"
    )
    tls_version: Optional[str] = Field(
        default=None, description="TLS version string"
    )

    # ARP (Layer 2 discovery events)
    arp_operation: Optional[str] = Field(
        default=None, description="ARP operation: 'request' or 'reply'"
    )
    arp_sender_mac: Optional[str] = Field(
        default=None, description="ARP sender hardware address"
    )
    arp_sender_ip: Optional[str] = Field(
        default=None, description="ARP sender protocol address"
    )
    arp_target_ip: Optional[str] = Field(
        default=None, description="ARP target protocol address"
    )

    # Discovery metadata dicts (raw from metadata_extractor)
    mdns_info: Optional[dict] = Field(default=None)
    ssdp_info: Optional[dict] = Field(default=None)
    dhcp_info: Optional[dict] = Field(default=None)

    # Raw flag
    is_discovery_relevant: bool = Field(
        default=False,
        description="True if packet is ARP/mDNS/SSDP/DHCP — triggers Layer 2 events"
    )


# ── TCP Flag Constants ───────────────────────────────────────────────

class TCPFlags:
    """TCP flag bitmask constants for readability."""
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    ECE = 0x40
    CWR = 0x80

    @staticmethod
    def to_list(flags: int) -> list[str]:
        """Convert TCP flags bitmask to list of flag names."""
        names = []
        if flags & TCPFlags.FIN:
            names.append("FIN")
        if flags & TCPFlags.SYN:
            names.append("SYN")
        if flags & TCPFlags.RST:
            names.append("RST")
        if flags & TCPFlags.PSH:
            names.append("PSH")
        if flags & TCPFlags.ACK:
            names.append("ACK")
        if flags & TCPFlags.URG:
            names.append("URG")
        if flags & TCPFlags.ECE:
            names.append("ECE")
        if flags & TCPFlags.CWR:
            names.append("CWR")
        return names


# ── Flow Record ──────────────────────────────────────────────────────

class FlowRecord(BaseModel):
    """
    Aggregated metadata for a tracked network flow.
    A flow is identified by the bidirectional 5-tuple:
    (src_ip, dst_ip, src_port, dst_port, protocol).

    This is the PRIMARY OUTPUT of Layer 1 — consumed by
    Device Discovery (Layer 2) and the AI Engine (Layer 3).
    """
    flow_id: str = Field(
        ..., description="Deterministic hash of the bidirectional 5-tuple"
    )

    # 5-tuple (canonical direction: lower IP is always 'src' for consistency)
    src_ip: str = Field(..., description="Source IP (canonical — lower IP)")
    dst_ip: str = Field(..., description="Destination IP (canonical — higher IP)")
    src_port: Optional[int] = Field(default=None)
    dst_port: Optional[int] = Field(default=None)
    protocol: L4Protocol = Field(default=L4Protocol.OTHER)

    # Layer 2
    src_mac: str = Field(default="", description="Most recent source MAC")
    dst_mac: str = Field(default="", description="Most recent destination MAC")
    vendor_src: str = Field(default="Unknown", description="Source MAC vendor name")
    vendor_dst: str = Field(default="Unknown", description="Destination MAC vendor name")

    # Layer 7 — Deep Packet Inspection
    app_protocol: str = Field(
        default="Unknown",
        description="Application-layer protocol identified by nDPI (e.g., HTTP, DNS, MQTT)"
    )
    tls_sni: Optional[str] = Field(default=None)
    http_user_agents: list[str] = Field(
        default_factory=list,
        description="Unique HTTP User-Agent strings seen in this flow"
    )
    dns_queries: list[str] = Field(
        default_factory=list,
        description="DNS query names observed in this flow"
    )
    dns_answers: list[str] = Field(
        default_factory=list,
        description="DNS answer records observed in this flow"
    )
    http_hosts: list[str] = Field(
        default_factory=list,
        description="HTTP Host header values seen in this flow"
    )

    # Counters
    packet_count: int = Field(default=0, description="Total packets in this flow")
    byte_count: int = Field(default=0, description="Total bytes in this flow")
    packet_count_fwd: int = Field(
        default=0, description="Packets in the forward direction (src→dst)"
    )
    packet_count_bwd: int = Field(
        default=0, description="Packets in the backward direction (dst→src)"
    )
    byte_count_fwd: int = Field(default=0, description="Bytes forward")
    byte_count_bwd: int = Field(default=0, description="Bytes backward")

    # Timestamps
    start_time: float = Field(..., description="Flow start epoch timestamp")
    end_time: float = Field(..., description="Most recent packet epoch timestamp")
    last_activity: float = Field(..., description="Last packet seen (for idle timeout)")

    # TCP state tracking
    tcp_flags_seen: list[str] = Field(
        default_factory=list, description="Unique TCP flags observed"
    )
    tcp_flags_fwd: list[str] = Field(
        default_factory=list, description="TCP flags in forward direction"
    )
    tcp_flags_bwd: list[str] = Field(
        default_factory=list, description="TCP flags in backward direction"
    )

    # Status
    status: FlowStatus = Field(
        default=FlowStatus.ACTIVE, description="Flow lifecycle state"
    )

    @computed_field  # type: ignore[misc]
    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return round(self.end_time - self.start_time, 6)

    @computed_field  # type: ignore[misc]
    @property
    def packets_per_second(self) -> float:
        """Average packets per second over the flow duration."""
        d = self.duration
        return round(self.packet_count / d, 2) if d > 0 else 0.0

    @computed_field  # type: ignore[misc]
    @property
    def bytes_per_second(self) -> float:
        """Average throughput in bytes per second."""
        d = self.duration
        return round(self.byte_count / d, 2) if d > 0 else 0.0

    @computed_field  # type: ignore[misc]
    @property
    def start_time_iso(self) -> str:
        """Flow start as ISO 8601 string."""
        return datetime.fromtimestamp(
            self.start_time, tz=timezone.utc
        ).isoformat()


# ── Flow Event (published to Redis) ─────────────────────────────────

class FlowEvent(BaseModel):
    """
    Event object published to Redis when a flow changes state.
    Consumed by Layer 2 (Device Discovery), Layer 3 (AI Engine),
    and Layer 7 (Dashboard) via their Redis subscribers.
    """
    event_type: FlowEventType = Field(
        ..., description="Type of flow lifecycle event"
    )
    flow: FlowRecord = Field(
        ..., description="Full flow record at the time of the event"
    )
    timestamp: float = Field(
        ..., description="Event emission epoch timestamp"
    )

    @computed_field  # type: ignore[misc]
    @property
    def timestamp_iso(self) -> str:
        """Event timestamp as ISO 8601 string."""
        return datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()


# ── Discovery Event (ARP / mDNS / SSDP / DHCP) ─────────────────────

class DiscoveryEvent(BaseModel):
    """
    Event published to Redis when a discovery-relevant packet is captured.
    Consumed by Layer 2 (Device Discovery & Risk Profiler Agent).
    """
    event_type: str = Field(
        ..., description="Discovery protocol: 'arp', 'mdns', 'ssdp', 'dhcp'"
    )
    timestamp: float = Field(...)

    # Common fields
    mac_address: str = Field(default="", description="Device MAC address")
    ip_address: str = Field(default="", description="Device IP address")
    vendor: str = Field(default="Unknown", description="MAC vendor name")

    # ARP-specific
    arp_operation: Optional[str] = Field(default=None)

    # mDNS-specific
    mdns_hostname: Optional[str] = Field(default=None)
    mdns_service_type: Optional[str] = Field(default=None)
    mdns_service_name: Optional[str] = Field(default=None)

    # SSDP-specific
    ssdp_location: Optional[str] = Field(default=None)
    ssdp_server: Optional[str] = Field(default=None)
    ssdp_usn: Optional[str] = Field(default=None)
    ssdp_service_type: Optional[str] = Field(default=None)

    # DHCP-specific
    dhcp_hostname: Optional[str] = Field(default=None)
    dhcp_vendor_class: Optional[str] = Field(default=None)
    dhcp_requested_ip: Optional[str] = Field(default=None)


# ── Capture Statistics ───────────────────────────────────────────────

class CaptureStats(BaseModel):
    """
    Running statistics for the capture engine.
    Published periodically to Redis for dashboard display.
    """
    packets_captured: int = Field(default=0)
    packets_per_second: float = Field(default=0.0)
    bytes_captured: int = Field(default=0)
    bytes_per_second: float = Field(default=0.0)
    flows_active: int = Field(default=0)
    flows_total: int = Field(default=0)
    flows_timed_out: int = Field(default=0)
    capture_interface: str = Field(default="")
    uptime_seconds: float = Field(default=0.0)
    capture_started: float = Field(default=0.0)
    dpi_enabled: bool = Field(default=False)
    redis_connected: bool = Field(default=False)
    errors: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def capture_started_iso(self) -> str:
        if self.capture_started > 0:
            return datetime.fromtimestamp(
                self.capture_started, tz=timezone.utc
            ).isoformat()
        return ""
