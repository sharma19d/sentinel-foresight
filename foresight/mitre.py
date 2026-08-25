"""
MITRE ATT&CK kill-chain stage vocabulary for attack-progression prediction.

PS 26153 asks predictions to be mapped to recognised attack stages. We use a
small ordered set of kill-chain phases; the world model predicts which stage
the network's *future* state corresponds to.
"""

from enum import IntEnum


class AttackStage(IntEnum):
    """Ordered kill-chain phases. Order matters: progression = increasing stage."""
    BENIGN = 0
    RECONNAISSANCE = 1      # port/host scanning, probing
    INITIAL_ACCESS = 2      # brute-force, exploit landing, web attacks
    LATERAL_MOVEMENT = 3    # internal pivoting / infiltration foothold
    COMMAND_AND_CONTROL = 4 # C2 / botnet beaconing
    EXFILTRATION = 5        # final-stage impact — data theft or DoS/DDoS payoff


def stage_from_label(label: str) -> AttackStage:
    """
    Map a raw dataset label to a kill-chain stage by keyword — robust to the
    messy, inconsistent label strings in CIC-IDS-2018 / CTU-13 (mixed casing,
    attack sub-names, and the dataset's infamous 'Infilteration' misspelling).
    """
    s = (label or "").strip().lower()
    if not s or "benign" in s or s == "normal":
        return AttackStage.BENIGN
    # reconnaissance
    if "portscan" in s or "port scan" in s or "recon" in s or "scan" in s:
        return AttackStage.RECONNAISSANCE
    # infiltration foothold — note the dataset spells it "Infilteration"
    if "infilter" in s or "infiltrat" in s:
        return AttackStage.LATERAL_MOVEMENT
    # initial access — brute force + web app attacks
    if "brute" in s or "sql" in s or "xss" in s or "injection" in s or "web" in s:
        return AttackStage.INITIAL_ACCESS
    # command & control
    if "bot" in s or "c2" in s or "beacon" in s:
        return AttackStage.COMMAND_AND_CONTROL
    # final impact — DoS/DDoS and their sub-tools
    if any(k in s for k in ("ddos", "dos", "goldeneye", "hulk", "slowloris",
                            "slowhttp", "hoic", "loic", "exfil")):
        return AttackStage.EXFILTRATION
    return AttackStage.BENIGN   # unknown → conservative
