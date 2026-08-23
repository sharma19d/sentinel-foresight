"""
MITRE ATT&CK kill-chain stage vocabulary for attack-progression prediction.

PS 26153 asks predictions to be mapped to recognised attack stages. We use a
small ordered set of kill-chain phases; the world model predicts which stage
the network's *future* state corresponds to, not just the current traffic.
"""

from enum import IntEnum


class AttackStage(IntEnum):
    """Ordered kill-chain phases. Order matters: progression = increasing stage."""
    BENIGN = 0
    RECONNAISSANCE = 1      # port/host scanning, probing
    INITIAL_ACCESS = 2      # brute-force, exploit landing
    LATERAL_MOVEMENT = 3    # internal pivoting
    COMMAND_AND_CONTROL = 4 # C2 beaconing
    EXFILTRATION = 5        # data theft / DoS payoff


# Map common dataset attack labels (CIC-IDS-2018 / CTU-13) to a kill-chain stage.
# Extended as we wire in each dataset's label taxonomy.
DATASET_LABEL_TO_STAGE = {
    "benign": AttackStage.BENIGN,
    "portscan": AttackStage.RECONNAISSANCE,
    "port scan": AttackStage.RECONNAISSANCE,
    "infiltration": AttackStage.INITIAL_ACCESS,
    "ftp-bruteforce": AttackStage.INITIAL_ACCESS,
    "ssh-bruteforce": AttackStage.INITIAL_ACCESS,
    "brute force": AttackStage.INITIAL_ACCESS,
    "bot": AttackStage.COMMAND_AND_CONTROL,
    "botnet": AttackStage.COMMAND_AND_CONTROL,
    "ddos": AttackStage.EXFILTRATION,
    "dos": AttackStage.EXFILTRATION,
}


def stage_from_label(label: str) -> AttackStage:
    return DATASET_LABEL_TO_STAGE.get((label or "").strip().lower(), AttackStage.BENIGN)
