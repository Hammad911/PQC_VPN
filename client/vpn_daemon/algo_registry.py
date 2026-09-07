# client/vpn_daemon/algo_registry.py
"""
Defines which algorithms are active in the RL agent's action space.
HQC is defined but disabled until NIST finalises the standard.
"""

ALGORITHMS = {
    0: {
        "name":     "ML-KEM-512",
        "standard": "NIST FIPS 203",
        "active":   True,
        "security": 0.70,
        "cpu_cost": 0.5,
    },
    1: {
        "name":     "ML-KEM-768",
        "standard": "NIST FIPS 203",
        "active":   True,
        "security": 0.85,
        "cpu_cost": 1.0,
    },
    2: {
        "name":     "ML-KEM-1024",
        "standard": "NIST FIPS 203",
        "active":   True,
        "security": 1.00,
        "cpu_cost": 2.0,
    },
    3: {
        # HQC DISABLED — draft standard, CVE-2024-54137 fixed in
        # liboqs 0.12.0 but still not finalized. Enable in future.
        "name":     "HQC-256",
        "standard": "NIST 2025 draft",
        "active":   False,
        "security": 0.90,
        "cpu_cost": 1.5,
    },
    4: {
        "name":     "rekey-now",
        "standard": "N/A",
        "active":   True,
        "security": 1.00,
        "cpu_cost": 3.0,
    },
}

# Only active algorithms exposed to RL agent
ACTIVE_ACTIONS = {
    k: v for k, v in ALGORITHMS.items() if v["active"]
}

# ------------------------------------------------------------ action tables
#
# The RL action space is the ACTIVE_ACTIONS keys renumbered 0..N-1 and
# contiguous (registry key 3, the disabled HQC-256, is skipped). That index
# ordering is what `contracts/algo_registry.json` freezes and what the ONNX
# output columns mean, so it is defined once, here, in the module the
# contract names as its source of truth — rather than re-derived in each
# module that needs it, which is how seven slightly different spellings of
# the same list appeared. Enabling HQC-256 later is then one re-export, not
# an audit of every derivation.
#
# Kept to plain lists and stdlib: decision_gate.py consumes these and is
# meant to stay importable without numpy or gymnasium.

# action index (0..N-1) -> algo_registry key
ACTION_TO_ALGO_KEY = {i: k for i, k in enumerate(ACTIVE_ACTIONS)}
ACTION_NAMES = [ACTIVE_ACTIONS[k]["name"] for k in ACTION_TO_ALGO_KEY.values()]
ACTION_SECURITY = [ACTIVE_ACTIONS[k]["security"] for k in ACTION_TO_ALGO_KEY.values()]
N_ACTIONS = len(ACTION_NAMES)
REKEY_ACTION_IDX = ACTION_NAMES.index("rekey-now")
KEM_ACTIONS = [i for i, n in enumerate(ACTION_NAMES) if n != "rekey-now"]