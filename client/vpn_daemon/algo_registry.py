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