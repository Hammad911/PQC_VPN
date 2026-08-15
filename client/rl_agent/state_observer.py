# client/rl_agent/state_observer.py
"""
Reads live device/network metrics and packages them into the 7-dim
normalised state vector consumed by the RL agent (proposal section 3.2):

    [cpu_load, ram_available, latency, upload_volume,
     connection_type, time_since_rekey, threat_score]

All values are floats in [0, 1]. Anything that cannot be measured
(e.g. no network reachable) degrades to a conservative default rather
than raising, since the agent must always be able to observe *a* state.
"""
import platform
import subprocess
import time

import numpy as np
import psutil

STATE_DIM = 7

# Normalisation caps — chosen so a "bad" reading saturates near 1.0
# rather than blowing up the vector; these are tuned defaults, not
# measured constants, and are the first thing to revisit once we have
# real traffic data to calibrate against.
LATENCY_CAP_MS = 300.0
UPLOAD_CAP_BYTES_PER_SEC = 5_000_000.0
REKEY_INTERVAL_CAP_SEC = 3600.0

CONNECTION_TYPE_SCORE = {
    "wired": 0.0,
    "wifi": 0.5,
    "cellular": 1.0,
    "unknown": 0.5,
}


class StateObserver:
    """Stateful — tracks deltas (traffic volume, rekey timer) across calls."""

    def __init__(self, target_host: str = "1.1.1.1"):
        self.target_host = target_host
        self._last_net = psutil.net_io_counters()
        self._last_sample_time = time.monotonic()
        self._last_rekey_time = time.monotonic()

    def mark_rekey(self):
        """Call this whenever the agent triggers a key re-exchange."""
        self._last_rekey_time = time.monotonic()

    def _cpu_load(self) -> float:
        return psutil.cpu_percent(interval=0.1) / 100.0

    def _ram_available(self) -> float:
        return psutil.virtual_memory().available / psutil.virtual_memory().total

    def _latency(self) -> float:
        """Single ICMP ping to target_host, normalised against LATENCY_CAP_MS.
        Falls back to 1.0 (worst case) if the host is unreachable."""
        count_flag = "-n" if platform.system() == "Windows" else "-c"
        timeout_flag = "-w" if platform.system() == "Windows" else "-W"
        timeout_val = "1000" if platform.system() == "Windows" else "1"
        try:
            start = time.monotonic()
            result = subprocess.run(
                ["ping", count_flag, "1", timeout_flag, timeout_val, self.target_host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            if result.returncode != 0:
                return 1.0
            return min(elapsed_ms / LATENCY_CAP_MS, 1.0)
        except (subprocess.TimeoutExpired, OSError):
            return 1.0

    def _upload_volume(self) -> float:
        now = psutil.net_io_counters()
        elapsed = max(time.monotonic() - self._last_sample_time, 1e-6)
        bytes_sent = max(now.bytes_sent - self._last_net.bytes_sent, 0)
        rate = bytes_sent / elapsed
        self._last_net = now
        self._last_sample_time = time.monotonic()
        return min(rate / UPLOAD_CAP_BYTES_PER_SEC, 1.0)

    def _connection_type(self) -> float:
        """Best-effort classification from active interface names.
        No cross-platform API gives this reliably without elevated
        privileges, so we pattern-match common interface naming."""
        try:
            stats = psutil.net_if_stats()
            active = [name for name, s in stats.items() if s.isup]
        except Exception:
            return CONNECTION_TYPE_SCORE["unknown"]

        lowered = [name.lower() for name in active]
        if any("wl" in n or "wifi" in n or "wlan" in n for n in lowered):
            return CONNECTION_TYPE_SCORE["wifi"]
        if any(
            n.startswith(("en", "eth")) and "wl" not in n for n in lowered
        ):
            return CONNECTION_TYPE_SCORE["wired"]
        if any("cellular" in n or "wwan" in n for n in lowered):
            return CONNECTION_TYPE_SCORE["cellular"]
        return CONNECTION_TYPE_SCORE["unknown"]

    def _time_since_rekey(self) -> float:
        elapsed = time.monotonic() - self._last_rekey_time
        return min(elapsed / REKEY_INTERVAL_CAP_SEC, 1.0)

    def read_state(self, threat_score: float = 0.0) -> np.ndarray:
        """threat_score comes from the anomaly detection pipeline
        (client/rl_agent/anomaly_detector.py) — passed in rather than
        computed here so the two subsystems stay decoupled."""
        vector = np.array(
            [
                self._cpu_load(),
                self._ram_available(),
                self._latency(),
                self._upload_volume(),
                self._connection_type(),
                self._time_since_rekey(),
                float(np.clip(threat_score, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return np.clip(vector, 0.0, 1.0)
