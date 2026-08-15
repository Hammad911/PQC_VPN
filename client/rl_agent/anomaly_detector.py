# client/rl_agent/anomaly_detector.py
"""
Anomaly detection pipeline (proposal section 3.3).

Only Layer 1 is implemented so far — the always-active statistical
Z-score baseline over latency, packet rate, and packet size. Layers 2
(rule-based signatures, active below 70% CPU) and 3 (Isolation Forest,
active below 40% CPU) are not built yet; see PROGRESS.md.
"""
from collections import deque

import numpy as np

Z_SCORE_THRESHOLD = 3.0


class ZScoreBaseline:
    """Rolling mean/std baseline per metric. Flags any sample whose
    Z-score exceeds Z_SCORE_THRESHOLD. Needs `window` samples before
    it can make a judgement — returns not-anomalous until then, since
    there isn't enough history to say otherwise."""

    def __init__(self, window: int = 50):
        self.window = window
        self._history = {
            "latency": deque(maxlen=window),
            "packet_rate": deque(maxlen=window),
            "packet_size": deque(maxlen=window),
        }

    def _z_score(self, metric: str, value: float) -> float:
        hist = self._history[metric]
        if len(hist) < 2:
            return 0.0
        mean = np.mean(hist)
        std = np.std(hist)
        if std == 0:
            return 0.0
        return abs((value - mean) / std)

    def update(self, latency: float, packet_rate: float, packet_size: float) -> dict:
        """Feed one sample, get back per-metric Z-scores and a combined
        threat_score in [0, 1]. History is updated with the new sample
        after scoring it, so the sample itself doesn't inflate its own
        baseline."""
        scores = {
            "latency": self._z_score("latency", latency),
            "packet_rate": self._z_score("packet_rate", packet_rate),
            "packet_size": self._z_score("packet_size", packet_size),
        }

        self._history["latency"].append(latency)
        self._history["packet_rate"].append(packet_rate)
        self._history["packet_size"].append(packet_size)

        max_z = max(scores.values())
        anomalous = max_z > Z_SCORE_THRESHOLD
        # squash to [0,1]: anything at/above 2x threshold saturates to 1.0
        threat_score = float(np.clip(max_z / (Z_SCORE_THRESHOLD * 2), 0.0, 1.0))

        return {
            "anomalous": anomalous,
            "threat_score": threat_score,
            "z_scores": scores,
            "ready": len(self._history["latency"]) >= self.window,
        }
