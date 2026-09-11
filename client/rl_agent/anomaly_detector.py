# client/rl_agent/anomaly_detector.py
"""
Anomaly detection pipeline (proposal section 3.3).

Layer 1 (always-active statistical Z-score baseline over latency, packet
rate, and packet size), Layer 2 (rule-based signatures for known attack
patterns, active below 70% CPU), and the combiner that folds them together
under the CPU gate (`AnomalyCombiner`) are implemented here. Layer 3
(Isolation Forest, active below 40% CPU) is not built yet; see PROGRESS.md.

Every class in this module takes plain numeric inputs and is decoupled
from psutil / live sourcing, the same way ZScoreBaseline always has been —
`AnomalyCombiner` is no exception: it still takes plain numbers for every
layer's metrics (including `cpu_load`, for the gate itself). Wiring a real
network/packet-capture source into those inputs is separate, later work —
see the combiner's own docstring.
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


# ------------------------------------------------------------- Layer 2
#
# Layer 1 is a statistical baseline: it learns what "normal" looks like for
# *this* session and flags deviation from it. Layer 2 is the opposite kind
# of check — a fixed rule standing in for a specific, named attack pattern,
# independent of any per-session baseline. Each class below returns the
# same key set as ZScoreBaseline (minus the metric-specific z_scores,
# replaced by a generic `detail`) so the Week 6 combiner can fold all six
# detectors together without knowing which rule produced which number.

def _ratio_score(value: float, threshold: float, ceiling: float = 2.0) -> float:
    """Squash a rule-based metric to [0, 1]: 0 at zero, saturating to 1.0
    once `value` reaches `ceiling`x the trigger threshold. Same squash shape
    ZScoreBaseline already uses for its Z-score (`max_z / (threshold * 2)`),
    generalized once here instead of five copies of the same clip-and-divide
    — and it keeps every Layer 2 class's threat_score calibrated the same
    way Layer 1's already is."""
    if threshold <= 0:
        return 0.0
    return float(np.clip(value / (threshold * ceiling), 0.0, 1.0))


PORT_SCAN_DISTINCT_PORTS = 15
# A legitimate client touches a handful of well-known ports per 5s tick (a
# page load hits 80/443 and maybe a couple more); 15 is comfortably above
# that and comfortably below even a conservative scan's per-tick footprint.
PORT_SCAN_FAILURE_RATIO = 0.7
# SYN scans deliberately never complete the handshake; ordinary flaky-network
# failures are rarely a majority of attempts. 0.7 catches the scan pattern
# while tolerating transient real-world failure noise.


class PortScanSignature:
    """Fanout + failure rate: many distinct destination ports touched in one
    tick, most of which never complete — the signature of a SYN scan rather
    than normal multi-service traffic. No rolling window: the fanout is
    already visible within a single 5-second tick, so this is a per-tick
    rule, not a multi-tick statistic (`ready` is always True)."""

    def update(self, distinct_ports: int, connection_attempts: int,
               failed_attempts: int) -> dict:
        failure_ratio = failed_attempts / connection_attempts if connection_attempts else 0.0
        port_score = _ratio_score(distinct_ports, PORT_SCAN_DISTINCT_PORTS)
        failure_score = _ratio_score(failure_ratio, PORT_SCAN_FAILURE_RATIO)
        # Both conditions must hold — high fanout alone isn't a scan if every
        # connection also succeeds (e.g. a legitimately chatty application).
        anomalous = (distinct_ports >= PORT_SCAN_DISTINCT_PORTS
                     and failure_ratio >= PORT_SCAN_FAILURE_RATIO)
        return {
            "anomalous": anomalous,
            "threat_score": min(port_score, failure_score),
            "ready": True,
            "detail": {"distinct_ports": distinct_ports, "failure_ratio": failure_ratio},
        }


RETRANS_RATE_THRESHOLD = 0.10
# Network-monitoring convention treats a few percent retransmission/loss as
# ordinary congestion noise; >5-10% is "degraded". 10% flags disruption
# severe enough to be attack-consistent (on-path interference forcing
# retransmits) rather than routine wifi jitter.
RETRANS_CONSECUTIVE_TICKS = 2
# One elevated tick is a plausible one-off blip. Requiring two ticks in a
# row is what makes this a *sustained-disruption* signature rather than a
# re-implementation of Layer 1's single-sample Z-score on a new metric.


class RetransmissionSpikeSignature:
    """Sustained elevated TCP retransmission rate. A streak counter, not a
    deque of values — it only needs to know how many ticks in a row have
    crossed the line."""

    def __init__(self, consecutive_ticks: int = RETRANS_CONSECUTIVE_TICKS):
        self.consecutive_ticks = consecutive_ticks
        self._streak = 0

    def update(self, retransmit_rate: float) -> dict:
        if retransmit_rate >= RETRANS_RATE_THRESHOLD:
            self._streak += 1
        else:
            self._streak = 0
        anomalous = self._streak >= self.consecutive_ticks
        return {
            "anomalous": anomalous,
            "threat_score": _ratio_score(retransmit_rate, RETRANS_RATE_THRESHOLD),
            "ready": True,
            "detail": {"retransmit_rate": retransmit_rate, "streak": self._streak},
        }


MITM_JITTER_MS_THRESHOLD = 80.0
# Healthy consumer wifi/broadband/cellular commonly shows tick-to-tick ICMP
# jitter within a few tens of ms; a relay/interception hop adds tens-to-
# hundreds of ms of *variable* overhead. 80ms sits above ordinary noise and
# below where it would be indistinguishable from a badly congested but
# benign link.
MITM_WINDOW_TICKS = 3
MITM_CONSECUTIVE_TICKS = 2
# One jittery window could be a transient hiccup; two consecutive elevated
# windows is much less likely from ordinary noise alone.


class MitMLatencySignature:
    """Checks *jitter* (max - min over a short window of raw latencies), not
    absolute level — Layer 1 already flags an absolute-latency outlier. What
    is specific to an interception/relay path is variably-added latency
    tick to tick, not one clean step-change."""

    def __init__(self, window_ticks: int = MITM_WINDOW_TICKS,
                 consecutive_ticks: int = MITM_CONSECUTIVE_TICKS):
        self._history = deque(maxlen=window_ticks)
        self.window_ticks = window_ticks
        self.consecutive_ticks = consecutive_ticks
        self._streak = 0

    def update(self, latency_ms: float) -> dict:
        """`latency_ms` is raw milliseconds, not the normalized [0,1]
        LATENCY state-vector dimension — the threshold above is only
        interpretable against raw units."""
        self._history.append(latency_ms)
        ready = len(self._history) >= self.window_ticks
        jitter = (max(self._history) - min(self._history)) if ready else 0.0
        if ready and jitter >= MITM_JITTER_MS_THRESHOLD:
            self._streak += 1
        else:
            self._streak = 0
        anomalous = self._streak >= self.consecutive_ticks
        return {
            "anomalous": anomalous,
            "threat_score": _ratio_score(jitter, MITM_JITTER_MS_THRESHOLD),
            "ready": ready,
            "detail": {"jitter_ms": jitter, "streak": self._streak},
        }


EXFIL_UPLOAD_BPS_THRESHOLD = 2_000_000.0
# A fraction of state_observer.py's UPLOAD_CAP_BYTES_PER_SEC = 5_000_000.0
# (that file's normalization cap for "saturated" upload) — reusing a number
# the codebase already agreed on for "a lot of upload" keeps the two
# subsystems' notion of high upload consistent instead of inventing an
# unrelated figure. 2MB/s is comfortably below full saturation.
EXFIL_UPLOAD_DOWNLOAD_RATIO = 3.0
# Even upload-heavy legitimate traffic (a video call) sits close to 1:1;
# 3:1 requires a clear inversion of normal traffic shape before firing.
EXFIL_CONSECUTIVE_TICKS = 2
# Sustained means "still happening 10 seconds later" — a single burst
# (sending one large file) shouldn't fire.


class BandwidthExfiltrationSignature:
    """Upload rate alone can't distinguish exfiltration from a legitimate
    large upload; the cheap heuristic is asymmetry — normal client traffic
    is download-dominated or roughly balanced, so a host pushing far more up
    than down is doing something atypical for ordinary consumption."""

    def __init__(self, consecutive_ticks: int = EXFIL_CONSECUTIVE_TICKS):
        self.consecutive_ticks = consecutive_ticks
        self._streak = 0

    def update(self, upload_bytes_per_sec: float, download_bytes_per_sec: float) -> dict:
        ratio = upload_bytes_per_sec / max(download_bytes_per_sec, 1.0)
        clears_line = (upload_bytes_per_sec >= EXFIL_UPLOAD_BPS_THRESHOLD
                        and ratio >= EXFIL_UPLOAD_DOWNLOAD_RATIO)
        self._streak = self._streak + 1 if clears_line else 0
        anomalous = self._streak >= self.consecutive_ticks
        return {
            "anomalous": anomalous,
            "threat_score": min(
                _ratio_score(upload_bytes_per_sec, EXFIL_UPLOAD_BPS_THRESHOLD),
                _ratio_score(ratio, EXFIL_UPLOAD_DOWNLOAD_RATIO),
            ),
            "ready": True,
            "detail": {"upload_bps": upload_bytes_per_sec, "ratio": ratio, "streak": self._streak},
        }


class DNSServerChangeSignature:
    """Deliberately the odd one out: edge-triggered on a discrete state
    change, not windowed statistics. A DNS-server change (a classic
    MitM/DNS-hijack technique) is already fully informative on the first
    observation after it happens — forcing a rolling window onto it would
    just delay detection of an event that doesn't need one.

    Known, explicit limitation (not silently swept under the rug): this
    fires on ANY DNS change, including benign ones (switching wifi
    networks). Properly suppressing benign changes needs an explicit
    "expect a reconfiguration" hook, analogous to
    `StateObserver.mark_rekey()` — out of scope for this week.
    """

    def __init__(self, expected: str | None = None):
        self._last = expected

    def update(self, dns_server: str) -> dict:
        # First observation only seeds the baseline — can't call something
        # "changed" with no prior value, matching ZScoreBaseline's own
        # "not enough history to say otherwise" stance.
        ready = self._last is not None
        anomalous = ready and dns_server != self._last
        changed_from = self._last
        self._last = dns_server
        return {
            "anomalous": anomalous,
            # Binary signature match, not a continuous metric — there's no
            # meaningful "how far over the line" for a value that either
            # changed or didn't.
            "threat_score": 1.0 if anomalous else 0.0,
            "ready": ready,
            "detail": {"previous": changed_from, "current": dns_server},
        }


# ------------------------------------------------------------- Combiner
#
# Week 6: fold Layer 1 (always on) and the five Layer 2 signatures (gated)
# into the single `threat_score` the state vector's THREAT dimension wants
# (contracts/state_vector.json: "Combined anomaly score across whichever
# detection layers the CPU gate allows; the consumer never needs to know
# which are active"). Layer 3 is not built yet, so there is nothing to gate
# below 40% CPU here — that lands Week 7.

LAYER2_CPU_GATE = 0.70
# Normalized [0,1], matching StateObserver.cpu_load() (psutil.cpu_percent()
# / 100) — the same value the combiner's caller is expected to pass in, so
# the gate compares directly against it instead of a second, raw-percent
# convention living only in this module.


class AnomalyCombiner:
    """Owns one ZScoreBaseline (Layer 1) and the five Layer 2 signatures,
    and combines whichever of them are active into one reading.

    Deliberately still decoupled from psutil / live network sourcing, same
    as every class above: `update()` takes `cpu_load` and every Layer 1/2
    metric as plain numbers. Wiring those to real device/network sources
    (psutil for cpu_load, a packet-capture or netstat-derived source for
    the Layer 2 metrics) is separate, later work — there is no live daemon
    to call this yet, the same reason StateObserver's own live psutil
    reads have so far only ever been exercised by demo.py.
    """

    def __init__(self, window: int = 50):
        self.layer1 = ZScoreBaseline(window=window)
        self.port_scan = PortScanSignature()
        self.retransmission = RetransmissionSpikeSignature()
        self.mitm_latency = MitMLatencySignature()
        self.exfiltration = BandwidthExfiltrationSignature()
        self.dns_change = DNSServerChangeSignature()

    def update(
        self,
        cpu_load: float,
        *,
        latency: float,
        packet_rate: float,
        packet_size: float,
        distinct_ports: int,
        connection_attempts: int,
        failed_attempts: int,
        retransmit_rate: float,
        latency_ms: float,
        upload_bytes_per_sec: float,
        download_bytes_per_sec: float,
        dns_server: str,
    ) -> dict:
        """One tick. `cpu_load` is normalized [0,1] (StateObserver.cpu_load()).
        Layer 1 always runs. The five Layer 2 signatures run only when
        `cpu_load < LAYER2_CPU_GATE` — when gated off they are simply not
        called this tick, so a transient CPU spike doesn't corrupt their
        streak/window state, it just means they sit this tick out.

        Returns the module's usual {anomalous, threat_score, ready, detail}
        shape: `ready` mirrors Layer 1's own `ready` (the always-on floor
        every prior direct-ZScoreBaseline caller relied on); `threat_score`
        is the max — not average — over every active layer whose own
        `ready` is True, so one confirmed signature isn't diluted by four
        quiet ones; `detail` carries a per-layer breakdown for debugging
        and tests, which the state-vector consumer is free to ignore.
        """
        layer1_result = self.layer1.update(
            latency=latency, packet_rate=packet_rate, packet_size=packet_size,
        )

        layer2_active = cpu_load < LAYER2_CPU_GATE
        layer_results = {"layer1": layer1_result}

        if layer2_active:
            layer_results["port_scan"] = self.port_scan.update(
                distinct_ports=distinct_ports,
                connection_attempts=connection_attempts,
                failed_attempts=failed_attempts,
            )
            layer_results["retransmission"] = self.retransmission.update(
                retransmit_rate=retransmit_rate,
            )
            layer_results["mitm_latency"] = self.mitm_latency.update(
                latency_ms=latency_ms,
            )
            layer_results["exfiltration"] = self.exfiltration.update(
                upload_bytes_per_sec=upload_bytes_per_sec,
                download_bytes_per_sec=download_bytes_per_sec,
            )
            layer_results["dns_change"] = self.dns_change.update(
                dns_server=dns_server,
            )

        # Only a ready layer's threat_score counts — an unready layer (not
        # enough history yet) contributes nothing rather than a misleading
        # 0.0 that would mask a real hit from a layer that IS ready.
        contributing = [r for r in layer_results.values() if r["ready"]]
        threat_score = max((r["threat_score"] for r in contributing), default=0.0)
        anomalous = any(r["anomalous"] for r in contributing)

        return {
            "anomalous": anomalous,
            "threat_score": threat_score,
            "ready": layer1_result["ready"],
            "detail": {
                "layer2_active": layer2_active,
                "layers": layer_results,
            },
        }
