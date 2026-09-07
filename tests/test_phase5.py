"""
Week 5 (Member 3 track): Layer 2 rule-based anomaly signatures — one
independent stateful check per known attack pattern (port scan,
retransmission spike, MitM-latency, bandwidth exfiltration, DNS-server
change), plus the shared `_ratio_score` squash and a structural drift-guard
on the common return shape all six detectors (Layer 1 + the five Layer 2
signatures) share.

Each signature gets a fires-on-attack / doesn't-fire-on-normal-traffic pair,
mirroring `test_anomaly_detector_flags_injected_outlier`'s existing style for
Layer 1. Where a signature requires consecutive ticks, a third test proves
the streak gate actually gates (one bad tick alone must not fire).

Wiring these into the CPU-gated combiner alongside Layer 1, and testing that
integration, is Week 6's job — these tests only exercise each class in
isolation, the same way ZScoreBaseline was tested standalone when it shipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    BandwidthExfiltrationSignature, DNSServerChangeSignature,
    MitMLatencySignature, PortScanSignature, RetransmissionSpikeSignature,
    ZScoreBaseline, _ratio_score,
)


# --------------------------------------------------------------- port scan

def test_port_scan_fires_on_synthetic_scan_pattern():
    sig = PortScanSignature()
    result = sig.update(distinct_ports=20, connection_attempts=20, failed_attempts=18)
    assert result["anomalous"]
    assert result["threat_score"] > 0.5


def test_port_scan_does_not_fire_on_normal_traffic():
    sig = PortScanSignature()
    result = sig.update(distinct_ports=3, connection_attempts=10, failed_attempts=1)
    assert not result["anomalous"]


def test_port_scan_does_not_fire_on_high_fanout_with_low_failure():
    """High fanout alone isn't a scan if every connection also succeeds —
    e.g. a legitimately chatty application."""
    sig = PortScanSignature()
    result = sig.update(distinct_ports=25, connection_attempts=25, failed_attempts=1)
    assert not result["anomalous"]


# --------------------------------------------------------- retransmission

def test_retransmission_spike_fires_on_sustained_high_retransmit_rate():
    sig = RetransmissionSpikeSignature()
    sig.update(0.25)
    result = sig.update(0.25)
    assert result["anomalous"]


def test_retransmission_spike_does_not_fire_on_single_transient_tick():
    """One bad tick, then normal — must not fire, proving the
    consecutive-ticks gate actually gates."""
    sig = RetransmissionSpikeSignature()
    result = sig.update(0.25)
    assert not result["anomalous"]
    result = sig.update(0.01)
    assert not result["anomalous"]


def test_retransmission_spike_does_not_fire_on_normal_traffic():
    sig = RetransmissionSpikeSignature()
    for _ in range(5):
        result = sig.update(0.01)
    assert not result["anomalous"]


# ----------------------------------------------------------------- MitM

def test_mitm_latency_signature_fires_on_sustained_jitter():
    sig = MitMLatencySignature()
    for latency in (20.0, 20.0, 150.0):
        result = sig.update(latency)
    assert not result["anomalous"]  # only the first jittery window so far
    result = sig.update(20.0)
    assert result["anomalous"]


def test_mitm_latency_signature_does_not_fire_on_stable_latency():
    sig = MitMLatencySignature()
    for latency in (22.0, 24.0, 21.0, 23.0, 22.0):
        result = sig.update(latency)
    assert not result["anomalous"]


def test_mitm_latency_signature_not_ready_before_window_fills():
    sig = MitMLatencySignature()
    result = sig.update(20.0)
    assert not result["ready"]


# ------------------------------------------------------ bandwidth/exfil

def test_bandwidth_exfiltration_fires_on_sustained_upload_spike_with_high_ratio():
    sig = BandwidthExfiltrationSignature()
    sig.update(upload_bytes_per_sec=3_000_000.0, download_bytes_per_sec=100_000.0)
    result = sig.update(upload_bytes_per_sec=3_000_000.0, download_bytes_per_sec=100_000.0)
    assert result["anomalous"]


def test_bandwidth_exfiltration_does_not_fire_on_download_heavy_traffic():
    sig = BandwidthExfiltrationSignature()
    for _ in range(3):
        result = sig.update(upload_bytes_per_sec=200_000.0, download_bytes_per_sec=5_000_000.0)
    assert not result["anomalous"]


def test_bandwidth_exfiltration_does_not_fire_on_single_burst_tick():
    """A single large-file send shouldn't fire — sustained means still
    happening ticks later."""
    sig = BandwidthExfiltrationSignature()
    result = sig.update(upload_bytes_per_sec=3_000_000.0, download_bytes_per_sec=100_000.0)
    assert not result["anomalous"]
    result = sig.update(upload_bytes_per_sec=200_000.0, download_bytes_per_sec=1_000_000.0)
    assert not result["anomalous"]


# ------------------------------------------------------------ DNS change

def test_dns_server_change_does_not_fire_before_baseline_established():
    sig = DNSServerChangeSignature()
    result = sig.update("1.1.1.1")
    assert not result["anomalous"]
    assert not result["ready"]


def test_dns_server_change_does_not_fire_when_unchanged():
    sig = DNSServerChangeSignature()
    sig.update("1.1.1.1")
    result = sig.update("1.1.1.1")
    assert not result["anomalous"]


def test_dns_server_change_fires_on_change_after_baseline():
    sig = DNSServerChangeSignature()
    sig.update("1.1.1.1")
    result = sig.update("10.0.0.53")
    assert result["anomalous"]
    assert result["threat_score"] == 1.0


# -------------------------------------------------------- shared helper

def test_ratio_score_helper_clips_and_saturates():
    assert _ratio_score(0.0, 10.0) == 0.0
    assert _ratio_score(10.0, 10.0) == 0.5     # at the threshold itself
    assert _ratio_score(20.0, 10.0) == 1.0      # at ceiling (2x threshold)
    assert _ratio_score(1000.0, 10.0) == 1.0    # saturates, doesn't overshoot
    assert _ratio_score(5.0, 0.0) == 0.0        # no divide-by-zero


# --------------------------------------------------- common return shape

def test_all_layer2_detectors_share_the_common_return_shape():
    """Pins the contract Week 6's combiner is about to depend on: a future
    edit that silently changes one class's return shape fails here instead
    of at combiner integration."""
    expected_keys = {"anomalous", "threat_score", "ready", "detail"}

    detectors_and_calls = [
        (PortScanSignature(), lambda d: d.update(3, 10, 1)),
        (RetransmissionSpikeSignature(), lambda d: d.update(0.01)),
        (MitMLatencySignature(), lambda d: d.update(20.0)),
        (BandwidthExfiltrationSignature(), lambda d: d.update(200_000.0, 5_000_000.0)),
        (DNSServerChangeSignature(), lambda d: d.update("1.1.1.1")),
    ]
    for detector, call in detectors_and_calls:
        result = call(detector)
        assert set(result.keys()) == expected_keys, type(detector).__name__
        assert isinstance(result["anomalous"], bool)
        assert isinstance(result["threat_score"], float)
        assert 0.0 <= result["threat_score"] <= 1.0
        assert isinstance(result["ready"], bool)
        assert isinstance(result["detail"], dict)


def test_zscore_baseline_still_importable_alongside_layer2():
    """Layer 1 and Layer 2 now share a module — a smoke check that the
    append didn't disturb the existing class."""
    detector = ZScoreBaseline(window=5)
    for _ in range(5):
        result = detector.update(latency=0.1, packet_rate=0.1, packet_size=0.1)
    assert result["ready"]
    assert not result["anomalous"]
