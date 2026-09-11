"""
Week 6 (Member 3 track): the anomaly combiner.

Layer 1 (`ZScoreBaseline`) and the five Layer 2 signatures
(`tests/test_phase5.py`) were each verified standalone; this file is the
piece `test_phase5.py` explicitly deferred — "wiring these into the
CPU-gated combiner alongside Layer 1, and testing that integration, is
Week 6's job." `AnomalyCombiner` is the thing under test here, not the
individual signatures again.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    AnomalyCombiner, LAYER2_CPU_GATE, ZScoreBaseline,
)

# A benign tick for every metric the combiner's update() needs, so each test
# only has to override the field(s) it cares about.
NORMAL = dict(
    latency=0.1, packet_rate=0.1, packet_size=0.1,
    distinct_ports=2, connection_attempts=2, failed_attempts=0,
    retransmit_rate=0.0, latency_ms=10.0,
    upload_bytes_per_sec=1_000.0, download_bytes_per_sec=1_000.0,
    dns_server="1.1.1.1",
)

PORT_SCAN = dict(distinct_ports=20, connection_attempts=20, failed_attempts=18)


def tick(combiner, cpu_load, **overrides):
    kwargs = {**NORMAL, **overrides}
    return combiner.update(cpu_load=cpu_load, **kwargs)


# --------------------------------------------------------------- CPU gate

def test_layer2_active_and_can_fire_below_the_gate():
    c = AnomalyCombiner()
    result = tick(c, LAYER2_CPU_GATE - 0.01, **PORT_SCAN)
    assert result["detail"]["layer2_active"]
    assert result["anomalous"]
    assert result["threat_score"] > 0.0


def test_layer2_gated_off_at_or_above_the_gate():
    c = AnomalyCombiner()
    # Same synthetic scan pattern, but CPU is at the gate itself.
    result = tick(c, LAYER2_CPU_GATE, **PORT_SCAN)
    assert not result["detail"]["layer2_active"]
    assert "port_scan" not in result["detail"]["layers"]
    assert not result["anomalous"]
    assert result["threat_score"] == 0.0


# ------------------------------------------------------ combining scores

def test_combined_threat_score_is_the_max_not_a_sum():
    c = AnomalyCombiner()
    # Port scan and bandwidth exfiltration both cross their lines in the
    # same tick. If the combiner summed/averaged instead of taking the max,
    # this would either exceed [0,1] or under-report either signature alone.
    result = tick(
        c, 0.1,
        distinct_ports=20, connection_attempts=20, failed_attempts=18,
        upload_bytes_per_sec=5_000_000.0, download_bytes_per_sec=100.0,
    )
    layers = result["detail"]["layers"]
    expected_max = max(layers["port_scan"]["threat_score"],
                        layers["exfiltration"]["threat_score"])
    assert result["threat_score"] == expected_max
    assert result["threat_score"] <= 1.0
    assert result["anomalous"]


def test_regression_parity_with_bare_zscore_baseline_when_layer2_is_quiet():
    """With CPU below the gate but no Layer 2 signature triggering,
    AnomalyCombiner must reproduce exactly what calling ZScoreBaseline
    directly already produced — this is what makes swapping it into
    demo.py's run_round() safe."""
    combiner = AnomalyCombiner(window=20)
    baseline = ZScoreBaseline(window=20)

    for _ in range(25):
        combiner.update(cpu_load=0.2, **{**NORMAL, "latency": 0.1,
                                          "packet_rate": 0.1, "packet_size": 0.1})
        baseline.update(latency=0.1, packet_rate=0.1, packet_size=0.1)

    combined = combiner.update(cpu_load=0.2, **{**NORMAL, "latency": 0.1,
                                                 "packet_rate": 0.1, "packet_size": 50.0})
    direct = baseline.update(latency=0.1, packet_rate=0.1, packet_size=50.0)

    assert combined["threat_score"] == direct["threat_score"]
    assert combined["anomalous"] == direct["anomalous"]
    assert combined["ready"] == direct["ready"]


# ------------------------------------------------ gating vs. streak state

def test_gated_off_tick_neither_resets_nor_advances_a_streak():
    c = AnomalyCombiner()
    # Tick 1: high retransmit rate, CPU low -> streak = 1 (not yet anomalous,
    # RETRANS_CONSECUTIVE_TICKS is 2).
    r1 = tick(c, 0.1, retransmit_rate=0.5)
    assert not r1["anomalous"]

    # Tick 2: CPU spikes above the gate -> retransmission signature isn't
    # consulted at all this tick. A normal (low) tick here would reset the
    # streak to 0; this must not.
    r2 = tick(c, 0.9, retransmit_rate=0.0)
    assert not r2["detail"]["layer2_active"]

    # Tick 3: CPU back down, still high retransmit rate -> if the streak
    # survived the gated tick untouched, this is the second consecutive
    # qualifying tick and it fires.
    r3 = tick(c, 0.1, retransmit_rate=0.5)
    assert r3["anomalous"]
    assert r3["detail"]["layers"]["retransmission"]["detail"]["streak"] == 2


# ------------------------------------------------------------- readiness

def test_not_ready_layer2_signature_does_not_suppress_a_real_layer1_hit():
    c = AnomalyCombiner(window=20)
    # Pre-warm Layer 1 only (bypassing update() so mitm_latency's own
    # 3-tick window stays untouched) — Layer 1's window (20) is wider than
    # MitM's (3), so there's no way to reach it "naturally" through
    # combiner.update() without MitM also becoming ready first.
    for _ in range(20):
        c.layer1.update(latency=0.1, packet_rate=0.1, packet_size=0.1)

    result = tick(c, 0.1, packet_size=50.0)  # Layer 1 outlier, MitM's first tick
    mitm = result["detail"]["layers"]["mitm_latency"]
    assert not mitm["ready"]
    assert result["anomalous"]  # Layer 1's hit still comes through
    assert result["threat_score"] > 0.0


# --------------------------------------------------------------- shape

def test_combiner_result_matches_the_shared_return_shape():
    c = AnomalyCombiner()
    result = tick(c, 0.1)
    assert set(result.keys()) == {"anomalous", "threat_score", "ready", "detail"}
    assert isinstance(result["anomalous"], bool)
    assert 0.0 <= result["threat_score"] <= 1.0
    assert isinstance(result["ready"], bool)
    assert "layer2_active" in result["detail"]
    assert "layers" in result["detail"]
    assert "layer1" in result["detail"]["layers"]
