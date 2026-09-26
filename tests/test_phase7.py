"""
Week 7 (Member 3 track): anomaly-detection Layer 3, the Isolation Forest.

Pins the feature contract the Week 8 Rust port will be built against, the
feature extraction itself, and floors on the measured behaviour of the
shipped model. The floors are set below what was measured across seeds 0-3
(benign FP 0.5% overall / idle <= 2.5%; exfiltration 100%, port scan
>= 98.8%, beaconing 81.5-88.5%) so sampling noise cannot fail the suite, and
well above what a broken model would score.

Scored on fresh seeds the model was neither trained nor calibrated on.
Combiner wiring is Week 8 and is not tested here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    LAYER3_CALIBRATION_PATH, LAYER3_CPU_GATE, LAYER3_FEATURES, LAYER3_MIN_PACKETS,
    LAYER2_CPU_GATE, IsolationForestLayer, tick_features,
)
from client.rl_agent import train_isolation_forest as tif  # noqa: E402

EVAL_SEED = 1234
TICKS = 150


@pytest.fixture(scope="module")
def layer():
    return IsolationForestLayer()


def _flag_rate(layer, profile, seed):
    rng = np.random.default_rng(seed)
    results = []
    while len(results) < TICKS:
        f = tick_features(*tif.simulate_tick(profile, rng))
        if f is not None:
            results.append(layer.update(f))
    return results, float(np.mean([r["anomalous"] for r in results]))


# ------------------------------------------------------- feature contract

def test_feature_order_is_pinned():
    """The saved model and the Week 8 Rust port index features by position.
    Reordering or renaming one silently scrambles every score, so a change
    here must be deliberate: retrain, recalibrate, then update this list."""
    assert LAYER3_FEATURES == (
        "out_size_mean", "in_size_mean", "pkt_size_std",
        "log_pkt_rate", "iat_cv", "upload_share",
    )


def test_calibration_file_matches_the_module_feature_order():
    cal = json.loads(LAYER3_CALIBRATION_PATH.read_text())
    assert tuple(cal["features"]) == LAYER3_FEATURES
    assert cal["score_median"] < cal["score_threshold"]


def test_layer_refuses_a_calibration_with_a_different_feature_order(layer):
    cal = json.loads(LAYER3_CALIBRATION_PATH.read_text())
    cal["features"] = list(reversed(cal["features"]))
    with pytest.raises(ValueError):
        IsolationForestLayer(model=layer.model, calibration=cal)


def test_layer3_gate_is_stricter_than_layer2():
    """Proposal 3.3: the heavier the layer, the lower the CPU it needs."""
    assert 0.0 < LAYER3_CPU_GATE < LAYER2_CPU_GATE


# ----------------------------------------------------- feature extraction

def test_tick_features_on_a_hand_built_tick():
    sizes = [100, 1500, 100, 1500]
    times = [0.0, 1.0, 2.0, 3.0]            # perfectly regular
    outbound = [True, False, True, False]
    f = tick_features(sizes, times, outbound, tick_seconds=5.0)
    assert f["out_size_mean"] == pytest.approx(100.0)
    assert f["in_size_mean"] == pytest.approx(1500.0)
    assert f["pkt_size_std"] == pytest.approx(700.0)
    assert f["log_pkt_rate"] == pytest.approx(np.log10(1 + 4 / 5.0))
    assert f["iat_cv"] == pytest.approx(0.0)
    assert f["upload_share"] == pytest.approx(200 / 3200)


def test_tick_features_sorts_arrival_times():
    a = tick_features([60, 60, 60], [0.0, 1.0, 3.0], [True] * 3)
    b = tick_features([60, 60, 60], [3.0, 0.0, 1.0], [True] * 3)
    assert a == b


def test_one_direction_only_reports_zero_for_the_missing_direction():
    f = tick_features([200, 200, 200], [0.0, 1.0, 2.0], [True] * 3)
    assert f["in_size_mean"] == 0.0
    assert f["upload_share"] == pytest.approx(1.0)


def test_sparse_tick_is_not_ready(layer):
    sparse = tick_features([60] * (LAYER3_MIN_PACKETS - 1),
                           list(range(LAYER3_MIN_PACKETS - 1)),
                           [True] * (LAYER3_MIN_PACKETS - 1))
    assert sparse is None
    r = layer.update(sparse)
    assert r == {"anomalous": False, "threat_score": 0.0, "ready": False,
                 "detail": {"raw_score": None}}


# ------------------------------------------------------ measured behaviour

@pytest.mark.parametrize("name", list(tif.BENIGN_PROFILES))
def test_benign_traffic_rarely_fires(layer, name):
    _, rate = _flag_rate(layer, tif.BENIGN_PROFILES[name], EVAL_SEED)
    # idle is the noisiest profile (5-25 packets a tick) and measured ~2.5%.
    assert rate <= (0.06 if name == "idle" else 0.02), f"{name} false-positive rate {rate:.1%}"


@pytest.mark.parametrize("name,floor", [
    ("exfiltration", 0.95),
    ("port_scan", 0.95),
    ("beaconing", 0.70),
])
def test_attack_mixes_fire(layer, name, floor):
    _, rate = _flag_rate(layer, tif.ATTACK_MIXES[name], EVAL_SEED)
    assert rate >= floor, f"{name} detection rate {rate:.1%}"


def test_scores_are_in_range_and_shaped_like_the_other_layers(layer):
    for profile in [*tif.BENIGN_PROFILES.values(), *tif.ATTACK_MIXES.values()]:
        results, _ = _flag_rate(layer, profile, EVAL_SEED)
        for r in results:
            assert set(r) == {"anomalous", "threat_score", "ready", "detail"}
            assert 0.0 <= r["threat_score"] <= 1.0
            assert r["ready"]


class _FixedScore:
    """Stands in for the forest so `update`'s squash can be checked exactly."""
    def __init__(self, raw):
        self.raw = raw

    def score_samples(self, x):
        return np.array([-self.raw])


def test_threat_score_squash_matches_the_other_layers():
    """Same squash as Layers 1 and 2: 0 at the benign median, 0.5 exactly at
    the trigger point, saturating at twice that distance."""
    cal = {"features": list(LAYER3_FEATURES), "score_median": 0.5, "score_threshold": 0.6}
    f = dict.fromkeys(LAYER3_FEATURES, 0.0)
    for raw, want_threat, want_anomalous in [
        (0.40, 0.0, False), (0.50, 0.0, False), (0.55, 0.25, False),
        (0.60, 0.5, True), (0.70, 1.0, True), (0.90, 1.0, True),
    ]:
        r = IsolationForestLayer(model=_FixedScore(raw), calibration=cal).update(f)
        assert r["threat_score"] == pytest.approx(want_threat), raw
        assert r["anomalous"] is want_anomalous, raw


def test_training_is_deterministic_under_a_seed():
    x = tif.benign_matrix(20, np.random.default_rng(7))
    a = tif.train(seed=3, n_per_profile=100).score_samples(x)
    b = tif.train(seed=3, n_per_profile=100).score_samples(x)
    assert np.array_equal(a, b)
