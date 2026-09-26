"""
Trains and calibrates anomaly-detection Layer 3 (`IsolationForestLayer`).

There is no captured traffic to train on yet (the live packet source is
Member 1's Week 8 desktop work), so benign traffic is simulated. Each tick
is built packet by packet from one of five profiles (idle, browsing,
streaming, bulk download, video call), then reduced with the same
`tick_features` the live path will use, so the generator exercises the real
feature code rather than skipping it.

Three simulated attack mixes (exfiltration, port scan, beaconing) are used
only to *measure* detection. The forest never sees them: an Isolation
Forest is trained on benign data alone, which is the point of Layer 3.

Every profile below is a reasoned first-pass shape, not a measurement, in
the same spirit as `state_observer.py`'s normalization caps. The numbers are
there to be re-checked once real traffic can be captured, not to be trusted.

Writes:
  client/rl_agent/models/isolation_forest.joblib
  client/rl_agent/models/isolation_forest_calibration.json

Run with: python -m client.rl_agent.train_isolation_forest
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.rl_agent.anomaly_detector import (  # noqa: E402
    LAYER3_CALIBRATION_PATH,
    LAYER3_FEATURES,
    LAYER3_MODEL_PATH,
    IsolationForestLayer,
    tick_features,
)

TICK_SECONDS = 5.0
SEED = 0
TRAIN_TICKS_PER_PROFILE = 1_000
HELDOUT_TICKS_PER_PROFILE = 400
ATTACK_TICKS_PER_MIX = 400
N_ESTIMATORS = 100

THRESHOLD_QUANTILE = 0.995
# The anomaly threshold is the 99.5th percentile of held-out benign scores,
# so about 1 benign tick in 200 crosses it. At 5 s ticks that is one false
# alarm every ~17 minutes *from this layer alone*, and it lands as a
# threat_score near 0.5, not 1.0. Tighter means missed beacons; looser means
# the RL agent reacts to background noise. Re-derive once real traffic exists.

# Per-profile ranges. A tick draws each value uniformly from its range, so a
# profile covers a spread of intensities rather than one point.
#   rate:  packets/sec
#   cv:    coefficient of variation of inter-arrival gaps (gamma-distributed)
#   p_out: share of packets that are outbound
#   out / in: (mean, sd) packet size in bytes per direction
BENIGN_PROFILES = {
    # Keepalives, background sync, the odd DNS lookup.
    "idle":          dict(rate=(1, 5),      cv=(0.8, 1.3), p_out=(0.4, 0.6),
                          out=(120, 60),   inn=(150, 80)),
    # Page loads: bursts of large inbound packets, small outbound requests/ACKs.
    "browsing":      dict(rate=(20, 200),   cv=(1.5, 3.0), p_out=(0.3, 0.45),
                          out=(90, 60),    inn=(1200, 350)),
    # Near-full-size inbound segments, steady-ish, ACKs going out.
    "streaming":     dict(rate=(150, 600),  cv=(0.6, 1.5), p_out=(0.2, 0.35),
                          out=(66, 10),    inn=(1400, 120)),
    # MTU-sized inbound as fast as the link allows.
    "bulk_download": dict(rate=(500, 2000), cv=(0.4, 1.0), p_out=(0.3, 0.5),
                          out=(66, 8),     inn=(1460, 40)),
    # Symmetric mid-size media frames at a fairly regular cadence.
    "video_call":    dict(rate=(80, 300),   cv=(0.3, 0.7), p_out=(0.45, 0.55),
                          out=(1000, 200), inn=(1000, 200)),
}

ATTACK_MIXES = {
    # Sustained MTU-sized outbound, almost nothing coming back but ACKs.
    "exfiltration":  dict(rate=(300, 1500), cv=(0.3, 1.0), p_out=(0.7, 0.85),
                          out=(1450, 50),  inn=(66, 8)),
    # SYN out, RST back: tiny uniform packets at a fast, machine-regular pace.
    "port_scan":     dict(rate=(200, 2000), cv=(0.1, 0.4), p_out=(0.5, 0.65),
                          out=(58, 4),     inn=(54, 4)),
    # C2 check-ins on a timer: low rate, near-zero jitter, mostly outbound.
    "beaconing":     dict(rate=(1, 3),      cv=(0.0, 0.05), p_out=(0.6, 0.9),
                          out=(250, 40),   inn=(180, 40)),
}


def simulate_tick(profile: dict, rng: np.random.Generator):
    """One tick's packets for `profile`: (sizes, arrival_times, outbound)."""
    rate = rng.uniform(*profile["rate"])
    cv = max(rng.uniform(*profile["cv"]), 0.01)
    p_out = rng.uniform(*profile["p_out"])

    n = max(int(rng.poisson(rate * TICK_SECONDS)), 1)
    # Gamma gaps with shape k have cv = 1/sqrt(k), and mean 1/rate.
    shape = 1.0 / cv ** 2
    gaps = rng.gamma(shape, 1.0 / (rate * shape), size=n)
    times = np.cumsum(gaps)

    outbound = rng.random(n) < p_out
    sizes = np.where(
        outbound,
        rng.normal(*profile["out"], size=n),
        rng.normal(*profile["inn"], size=n),
    )
    sizes = np.clip(sizes, 40, 1500)
    return sizes, times, outbound


def feature_matrix(profile: dict, n_ticks: int, rng: np.random.Generator) -> np.ndarray:
    """`n_ticks` feature rows for `profile`. Ticks below LAYER3_MIN_PACKETS
    are skipped, exactly as the live layer would skip them."""
    rows = []
    while len(rows) < n_ticks:
        f = tick_features(*simulate_tick(profile, rng), tick_seconds=TICK_SECONDS)
        if f is not None:
            rows.append([f[name] for name in LAYER3_FEATURES])
    return np.array(rows)


def benign_matrix(n_per_profile: int, rng: np.random.Generator) -> np.ndarray:
    return np.vstack([feature_matrix(p, n_per_profile, rng) for p in BENIGN_PROFILES.values()])


def train(seed: int = SEED, n_per_profile: int = TRAIN_TICKS_PER_PROFILE) -> IsolationForest:
    rng = np.random.default_rng(seed)
    x = benign_matrix(n_per_profile, rng)
    return IsolationForest(n_estimators=N_ESTIMATORS, contamination="auto",
                           random_state=seed).fit(x)


def calibrate_and_measure(model: IsolationForest, seed: int = SEED) -> dict:
    """Calibrate the thresholds on held-out benign traffic (a different seed
    stream from training), then measure per-profile false positives and
    per-attack detection with those thresholds."""
    rng = np.random.default_rng(seed + 1)
    heldout = {name: feature_matrix(p, HELDOUT_TICKS_PER_PROFILE, rng)
               for name, p in BENIGN_PROFILES.items()}
    benign_scores = -model.score_samples(np.vstack(list(heldout.values())))
    median = float(np.median(benign_scores))
    threshold = float(np.quantile(benign_scores, THRESHOLD_QUANTILE))

    def flag_rate(x):
        return float(np.mean(-model.score_samples(x) >= threshold))

    attacks = {name: feature_matrix(p, ATTACK_TICKS_PER_MIX, rng)
               for name, p in ATTACK_MIXES.items()}
    return {
        "score_median": median,
        "score_threshold": threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "benign_false_positive_rate": {n: flag_rate(x) for n, x in heldout.items()},
        "attack_detection_rate": {n: flag_rate(x) for n, x in attacks.items()},
    }


def main() -> int:
    model = train()
    measured = calibrate_and_measure(model)

    LAYER3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LAYER3_MODEL_PATH, compress=3)
    calibration = {
        "generated_by": "python -m client.rl_agent.train_isolation_forest",
        "features": list(LAYER3_FEATURES),
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "n_estimators": N_ESTIMATORS,
        "train_ticks": TRAIN_TICKS_PER_PROFILE * len(BENIGN_PROFILES),
        "note": ("Trained and measured on simulated traffic only; the profiles "
                 "in train_isolation_forest.py are first-pass shapes, not "
                 "captures. Re-derive once real traffic is available."),
        **measured,
    }
    LAYER3_CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n")

    # Load back through the real layer, so a save/load mismatch fails here.
    IsolationForestLayer()

    print(f"wrote {LAYER3_MODEL_PATH.name} ({LAYER3_MODEL_PATH.stat().st_size / 1024:.0f} KB) "
          f"and {LAYER3_CALIBRATION_PATH.name}")
    print(f"score median {measured['score_median']:.4f}, "
          f"threshold {measured['score_threshold']:.4f} (q{THRESHOLD_QUANTILE})")
    print("benign false-positive rate (held-out):")
    for name, rate in measured["benign_false_positive_rate"].items():
        print(f"  {name:<14} {rate:6.2%}")
    print("attack detection rate:")
    for name, rate in measured["attack_detection_rate"].items():
        print(f"  {name:<14} {rate:6.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
