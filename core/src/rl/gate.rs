//! The decision gate: debounces the policy's raw per-tick output into the
//! actions the client actually takes.
//!
//! Reference port of `client/rl_agent/decision_gate.py` (Member 3), verified
//! tick-for-tick against `contracts/decision_gate_vectors.json`. The Python
//! module is the source of truth, and its docstrings carry the measurements
//! behind every constant below; change it there, regenerate the vectors, and
//! this file's tests fail until the port follows.
//!
//! Why it exists: the policy is a pure function of a noisy state, and a
//! *change* of algorithm costs a full handshake. Acting on the raw argmax
//! means paying a handshake for sensor noise (measured: 71.0 handshakes/hour
//! raw vs 34.5 gated at ±0.05 noise). So a challenger must win
//! [`CONFIRM_TICKS`] consecutive ticks by at least [`MIN_MARGIN`] before the
//! algorithm changes, while `rekey-now` fires immediately, subject only to a
//! cooldown.

use crate::crypto::MlKemLevel;

/// Seconds between policy ticks.
pub const TICK_SECONDS: u32 = 5;
/// Consecutive ticks a challenger must win before the algorithm changes.
pub const CONFIRM_TICKS: u32 = 3;
/// Probability mass a challenger must lead the in-force algorithm by.
pub const MIN_MARGIN: f32 = 0.15;
/// Ticks after a rekey during which another rekey is suppressed (60 s).
pub const REKEY_COOLDOWN_TICKS: u32 = 12;
/// `contracts/DECISIONS.md` Decision 1, approved at the Week 4 checkpoint: a
/// rekey is performed at the stronger of {in force, the policy's top-ranked
/// KEM}, never a weaker one.
pub const REKEY_ESCALATES: bool = true;

/// Width of the policy's output: ML-KEM-512/768/1024, then rekey-now.
pub const ACTION_COUNT: usize = 4;
/// Output index of `rekey-now` (`contracts/algo_registry.json`).
pub const REKEY_ACTION_INDEX: usize = 3;

// Fixed strings so a tick allocates nothing. They match the Python REASON_*
// constants byte-for-byte; the vector file asserts them.
pub const REASON_AGREES: &str = "policy agrees with in-force";
pub const REASON_LOW_MARGIN: &str = "challenger margin below the minimum";
pub const REASON_ON_STREAK: &str = "challenger on streak, not yet confirmed";
pub const REASON_CONFIRMED: &str = "challenger confirmed for the full streak";
pub const REASON_REKEY: &str = "rekey requested and cooldown clear";
pub const REASON_REKEY_ESCALATED: &str =
    "rekey requested; escalating to the policy's top-ranked KEM";
pub const REASON_REKEY_SUPPRESSED: &str = "rekey suppressed, cooldown still running";

/// One of the policy's four outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyAction {
    Kem(MlKemLevel),
    RekeyNow,
}

impl PolicyAction {
    /// Maps a policy output index to its action. KEM indices equal the wire
    /// codes (`algo_registry.json`'s `action_index`), so this reuses
    /// [`MlKemLevel::from_wire_code`] rather than keeping a second table.
    pub fn from_index(index: usize) -> Option<Self> {
        if index == REKEY_ACTION_INDEX {
            return Some(PolicyAction::RekeyNow);
        }
        u8::try_from(index)
            .ok()
            .and_then(MlKemLevel::from_wire_code)
            .map(PolicyAction::Kem)
    }

    /// Inverse of [`from_index`](Self::from_index).
    pub fn index(self) -> usize {
        match self {
            PolicyAction::Kem(level) => level.wire_code() as usize,
            PolicyAction::RekeyNow => REKEY_ACTION_INDEX,
        }
    }
}

/// Softmax over the ONNX policy's raw logits. The gate's margin test is a
/// probability-mass comparison, so it needs probabilities, not logits.
pub fn softmax(logits: &[f32; ACTION_COUNT]) -> [f32; ACTION_COUNT] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = logits.map(|x| (x - max).exp());
    let sum: f32 = out.iter().sum();
    for p in &mut out {
        *p /= sum;
    }
    out
}

/// What the client should do this tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// The algorithm now in force.
    pub in_force: MlKemLevel,
    /// Renegotiate to `in_force` with a fresh handshake.
    pub change_algorithm: bool,
    /// Re-run the handshake at `in_force`.
    pub rekey: bool,
    /// Why, for the session log. One of the `REASON_*` constants.
    pub reason: &'static str,
}

/// Debounces raw policy output into stable client actions. Construct one per
/// VPN session, starting from the algorithm negotiated at connect time.
#[derive(Debug, Clone)]
pub struct DecisionGate {
    in_force: MlKemLevel,
    confirm_ticks: u32,
    min_margin: f32,
    rekey_cooldown_ticks: u32,
    rekey_escalates: bool,
    candidate: Option<usize>,
    streak: u32,
    rekey_blocked_for: u32,
}

impl DecisionGate {
    /// A gate with the shipped constants.
    pub fn new(in_force: MlKemLevel) -> Self {
        Self::with_params(
            in_force,
            CONFIRM_TICKS,
            MIN_MARGIN,
            REKEY_COOLDOWN_TICKS,
            REKEY_ESCALATES,
        )
    }

    /// A gate with explicit constants, mirroring the Python dataclass fields.
    /// `rekey_escalates = false` restores the Week 2 rekey semantics.
    pub fn with_params(
        in_force: MlKemLevel,
        confirm_ticks: u32,
        min_margin: f32,
        rekey_cooldown_ticks: u32,
        rekey_escalates: bool,
    ) -> Self {
        Self {
            in_force,
            confirm_ticks,
            min_margin,
            rekey_cooldown_ticks,
            rekey_escalates,
            candidate: None,
            streak: 0,
            rekey_blocked_for: 0,
        }
    }

    /// The algorithm currently in force.
    pub fn in_force(&self) -> MlKemLevel {
        self.in_force
    }

    /// Feed one tick of policy output: the softmax over the ONNX logits
    /// (see [`softmax`]).
    pub fn update(&mut self, probs: &[f32; ACTION_COUNT]) -> Decision {
        self.rekey_blocked_for = self.rekey_blocked_for.saturating_sub(1);

        let top = argmax(probs, 0..ACTION_COUNT);
        let in_force_idx = self.in_force.wire_code() as usize;

        // A tick extends the confirm-streak only if it is a genuine algorithm
        // challenger clearing the margin. Every other tick clears the streak
        // here, once, rather than on each return path below. Leaving a streak
        // alive across a rekey was the Week 4 bug.
        let extends_streak = top != REKEY_ACTION_INDEX
            && top != in_force_idx
            && probs[top] - probs[in_force_idx] >= self.min_margin;
        if !extends_streak {
            self.candidate = None;
            self.streak = 0;
        } else if self.candidate == Some(top) {
            self.streak += 1;
        } else {
            self.candidate = Some(top);
            self.streak = 1;
        }

        // rekey: fires immediately, subject only to the cooldown.
        if top == REKEY_ACTION_INDEX {
            if self.rekey_blocked_for > 0 {
                return self.decision(false, false, REASON_REKEY_SUPPRESSED);
            }
            self.rekey_blocked_for = self.rekey_cooldown_ticks;
            let mut reason = REASON_REKEY;
            if self.rekey_escalates {
                let best_kem = level_at(argmax(probs, 0..REKEY_ACTION_INDEX));
                if best_kem > self.in_force {
                    self.in_force = best_kem;
                    reason = REASON_REKEY_ESCALATED;
                }
            }
            return self.decision(false, true, reason);
        }

        // algorithm choice: needs a streak and a margin.
        if top == in_force_idx {
            return self.decision(false, false, REASON_AGREES);
        }
        if !extends_streak {
            return self.decision(false, false, REASON_LOW_MARGIN);
        }
        if self.streak >= self.confirm_ticks {
            self.in_force = level_at(top);
            self.candidate = None;
            self.streak = 0;
            return self.decision(true, false, REASON_CONFIRMED);
        }
        self.decision(false, false, REASON_ON_STREAK)
    }

    fn decision(&self, change_algorithm: bool, rekey: bool, reason: &'static str) -> Decision {
        Decision {
            in_force: self.in_force,
            change_algorithm,
            rekey,
            reason,
        }
    }
}

/// Index of the largest probability in `range`; the first one wins a tie,
/// matching Python's `max(range(...), key=...)`.
fn argmax(probs: &[f32; ACTION_COUNT], range: std::ops::Range<usize>) -> usize {
    let mut best = range.start;
    for i in range {
        if probs[i] > probs[best] {
            best = i;
        }
    }
    best
}

fn level_at(index: usize) -> MlKemLevel {
    match PolicyAction::from_index(index) {
        Some(PolicyAction::Kem(level)) => level,
        _ => unreachable!("index {index} is not a KEM action"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const VECTORS: &str = include_str!("../../../contracts/decision_gate_vectors.json");

    fn vectors() -> Value {
        serde_json::from_str(VECTORS).expect("decision_gate_vectors.json parses")
    }

    fn probs(v: &Value) -> [f32; ACTION_COUNT] {
        let arr = v.as_array().expect("probs is an array");
        assert_eq!(arr.len(), ACTION_COUNT);
        std::array::from_fn(|i| arr[i].as_f64().unwrap() as f32)
    }

    #[test]
    fn replays_every_contract_vector_tick_for_tick() {
        let payload = vectors();
        let cases = payload["cases"].as_array().unwrap();
        assert!(!cases.is_empty(), "no cases in the shipped vector file");

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let initial = case["initial_in_force"].as_u64().unwrap() as u8;
            let mut gate = DecisionGate::new(MlKemLevel::from_wire_code(initial).unwrap());

            for (i, tick) in case["ticks"].as_array().unwrap().iter().enumerate() {
                let d = gate.update(&probs(&tick["probs"]));
                let want = &tick["expect"];
                let at = format!("{name} tick {i}");
                assert_eq!(d.in_force.wire_code() as u64, want["in_force"].as_u64().unwrap(), "{at}");
                assert_eq!(d.in_force.name(), want["in_force_name"].as_str().unwrap(), "{at}");
                assert_eq!(d.change_algorithm, want["change_algorithm"].as_bool().unwrap(), "{at}");
                assert_eq!(d.rekey, want["rekey"].as_bool().unwrap(), "{at}");
                assert_eq!(d.reason, want["reason"].as_str().unwrap(), "{at}");
            }
        }
    }

    #[test]
    fn constants_match_the_contract() {
        let c = &vectors()["constants"];
        assert_eq!(c["tick_seconds"].as_u64(), Some(TICK_SECONDS as u64));
        assert_eq!(c["confirm_ticks"].as_u64(), Some(CONFIRM_TICKS as u64));
        assert_eq!(c["min_margin"].as_f64().map(|m| m as f32), Some(MIN_MARGIN));
        assert_eq!(c["rekey_cooldown_ticks"].as_u64(), Some(REKEY_COOLDOWN_TICKS as u64));
        assert_eq!(c["rekey_escalates"].as_bool(), Some(REKEY_ESCALATES));
    }

    #[test]
    fn rekey_cooldown_expires_after_the_full_cooldown() {
        let mut gate = DecisionGate::new(MlKemLevel::MlKem768);
        let rekey = [0.05, 0.10, 0.05, 0.80];
        assert!(gate.update(&rekey).rekey);
        let fired: Vec<bool> = (0..REKEY_COOLDOWN_TICKS).map(|_| gate.update(&rekey).rekey).collect();
        assert!(!fired[..fired.len() - 1].iter().any(|&f| f));
        assert!(fired[fired.len() - 1], "cooldown should expire after REKEY_COOLDOWN_TICKS");
    }

    #[test]
    fn escalation_can_be_disabled_for_week2_semantics() {
        let mut gate = DecisionGate::with_params(
            MlKemLevel::MlKem512, CONFIRM_TICKS, MIN_MARGIN, REKEY_COOLDOWN_TICKS, false,
        );
        let d = gate.update(&[0.02, 0.10, 0.28, 0.60]);
        assert!(d.rekey);
        assert_eq!(d.in_force, MlKemLevel::MlKem512);
    }

    #[test]
    fn action_index_round_trips() {
        for i in 0..ACTION_COUNT {
            assert_eq!(PolicyAction::from_index(i).unwrap().index(), i);
        }
        assert_eq!(PolicyAction::from_index(ACTION_COUNT), None);
        assert_eq!(PolicyAction::from_index(REKEY_ACTION_INDEX), Some(PolicyAction::RekeyNow));
    }

    #[test]
    fn softmax_is_a_distribution_preserving_argmax() {
        let p = softmax(&[2.0, -1.0, 0.5, 1.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&p, 0..ACTION_COUNT), 0);
        // Large logits must not overflow.
        let p = softmax(&[1000.0, 999.0, 0.0, 0.0]);
        assert!(p.iter().all(|x| x.is_finite()));
    }
}
