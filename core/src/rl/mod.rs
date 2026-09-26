//! RL decision layer: turns the policy's output into client actions.
//!
//! - [`gate`] — the decision gate (Week 7), ported from
//!   `client/rl_agent/decision_gate.py` and verified against
//!   `contracts/decision_gate_vectors.json`.
//!
//! ONNX inference over `ppo_vpn_agent.onnx` (via `ort`) is Member 1's Week 7
//! wiring: run the model, [`gate::softmax`] the logits, feed the gate.

pub mod gate;

pub use gate::{softmax, Decision, DecisionGate, PolicyAction, ACTION_COUNT, REKEY_ACTION_INDEX};
