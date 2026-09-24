//! Client side of the PQC-VPN handshake wire protocol.
//!
//! Full specification: `server/PROTOCOL.md` (FROZEN v1). That document
//! names this module directly: *"For Member 1: build the client side of
//! `core/protocol/` against this document and verify it against
//! `server/handshake-vectors.json`"* — Member 1 Week 6
//! (`TEAM_TIMELINE_PROPOSAL.md`).
//!
//! Layout:
//! - [`wire`] — message framing: encode/decode, no I/O, no crypto.
//! - `tags` (private) — the `ClientFinish` / `ServerFinish` confirmation
//!   MACs (`PROTOCOL.md` §5.3), the one piece of the key schedule that
//!   lived only server-side before this module existed.
//! - `client` (private) — the handshake state machine: generate keys, build
//!   `ClientHello`, verify `ServerHello` against a pinned server identity,
//!   derive session keys.
//! - [`transport`] — the only place in `core` that opens a real
//!   `TcpStream`; runs the state machine above against whatever is
//!   actually listening (`server/handshake-server` today, a mock in a
//!   test).
//!
//! Everything above `wire` is built entirely on `crate::crypto` (the Week 2
//! port) and `crate::state::TunnelConfig` (Week 5) — this module adds the
//! wire format and the network round trip, not new cryptography.
//!
//! What this module deliberately does not do yet: choose *when* to
//! (re)handshake (that is `rl/`, Week 7) or bring the local WireGuard
//! interface up with the result (that is `desktop/`'s `TunnelHandle`,
//! Week 9). [`transport::connect`] hands back a [`crate::state::TunnelConfig`]
//! and stops there.

pub mod wire;

mod client;
mod tags;
mod transport;

pub use client::{ClientHandshake, HandshakeError, VERIFYING_KEY_LEN};
pub use transport::{ProtocolError, connect, DEFAULT_TIMEOUT};
