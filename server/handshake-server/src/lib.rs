//! PQC-VPN server-side handshake responder — reference implementation of
//! `server/PROTOCOL.md` v1.
//!
//! Layering:
//! - [`wire`]            — message framing and encode/decode (`PROTOCOL.md` §3–4)
//! - [`transcript`]      — the byte string the server signs (`PROTOCOL.md` §5.1)
//! - [`kdf`]             — `hybrid_secret` → `{psk, confirm_key}` and the finish MACs (§5.3)
//! - [`identity`]        — the server's long-term ML-DSA-65 key, persisted to disk (§5.2)
//! - [`handshake`]       — server side: `ClientHello` → `ServerHello`
//! - [`client_handshake`]— client side, for the test client and vector emitter
//! - [`session`]         — in-memory session table
//!
//! The cryptographic primitives are all `vpn_core::crypto` (Member 1's port).
//! This crate adds only the wire protocol around them.

pub mod client_handshake;
pub mod error;
pub mod handshake;
pub mod identity;
pub mod kdf;
pub mod server;
pub mod session;
pub mod transcript;
pub mod tunnel;
pub mod wire;

pub use error::{HandshakeError, WireError};
