//! Error types. `WireError` maps to `PROTOCOL.md` §4.7 error codes.

use std::fmt;

/// Failure decoding or encoding a wire message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// Frame `VER` byte was not `PROTOCOL_VERSION`.
    UnsupportedVersion(u8),
    /// Frame `TYPE` byte is not a known message type.
    UnknownMsgType(u8),
    /// `LENGTH` exceeds `MAX_PAYLOAD`, or a length-prefixed field overruns the buffer.
    LengthOutOfRange,
    /// Payload ended before all declared fields were read.
    Truncated,
    /// A field had the wrong fixed size, or a value was not a valid enum.
    Invalid(&'static str),
}

impl WireError {
    /// `PROTOCOL.md` §4.7 code, for an `Error` reply to the peer.
    pub fn protocol_code(&self) -> u8 {
        match self {
            WireError::UnsupportedVersion(_) => 0x01, // UNSUPPORTED_VERSION
            _ => 0x02,                                // MALFORMED
        }
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireError::UnsupportedVersion(v) => write!(f, "unsupported protocol version {v:#04x}"),
            WireError::UnknownMsgType(t) => write!(f, "unknown message type {t:#04x}"),
            WireError::LengthOutOfRange => write!(f, "declared length out of range"),
            WireError::Truncated => write!(f, "message truncated"),
            WireError::Invalid(what) => write!(f, "invalid {what}"),
        }
    }
}

impl std::error::Error for WireError {}

/// Failure completing a handshake (crypto, auth, or protocol-state error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandshakeError {
    /// Server policy will not serve the requested algorithm level.
    AlgoRejected,
    /// Rekey referenced a `session_id` the server has no record of.
    UnknownSession,
    /// Rekey `algo` is weaker than in force, or stronger without `REKEY_ESCALATES`.
    AlgoMismatch,
    /// The client's confirmation tag did not verify.
    AuthFailed,
    /// A `core::crypto` operation failed (bad key encoding, etc).
    Crypto(String),
}

impl HandshakeError {
    /// `PROTOCOL.md` §4.7 code.
    pub fn protocol_code(&self) -> u8 {
        match self {
            HandshakeError::AlgoRejected => 0x03,
            HandshakeError::UnknownSession => 0x04,
            HandshakeError::AlgoMismatch => 0x05,
            HandshakeError::AuthFailed => 0x06,
            HandshakeError::Crypto(_) => 0x08, // INTERNAL
        }
    }
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandshakeError::AlgoRejected => write!(f, "algorithm rejected by server policy"),
            HandshakeError::UnknownSession => write!(f, "unknown session id"),
            HandshakeError::AlgoMismatch => write!(f, "rekey algorithm mismatch"),
            HandshakeError::AuthFailed => write!(f, "client authentication failed"),
            HandshakeError::Crypto(m) => write!(f, "crypto error: {m}"),
        }
    }
}

impl std::error::Error for HandshakeError {}

impl From<WireError> for HandshakeError {
    fn from(e: WireError) -> Self {
        HandshakeError::Crypto(e.to_string())
    }
}
