/// The ML-KEM security levels supported by our application.
///
/// `Ord` is derived in wire-strength order (512 < 768 < 1024) on purpose —
/// `server/PROTOCOL.md` §4.6/§6 (`REKEY_ESCALATES`) needs to compare a
/// rekey's requested level against the session's in-force level without
/// either side hand-rolling the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MlKemLevel {
    MlKem512,
    MlKem768,
    MlKem1024,
}

impl MlKemLevel {
    /// Returns the standard display name of the selected algorithm. Also the
    /// exact bytes `core::protocol` and the server sign into the handshake
    /// transcript (`server/PROTOCOL.md` §5.1) — do not reformat.
    pub fn name(&self) -> &'static str {
        match self {
            MlKemLevel::MlKem512 => "ML-KEM-512",
            MlKemLevel::MlKem768 => "ML-KEM-768",
            MlKemLevel::MlKem1024 => "ML-KEM-1024",
        }
    }

    /// The one-byte algorithm code on the wire (`server/PROTOCOL.md` §4.1),
    /// matching `contracts/algo_registry.json`'s `action_index`.
    pub fn wire_code(self) -> u8 {
        match self {
            MlKemLevel::MlKem512 => 0x00,
            MlKemLevel::MlKem768 => 0x01,
            MlKemLevel::MlKem1024 => 0x02,
        }
    }

    /// Inverse of [`wire_code`](Self::wire_code).
    pub fn from_wire_code(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(MlKemLevel::MlKem512),
            0x01 => Some(MlKemLevel::MlKem768),
            0x02 => Some(MlKemLevel::MlKem1024),
            _ => None,
        }
    }

    /// Encoded ML-KEM encapsulation-key length in bytes (client → server).
    pub fn mlkem_pub_len(self) -> usize {
        match self {
            MlKemLevel::MlKem512 => 800,
            MlKemLevel::MlKem768 => 1184,
            MlKemLevel::MlKem1024 => 1568,
        }
    }

    /// Encoded ML-KEM ciphertext length in bytes (server → client).
    pub fn ciphertext_len(self) -> usize {
        match self {
            MlKemLevel::MlKem512 => 768,
            MlKemLevel::MlKem768 => 1088,
            MlKemLevel::MlKem1024 => 1568,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MlKemLevel;

    #[test]
    fn returns_correct_ml_kem_names() {
        assert_eq!(MlKemLevel::MlKem512.name(), "ML-KEM-512");
        assert_eq!(MlKemLevel::MlKem768.name(), "ML-KEM-768");
        assert_eq!(MlKemLevel::MlKem1024.name(), "ML-KEM-1024");
    }

    #[test]
    fn wire_code_round_trips() {
        for level in [MlKemLevel::MlKem512, MlKemLevel::MlKem768, MlKemLevel::MlKem1024] {
            assert_eq!(MlKemLevel::from_wire_code(level.wire_code()), Some(level));
        }
        assert_eq!(MlKemLevel::from_wire_code(0x7f), None);
    }

    #[test]
    fn strength_ordering_supports_no_downgrade_checks() {
        assert!(MlKemLevel::MlKem512 < MlKemLevel::MlKem768);
        assert!(MlKemLevel::MlKem768 < MlKemLevel::MlKem1024);
    }
}
