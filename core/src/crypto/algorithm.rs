/// The ML-KEM security levels supported by our application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlKemLevel {
    MlKem512,
    MlKem768,
    MlKem1024,
}

impl MlKemLevel {
    /// Returns the standard display name of the selected algorithm.
    pub fn name(&self) -> &'static str {
        match self {
            MlKemLevel::MlKem512 => "ML-KEM-512",
            MlKemLevel::MlKem768 => "ML-KEM-768",
            MlKemLevel::MlKem1024 => "ML-KEM-1024",
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
}
