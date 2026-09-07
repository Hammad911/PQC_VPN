use std::collections::HashMap;
use zeroize::Zeroizing;

/// Stores secret key material in memory.
///
/// Each stored value is automatically zeroized when it is removed,
/// replaced, or when the complete key store is dropped.
#[derive(Default)]
pub struct SecureKeyStore {
    keys: HashMap<String, Zeroizing<Vec<u8>>>,
}

impl SecureKeyStore {
    /// Creates an empty key store.
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Stores secret bytes under the provided name.
    pub fn store(&mut self, name: &str, key_bytes: &[u8]) {
        let protected_key = Zeroizing::new(key_bytes.to_vec());

        self.keys.insert(name.to_string(), protected_key);
    }

    /// Returns a borrowed view of a stored key.
    ///
    /// Returns None when the requested name does not exist.
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.keys.get(name).map(|key| key.as_slice())
    }
    /// Removes one key.
    ///
    /// The Zeroizing wrapper overwrites the key bytes before releasing them.
    /// Returns true if the key existed.
    pub fn wipe(&mut self, name: &str) -> bool {
        self.keys.remove(name).is_some()
    }

    /// Removes and zeroizes every stored key.
    pub fn wipe_all(&mut self) {
        self.keys.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::SecureKeyStore;

    #[test]
    fn stores_and_retrieves_key() {
        let mut key_store = SecureKeyStore::new();

        let secret = [10_u8, 20, 30, 40, 50, 60];

        key_store.store("session_key", &secret);

        assert_eq!(key_store.get("session_key"), Some(secret.as_slice()));

        assert_eq!(key_store.get("missing_key"), None);
    }

    #[test]
    fn wipes_one_key() {
        let mut key_store = SecureKeyStore::new();

        key_store.store("first_key", &[1, 2, 3]);
        key_store.store("second_key", &[4, 5, 6]);

        let existed = key_store.wipe("first_key");

        assert!(existed);
        assert_eq!(key_store.get("first_key"), None);
        assert_eq!(key_store.get("second_key"), Some([4_u8, 5, 6].as_slice()));

        // Wiping a key that no longer exists returns false.
        assert!(!key_store.wipe("first_key"));
    }

    #[test]
    fn wipes_all_keys() {
        let mut key_store = SecureKeyStore::new();

        key_store.store("first_key", &[1, 2, 3]);
        key_store.store("second_key", &[4, 5, 6]);

        key_store.wipe_all();

        assert_eq!(key_store.get("first_key"), None);
        assert_eq!(key_store.get("second_key"), None);
    }
}
