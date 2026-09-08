//! In-memory session table.
//!
//! Minimal for Week 2 — enough to complete one handshake, record the result,
//! and look it up for a later `RekeyRequest`. Multi-peer lifecycle management is
//! Member 2 Week 5.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::SystemTime;

use crate::wire::AlgoCode;

#[derive(Clone)]
pub struct Session {
    pub session_id: [u8; 16],
    pub peer_wg_pubkey: [u8; 32],
    pub in_force_algo: AlgoCode,
    /// Current preshared key (would be installed on the WireGuard peer).
    pub psk: [u8; 32],
    pub created: SystemTime,
    pub rekeys: u32,
}

#[derive(Default)]
pub struct SessionTable {
    inner: Mutex<HashMap<[u8; 16], Session>>,
}

impl SessionTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, session: Session) {
        self.inner
            .lock()
            .unwrap()
            .insert(session.session_id, session);
    }

    pub fn in_force_algo(&self, id: &[u8; 16]) -> Option<AlgoCode> {
        self.inner.lock().unwrap().get(id).map(|s| s.in_force_algo)
    }

    pub fn snapshot(&self, id: &[u8; 16]) -> Option<Session> {
        self.inner.lock().unwrap().get(id).cloned()
    }

    /// Apply a completed rekey: swap the PSK, update the in-force algorithm
    /// (which may have escalated — `PROTOCOL.md` §6), bump the counter.
    pub fn apply_rekey(&self, id: &[u8; 16], new_algo: AlgoCode, new_psk: [u8; 32]) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.get_mut(id) {
            Some(s) => {
                s.in_force_algo = new_algo;
                s.psk = new_psk;
                s.rekeys += 1;
                true
            }
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}
