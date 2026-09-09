//! Platform boundary for everything `core` needs from the operating system.
//!
//! Nothing in `core/` is allowed to call a platform API directly (no
//! `sysinfo`, no `psutil` equivalent, no shelling out to `ping` or `wg`).
//! Every platform fact enters through one of the two traits defined here:
//!
//! - [`DeviceState`] — live device and network readings, the Rust
//!   counterpart of `client/rl_agent/state_observer.py`. The `desktop/`
//!   crate implements it with the `sysinfo` crate; a future mobile crate
//!   implements it against the OS APIs. `core` never reads a sensor itself.
//! - [`TunnelHandle`] — bringing the WireGuard interface up, rotating its
//!   pre-shared key on a rekey, and tearing it down. `desktop/` implements
//!   it by wrapping `wg` / `wg-quick`; mobile would use NetworkExtension /
//!   `VpnService`.
//!
//! [`StatePipeline`] sits on top of [`DeviceState`] and turns raw readings
//! into the frozen 7-dim `float32` state vector the RL policy consumes
//! (`contracts/state_vector.json`, `INTERFACE_FREEZE_PROPOSAL.md` §1). The
//! normalization caps live here, in shared code, on purpose: the policy was
//! trained against states scaled exactly this way, so the desktop and any
//! future mobile port must not each re-derive them.

use std::fmt;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use zeroize::Zeroizing;

/// Number of elements in the RL state vector. Frozen — see
/// `contracts/state_vector.json`.
pub const STATE_DIM: usize = 7;

/// Index of each state-vector element. The order is frozen; read these
/// names rather than hardcoding the integers.
pub const CPU_LOAD: usize = 0;
pub const RAM_AVAIL: usize = 1;
pub const LATENCY: usize = 2;
pub const UPLOAD: usize = 3;
pub const CONN_TYPE: usize = 4;
pub const TIME_SINCE_REKEY: usize = 5;
pub const THREAT: usize = 6;

/// Best-effort classification of the active network link.
///
/// The numeric scores match `state_observer.py`'s `CONNECTION_TYPE_SCORE`
/// and `contracts/state_vector.json` — Wi-Fi and "unknown" deliberately
/// share the middle value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    Wired,
    WifiOrUnknown,
    Cellular,
}

impl ConnectionType {
    /// Returns the value this link type contributes to the state vector.
    pub fn score(self) -> f32 {
        match self {
            ConnectionType::Wired => 0.0,
            ConnectionType::WifiOrUnknown => 0.5,
            ConnectionType::Cellular => 1.0,
        }
    }
}

/// Live device and network readings.
///
/// Implementations return raw, un-normalized facts. Turning these into the
/// `[0, 1]` state vector is [`StatePipeline`]'s job, not the platform's, so
/// the scaling stays identical across every port.
///
/// `TIME_SINCE_REKEY` and `THREAT` are intentionally absent: the rekey
/// timer is tracked inside [`StatePipeline`] (reset with
/// [`StatePipeline::mark_rekey`]), and the threat score is supplied by the
/// anomaly pipeline (`core::anomaly`), not by any device sensor.
pub trait DeviceState {
    /// Current CPU utilization, already a fraction in `[0, 1]`
    /// (`0.0` idle, `1.0` saturated).
    fn cpu_load(&self) -> f32;

    /// Fraction of system RAM currently available, in `[0, 1]`
    /// (`0.0` none free, `1.0` all free).
    fn ram_available_fraction(&self) -> f32;

    /// Round-trip time to the reachability probe target.
    ///
    /// `None` means the target was unreachable — [`StatePipeline`]
    /// saturates the latency dimension to `1.0` in that case, matching
    /// `state_observer.py`'s fallback.
    fn latency(&self) -> Option<Duration>;

    /// Recent upload throughput in bytes per second.
    fn upload_rate_bytes_per_sec(&self) -> f64;

    /// Best-effort classification of the active link.
    fn connection_type(&self) -> ConnectionType;
}

/// Normalization caps applied by [`StatePipeline`].
///
/// The defaults reproduce `state_observer.py` exactly. They are reasoned
/// first-pass choices, not calibrated against real traffic
/// (`contracts/state_vector.json` says the same); retuning them changes the
/// raw-metric mapping, not the interface.
#[derive(Debug, Clone, Copy)]
pub struct NormalizationCaps {
    /// Latency at or above this maps to `1.0`. Default 300 ms.
    pub latency_cap: Duration,
    /// Upload rate at or above this maps to `1.0`. Default 5 MB/s.
    pub upload_cap_bytes_per_sec: f64,
    /// Time since the last rekey at or above this maps to `1.0`.
    /// Default 1 hour.
    pub rekey_interval_cap: Duration,
}

impl Default for NormalizationCaps {
    fn default() -> Self {
        Self {
            latency_cap: Duration::from_millis(300),
            upload_cap_bytes_per_sec: 5_000_000.0,
            rekey_interval_cap: Duration::from_secs(3600),
        }
    }
}

/// Builds the frozen 7-dim state vector from a [`DeviceState`].
///
/// Owns the rekey timer and the normalization caps so those never leak
/// into platform code. One instance lives for the length of a session; the
/// control loop calls [`observe`](Self::observe) each tick and
/// [`mark_rekey`](Self::mark_rekey) whenever a key rotation completes.
pub struct StatePipeline<D> {
    device: D,
    caps: NormalizationCaps,
    time_since_rekey: Duration,
}

impl<D: DeviceState> StatePipeline<D> {
    /// Creates a pipeline with the default caps, as if a rekey just
    /// happened.
    pub fn new(device: D) -> Self {
        Self::with_caps(device, NormalizationCaps::default())
    }

    /// Creates a pipeline with explicit caps.
    pub fn with_caps(device: D, caps: NormalizationCaps) -> Self {
        Self {
            device,
            caps,
            time_since_rekey: Duration::ZERO,
        }
    }

    /// Borrows the underlying [`DeviceState`].
    pub fn device(&self) -> &D {
        &self.device
    }

    /// Resets the rekey timer. Call this when a key rotation completes.
    pub fn mark_rekey(&mut self) {
        self.time_since_rekey = Duration::ZERO;
    }

    /// Advances the rekey timer by the time elapsed since the last tick.
    ///
    /// The pipeline does not read a clock itself — the control loop owns
    /// the tick cadence and reports it here, which also keeps the timer
    /// testable without waiting on wall-clock time.
    pub fn advance(&mut self, elapsed: Duration) {
        self.time_since_rekey = self.time_since_rekey.saturating_add(elapsed);
    }

    /// Reads the device and produces the normalized state vector.
    ///
    /// `threat_score` comes from `core::anomaly` and is clamped into
    /// `[0, 1]` here, matching `state_observer.py::read_state`.
    pub fn observe(&self, threat_score: f32) -> [f32; STATE_DIM] {
        let mut vector = [0.0_f32; STATE_DIM];

        vector[CPU_LOAD] = self.device.cpu_load();

        vector[RAM_AVAIL] = self.device.ram_available_fraction();

        vector[LATENCY] = normalize_latency(self.device.latency(), self.caps.latency_cap);

        vector[UPLOAD] = normalize_rate(
            self.device.upload_rate_bytes_per_sec(),
            self.caps.upload_cap_bytes_per_sec,
        );

        vector[CONN_TYPE] = self.device.connection_type().score();

        vector[TIME_SINCE_REKEY] =
            normalize_duration(self.time_since_rekey, self.caps.rekey_interval_cap);

        vector[THREAT] = threat_score;

        for value in &mut vector {
            *value = value.clamp(0.0, 1.0);
        }

        vector
    }
}

/// Normalizes a latency reading against `cap`. A missing reading (an
/// unreachable target) saturates to `1.0`.
pub fn normalize_latency(latency: Option<Duration>, cap: Duration) -> f32 {
    match latency {
        Some(rtt) => normalize_duration(rtt, cap),
        None => 1.0,
    }
}

/// Normalizes a `Duration` against `cap`, clamped to `[0, 1]`.
pub fn normalize_duration(value: Duration, cap: Duration) -> f32 {
    if cap.is_zero() {
        return 1.0;
    }

    (value.as_secs_f64() / cap.as_secs_f64()).clamp(0.0, 1.0) as f32
}

/// Normalizes a rate against `cap`, clamped to `[0, 1]`.
pub fn normalize_rate(value: f64, cap: f64) -> f32 {
    if cap <= 0.0 {
        return 1.0;
    }

    (value.max(0.0) / cap).clamp(0.0, 1.0) as f32
}

/// The 32-byte WireGuard pre-shared key derived from a completed handshake.
pub type SessionPsk = Zeroizing<[u8; 32]>;

/// Everything the client needs to bring its WireGuard interface up after a
/// handshake, learned from `ServerFinish` (`server/PROTOCOL.md` §4.5) plus
/// the pre-shared key both sides derived (never sent on the wire).
pub struct TunnelConfig {
    /// IPv4 address the server assigned this peer inside the tunnel.
    pub assigned_ip: Ipv4Addr,
    /// The server's raw (not base64) WireGuard public key.
    pub server_wg_pubkey: [u8; 32],
    /// Where the server's WireGuard endpoint listens.
    pub server_endpoint: SocketAddr,
    /// The derived pre-shared key.
    pub psk: SessionPsk,
}

impl fmt::Debug for TunnelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print the PSK.
        f.debug_struct("TunnelConfig")
            .field("assigned_ip", &self.assigned_ip)
            .field("server_endpoint", &self.server_endpoint)
            .field("psk", &"<redacted>")
            .finish()
    }
}

/// A snapshot of the tunnel's state, for the dashboard (Week 10).
#[derive(Debug, Clone)]
pub struct TunnelStatus {
    /// Whether the interface is currently up.
    pub up: bool,
    /// The in-tunnel address, once assigned.
    pub assigned_ip: Option<Ipv4Addr>,
    /// Time since WireGuard last completed a handshake with the server.
    pub last_handshake_age: Option<Duration>,
}

/// Why a [`TunnelHandle`] operation failed.
#[derive(Debug)]
pub enum TunnelError {
    /// The process lacks the privilege to configure the interface
    /// (`CAP_NET_ADMIN`, admin rights).
    NotPermitted(String),
    /// The WireGuard interface does not exist or is not reachable.
    InterfaceUnavailable(String),
    /// The platform backend (`wg`, an OS API) reported a failure.
    Backend(String),
}

impl fmt::Display for TunnelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TunnelError::NotPermitted(detail) => {
                write!(f, "not permitted to configure the tunnel: {detail}")
            }
            TunnelError::InterfaceUnavailable(detail) => {
                write!(f, "wireguard interface unavailable: {detail}")
            }
            TunnelError::Backend(detail) => write!(f, "tunnel backend error: {detail}"),
        }
    }
}

impl std::error::Error for TunnelError {}

/// Controls the local WireGuard tunnel.
///
/// The PQC handshake (`core::protocol`, Week 6) produces a [`TunnelConfig`];
/// this trait is how `core` acts on it without knowing whether it is
/// driving `wg-quick` on a laptop or a `VpnService` on a phone.
pub trait TunnelHandle {
    /// Brings the interface up with the given configuration.
    fn bring_up(&self, config: &TunnelConfig) -> Result<(), TunnelError>;

    /// Swaps the peer's pre-shared key in place, without dropping the
    /// tunnel — the rekey path (`server/PROTOCOL.md` §6). WireGuard picks
    /// up the new key on its next handshake.
    fn rotate_psk(&self, psk: &SessionPsk) -> Result<(), TunnelError>;

    /// Tears the interface down.
    fn tear_down(&self) -> Result<(), TunnelError>;

    /// Reports the current tunnel state.
    fn status(&self) -> Result<TunnelStatus, TunnelError>;
}

#[cfg(any(test, feature = "mock"))]
pub mod mock {
    //! In-memory [`DeviceState`] and [`TunnelHandle`] doubles.
    //!
    //! Enabled by the `mock` feature (always on under `cfg(test)`), so the
    //! Weeks 5–9 wiring in `desktop/` — and Member 3's eventual
    //! integration — can build against `core` before the real platform
    //! implementations exist. The plan's risk table calls for exactly this:
    //! work against a stub rather than stalling on a dependency.

    use std::cell::RefCell;
    use std::time::Duration;

    use super::{
        ConnectionType, DeviceState, SessionPsk, TunnelConfig, TunnelError, TunnelHandle,
        TunnelStatus,
    };

    /// A [`DeviceState`] with every reading set explicitly.
    #[derive(Debug, Clone)]
    pub struct MockDeviceState {
        pub cpu_load: f32,
        pub ram_available_fraction: f32,
        pub latency: Option<Duration>,
        pub upload_rate_bytes_per_sec: f64,
        pub connection_type: ConnectionType,
    }

    impl Default for MockDeviceState {
        fn default() -> Self {
            Self {
                cpu_load: 0.0,
                ram_available_fraction: 1.0,
                latency: Some(Duration::from_millis(0)),
                upload_rate_bytes_per_sec: 0.0,
                connection_type: ConnectionType::Wired,
            }
        }
    }

    impl DeviceState for MockDeviceState {
        fn cpu_load(&self) -> f32 {
            self.cpu_load
        }

        fn ram_available_fraction(&self) -> f32 {
            self.ram_available_fraction
        }

        fn latency(&self) -> Option<Duration> {
            self.latency
        }

        fn upload_rate_bytes_per_sec(&self) -> f64 {
            self.upload_rate_bytes_per_sec
        }

        fn connection_type(&self) -> ConnectionType {
            self.connection_type
        }
    }

    /// A [`TunnelHandle`] that records calls instead of touching an
    /// interface.
    #[derive(Debug, Default)]
    pub struct MockTunnelHandle {
        up: RefCell<bool>,
        rotations: RefCell<u32>,
    }

    impl MockTunnelHandle {
        /// Whether [`bring_up`](TunnelHandle::bring_up) has been called
        /// more recently than [`tear_down`](TunnelHandle::tear_down).
        pub fn is_up(&self) -> bool {
            *self.up.borrow()
        }

        /// How many times [`rotate_psk`](TunnelHandle::rotate_psk) has been
        /// called.
        pub fn rotations(&self) -> u32 {
            *self.rotations.borrow()
        }
    }

    impl TunnelHandle for MockTunnelHandle {
        fn bring_up(&self, _config: &TunnelConfig) -> Result<(), TunnelError> {
            *self.up.borrow_mut() = true;

            Ok(())
        }

        fn rotate_psk(&self, _psk: &SessionPsk) -> Result<(), TunnelError> {
            if !*self.up.borrow() {
                return Err(TunnelError::InterfaceUnavailable(
                    "rotate_psk before bring_up".to_string(),
                ));
            }

            *self.rotations.borrow_mut() += 1;

            Ok(())
        }

        fn tear_down(&self) -> Result<(), TunnelError> {
            *self.up.borrow_mut() = false;

            Ok(())
        }

        fn status(&self) -> Result<TunnelStatus, TunnelError> {
            Ok(TunnelStatus {
                up: *self.up.borrow(),
                assigned_ip: None,
                last_handshake_age: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddr};
    use std::time::Duration;

    use zeroize::Zeroizing;

    use super::mock::{MockDeviceState, MockTunnelHandle};
    use super::{
        CONN_TYPE, CPU_LOAD, ConnectionType, LATENCY, RAM_AVAIL, STATE_DIM, StatePipeline, THREAT,
        TIME_SINCE_REKEY, TunnelConfig, TunnelHandle, UPLOAD, normalize_latency, normalize_rate,
    };

    #[test]
    fn connection_type_scores_match_the_python_observer() {
        assert_eq!(ConnectionType::Wired.score(), 0.0);
        assert_eq!(ConnectionType::WifiOrUnknown.score(), 0.5);
        assert_eq!(ConnectionType::Cellular.score(), 1.0);
    }

    #[test]
    fn unreachable_target_saturates_latency() {
        assert_eq!(normalize_latency(None, Duration::from_millis(300)), 1.0);
    }

    #[test]
    fn latency_and_upload_normalize_against_the_default_caps() {
        // Half of the 300 ms cap.
        assert_eq!(
            normalize_latency(Some(Duration::from_millis(150)), Duration::from_millis(300)),
            0.5
        );

        // Over the cap clamps to 1.0.
        assert_eq!(
            normalize_latency(Some(Duration::from_secs(2)), Duration::from_millis(300)),
            1.0
        );

        // A fifth of the 5 MB/s cap.
        assert_eq!(normalize_rate(1_000_000.0, 5_000_000.0), 0.2);

        // Negative rates are treated as zero.
        assert_eq!(normalize_rate(-1.0, 5_000_000.0), 0.0);
    }

    #[test]
    fn observe_places_every_reading_at_its_frozen_index() {
        let device = MockDeviceState {
            cpu_load: 0.4,
            ram_available_fraction: 0.7,
            latency: Some(Duration::from_millis(150)),
            upload_rate_bytes_per_sec: 2_500_000.0,
            connection_type: ConnectionType::Cellular,
        };

        let pipeline = StatePipeline::new(device);

        let state = pipeline.observe(0.25);

        assert_eq!(state.len(), STATE_DIM);
        assert_eq!(state[CPU_LOAD], 0.4);
        assert_eq!(state[RAM_AVAIL], 0.7);
        assert_eq!(state[LATENCY], 0.5);
        assert_eq!(state[UPLOAD], 0.5);
        assert_eq!(state[CONN_TYPE], 1.0);
        assert_eq!(state[TIME_SINCE_REKEY], 0.0);
        assert_eq!(state[THREAT], 0.25);
    }

    #[test]
    fn observe_clamps_out_of_range_readings() {
        let device = MockDeviceState {
            cpu_load: 1.5,
            ram_available_fraction: -0.2,
            ..MockDeviceState::default()
        };

        let state = StatePipeline::new(device).observe(9.0);

        assert_eq!(state[CPU_LOAD], 1.0);
        assert_eq!(state[RAM_AVAIL], 0.0);
        assert_eq!(state[THREAT], 1.0);
    }

    #[test]
    fn rekey_timer_advances_and_resets() {
        let mut pipeline = StatePipeline::new(MockDeviceState::default());

        // 30 minutes into the default 1-hour cap.
        pipeline.advance(Duration::from_secs(1800));
        assert_eq!(pipeline.observe(0.0)[TIME_SINCE_REKEY], 0.5);

        // Past the cap clamps to 1.0.
        pipeline.advance(Duration::from_secs(3600));
        assert_eq!(pipeline.observe(0.0)[TIME_SINCE_REKEY], 1.0);

        // A rekey resets it.
        pipeline.mark_rekey();
        assert_eq!(pipeline.observe(0.0)[TIME_SINCE_REKEY], 0.0);
    }

    #[test]
    fn mock_tunnel_tracks_up_state_and_rotations() {
        let tunnel = MockTunnelHandle::default();

        let config = TunnelConfig {
            assigned_ip: Ipv4Addr::new(10, 8, 0, 2),
            server_wg_pubkey: [0_u8; 32],
            server_endpoint: SocketAddr::from(([139, 59, 62, 4], 51820)),
            psk: Zeroizing::new([7_u8; 32]),
        };

        // A rotation before bring_up is rejected.
        assert!(tunnel.rotate_psk(&Zeroizing::new([1_u8; 32])).is_err());

        tunnel.bring_up(&config).expect("bring_up failed");
        assert!(tunnel.is_up());

        tunnel
            .rotate_psk(&Zeroizing::new([2_u8; 32]))
            .expect("rotate_psk failed");
        assert_eq!(tunnel.rotations(), 1);

        tunnel.tear_down().expect("tear_down failed");
        assert!(!tunnel.is_up());
        assert!(!tunnel.status().expect("status failed").up);
    }
}
