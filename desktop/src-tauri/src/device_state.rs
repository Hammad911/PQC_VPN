//! Desktop's real `vpn_core::state::DeviceState` — plan Week 5, the Rust
//! equivalent of `client/rl_agent/state_observer.py`.
//!
//! Reads live CPU, RAM, and network numbers via the `sysinfo` crate, and
//! latency via a single ICMP `ping` (the same technique
//! `state_observer.py::_latency` uses — sysinfo doesn't do reachability
//! probes, and a raw-socket ping needs elevated privileges on both Windows
//! and Linux, which a plain `ping` invocation does not).
//!
//! This is the only place in the `desktop` crate allowed to touch `sysinfo`
//! or shell out to `ping` — `core` never does, by the trait boundary
//! `core/README.md` documents.

use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sysinfo::{Networks, System};
use vpn_core::state::{ConnectionType, DeviceState};

/// Host pinged for the latency reading. Same default as
/// `state_observer.py::StateObserver.__init__`.
const DEFAULT_PING_TARGET: &str = "1.1.1.1";

/// `sysinfo::System`/`Networks` need `&mut self` to refresh, but
/// [`DeviceState`] hands out `&self` — real device access can be called
/// from any Tauri command thread. The `Mutex`es make that legal without
/// pushing interior mutability into `core`.
pub struct SysinfoDeviceState {
    system: Mutex<System>,
    networks: Mutex<Networks>,
    last_network_sample: Mutex<Instant>,
    ping_target: String,
}

impl SysinfoDeviceState {
    /// Creates a device state reader that pings [`DEFAULT_PING_TARGET`].
    pub fn new() -> Self {
        Self::with_ping_target(DEFAULT_PING_TARGET.to_string())
    }

    /// Creates a device state reader pinging a specific host — used by
    /// tests so they don't depend on real internet access.
    pub fn with_ping_target(ping_target: String) -> Self {
        Self {
            system: Mutex::new(System::new()),
            networks: Mutex::new(Networks::new_with_refreshed_list()),
            last_network_sample: Mutex::new(Instant::now()),
            ping_target,
        }
    }
}

impl Default for SysinfoDeviceState {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceState for SysinfoDeviceState {
    fn cpu_load(&self) -> f32 {
        let mut system = self.system.lock().expect("sysinfo System mutex poisoned");

        // sysinfo needs two refreshes at least MINIMUM_CPU_UPDATE_INTERVAL
        // apart to report a real (non-zero) usage figure - mirrors
        // `psutil.cpu_percent(interval=0.1)`'s blocking sample window.
        system.refresh_cpu_usage();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_cpu_usage();

        (system.global_cpu_usage() / 100.0).clamp(0.0, 1.0)
    }

    fn ram_available_fraction(&self) -> f32 {
        let mut system = self.system.lock().expect("sysinfo System mutex poisoned");

        system.refresh_memory();

        let total = system.total_memory();

        if total == 0 {
            // No reading possible - degrade to the conservative "no RAM
            // free" reading rather than dividing by zero, matching
            // state_observer.py's "never raise, degrade instead" rule.
            return 0.0;
        }

        (system.available_memory() as f32 / total as f32).clamp(0.0, 1.0)
    }

    fn latency(&self) -> Option<Duration> {
        ping_once(&self.ping_target)
    }

    fn upload_rate_bytes_per_sec(&self) -> f64 {
        let mut networks = self
            .networks
            .lock()
            .expect("sysinfo Networks mutex poisoned");
        let mut last_sample = self
            .last_network_sample
            .lock()
            .expect("last-sample mutex poisoned");

        networks.refresh(true);

        // `transmitted()` is bytes sent since the *previous* refresh, so the
        // elapsed time to divide by is the time since we last called this
        // function - not something sysinfo tracks for us.
        let elapsed = last_sample.elapsed().as_secs_f64().max(1e-6);
        *last_sample = Instant::now();

        let bytes_sent: u64 = networks.values().map(|data| data.transmitted()).sum();

        bytes_sent as f64 / elapsed
    }

    fn connection_type(&self) -> ConnectionType {
        let networks = self
            .networks
            .lock()
            .expect("sysinfo Networks mutex poisoned");

        let names: Vec<&str> = networks.keys().map(String::as_str).collect();

        classify_connection_type(&names)
    }
}

/// Runs a single ICMP ping against `target`, `None` if unreachable or the
/// `ping` binary itself is unavailable — mirrors
/// `state_observer.py::_latency`'s fallback exactly. Kept OS-specific only
/// in its argument list; Windows support is explicitly out of scope beyond
/// this (`RL-PQC-VPN_3-Month-Delivery-Plan.pdf` §8), but this crate builds
/// and is developed on Windows, so both branches exist.
fn ping_once(target: &str) -> Option<Duration> {
    let start = Instant::now();

    let output = if cfg!(target_os = "windows") {
        Command::new("ping")
            .args(["-n", "1", "-w", "1000", target])
            .output()
    } else {
        Command::new("ping")
            .args(["-c", "1", "-W", "1", target])
            .output()
    };

    match output {
        Ok(result) if result.status.success() => Some(start.elapsed()),
        _ => None,
    }
}

/// Best-effort classification from active interface names — same
/// substring heuristics as `state_observer.py::_connection_type`, adapted
/// to the collapsed `ConnectionType` (core treats Wi-Fi and "unknown" as
/// the same score, so both map to [`ConnectionType::WifiOrUnknown`]).
///
/// A pure function of the interface name list so it is testable without
/// touching real hardware. `sysinfo`'s `Networks` does not expose a
/// reliable "is this interface up" flag the way `psutil.net_if_stats`
/// does, so unlike the Python original this checks every interface
/// `sysinfo` reports rather than only the active ones — a known,
/// documented simplification, not a silent behavior change.
fn classify_connection_type(interface_names: &[&str]) -> ConnectionType {
    let lowered: Vec<String> = interface_names.iter().map(|n| n.to_lowercase()).collect();

    if lowered
        .iter()
        .any(|n| n.contains("wl") || n.contains("wifi") || n.contains("wlan"))
    {
        return ConnectionType::WifiOrUnknown;
    }

    if lowered
        .iter()
        .any(|n| (n.starts_with("en") || n.starts_with("eth")) && !n.contains("wl"))
    {
        return ConnectionType::Wired;
    }

    if lowered.iter().any(|n| n.contains("cellular") || n.contains("wwan")) {
        return ConnectionType::Cellular;
    }

    ConnectionType::WifiOrUnknown
}

#[cfg(test)]
mod tests {
    use super::{classify_connection_type, ping_once};

    #[test]
    fn wifi_interface_name_is_detected() {
        assert_eq!(
            classify_connection_type(&["wlan0"]),
            super::ConnectionType::WifiOrUnknown
        );
        assert_eq!(
            classify_connection_type(&["Wi-Fi"]),
            super::ConnectionType::WifiOrUnknown
        );
    }

    #[test]
    fn wired_interface_name_is_detected() {
        assert_eq!(
            classify_connection_type(&["eth0"]),
            super::ConnectionType::Wired
        );
        assert_eq!(
            classify_connection_type(&["Ethernet"]),
            super::ConnectionType::Wired
        );
    }

    #[test]
    fn cellular_interface_name_is_detected() {
        assert_eq!(
            classify_connection_type(&["wwan0"]),
            super::ConnectionType::Cellular
        );
    }

    #[test]
    fn wifi_takes_priority_over_wired_when_both_present() {
        // Matches state_observer.py's check order: Wi-Fi is checked first.
        assert_eq!(
            classify_connection_type(&["eth0", "wlan0"]),
            super::ConnectionType::WifiOrUnknown
        );
    }

    #[test]
    fn unrecognized_or_empty_interface_list_defaults_to_wifi_or_unknown() {
        assert_eq!(
            classify_connection_type(&["lo", "docker0"]),
            super::ConnectionType::WifiOrUnknown
        );
        assert_eq!(
            classify_connection_type(&[]),
            super::ConnectionType::WifiOrUnknown
        );
    }

    #[test]
    fn unreachable_host_returns_none_not_an_error() {
        // A reserved, non-routable TEST-NET-1 address (RFC 5737) - always
        // unreachable, no real network dependency, no flakiness from an
        // actual host being temporarily down.
        assert_eq!(ping_once("192.0.2.1"), None);
    }
}
