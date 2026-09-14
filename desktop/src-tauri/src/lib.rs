//! Tauri desktop shell.
//!
//! Week 4: window + system tray + a connect/disconnect button wired to a
//! stub. Week 5: `device_state` implements `core::state::DeviceState` for
//! real, using `sysinfo` — this module wires it into a Tauri command so the
//! frontend can show live numbers.
//!
//! What's still a stub: `core::state`'s `TunnelHandle` trait gets a real
//! desktop implementation in Week 9, once there is a real handshake
//! (Week 6) and a trained policy wired in (Week 7) to drive it.
//! `connect`/`disconnect` below still just flip an in-memory flag.

mod device_state;

use std::sync::Mutex;

use serde::Serialize;
use tauri::State;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use vpn_core::state::DeviceState;

use device_state::SysinfoDeviceState;

/// Connection state the frontend can read and drive.
///
/// `Connecting` exists now so the UI has somewhere to sit while a real
/// handshake (Week 6) is in flight later — the stub commands below jump
/// straight to `Connected`/`Disconnected`, but the frontend already renders
/// all three states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ConnectionStatus {
    Disconnected,
    // Not constructed yet - no command reports it until Week 6/9 wire a real,
    // non-instant handshake. Kept in the enum (not just the TS union) so the
    // frontend and backend never silently drift on what states exist.
    #[allow(dead_code)]
    Connecting,
    Connected,
}

/// App-wide state, managed by Tauri and shared across command invocations.
struct AppState {
    status: Mutex<ConnectionStatus>,
    device: SysinfoDeviceState,
}

/// A `DeviceState` reading, shaped for the frontend. Plain fields rather
/// than reusing `vpn_core::state::ConnectionType` directly — `core` does not
/// (and should not) depend on `serde`, so the enum has no `Serialize` impl;
/// its `Debug` output ("Wired" / "WifiOrUnknown" / "Cellular") is sent
/// as-is instead.
#[derive(Debug, Clone, Serialize)]
struct DeviceSnapshot {
    cpu_load: f32,
    ram_available_fraction: f32,
    /// `None` when the latency probe's target was unreachable.
    latency_ms: Option<f64>,
    upload_rate_bytes_per_sec: f64,
    connection_type: String,
}

/// Reads the five `DeviceState` values live from this machine. No caching,
/// no averaging — one on-demand snapshot per call, which is why the
/// frontend puts this behind a "Refresh" button rather than polling it.
///
/// This does **not** build the RL state vector — that's
/// `vpn_core::state::StatePipeline::observe`, which additionally needs a
/// rekey timer and a threat score from `core::anomaly` (Week 8). This
/// command exists to make Week 5's real device readings visible and
/// checkable on their own, before those other pieces exist.
#[tauri::command]
fn device_snapshot(state: State<AppState>) -> DeviceSnapshot {
    let device = &state.device;

    // Order matters only for wall-clock cost, not correctness: cpu_load
    // blocks ~200ms (sysinfo's minimum sample interval) and latency blocks
    // up to ~1s (the ping timeout) - both are read once each, not looped.
    let cpu_load = device.cpu_load();
    let ram_available_fraction = device.ram_available_fraction();
    let latency_ms = device.latency().map(|d| d.as_secs_f64() * 1000.0);
    let upload_rate_bytes_per_sec = device.upload_rate_bytes_per_sec();
    let connection_type = format!("{:?}", device.connection_type());

    DeviceSnapshot {
        cpu_load,
        ram_available_fraction,
        latency_ms,
        upload_rate_bytes_per_sec,
        connection_type,
    }
}

fn lock_status<'a>(state: &'a State<'a, AppState>) -> std::sync::MutexGuard<'a, ConnectionStatus> {
    state
        .status
        .lock()
        .expect("connection state mutex poisoned")
}

/// Stub: reports "connected" without touching a device, a socket, or a key.
/// Week 9 replaces this body with a real handshake result installed through
/// `core::state::TunnelHandle::bring_up`.
#[tauri::command]
fn connect(state: State<AppState>) -> ConnectionStatus {
    let mut status = lock_status(&state);

    *status = ConnectionStatus::Connected;

    *status
}

/// Stub: reports "disconnected". Week 9 replaces this with
/// `TunnelHandle::tear_down`.
#[tauri::command]
fn disconnect(state: State<AppState>) -> ConnectionStatus {
    let mut status = lock_status(&state);

    *status = ConnectionStatus::Disconnected;

    *status
}

/// Lets the frontend read the current status without changing it — used on
/// startup and whenever the window regains focus.
#[tauri::command]
fn connection_status(state: State<AppState>) -> ConnectionStatus {
    *lock_status(&state)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            status: Mutex::new(ConnectionStatus::Disconnected),
            device: SysinfoDeviceState::new(),
        })
        .setup(|app| {
            // System tray: an icon plus a minimal Quit menu. Real status
            // (connected / algorithm in use) starts showing here once
            // Week 10's dashboard work lands.
            let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

            let tray_menu = Menu::with_items(app, &[&quit_item])?;

            TrayIconBuilder::new()
                .icon(
                    app.default_window_icon()
                        .expect("the app icon is bundled at build time")
                        .clone(),
                )
                .menu(&tray_menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| {
                    if event.id.as_ref() == "quit" {
                        app.exit(0);
                    }
                })
                .build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            connect,
            disconnect,
            connection_status,
            device_snapshot
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
