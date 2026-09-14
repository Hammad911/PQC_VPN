//! Tauri desktop shell — plan Week 4: window + system tray + a
//! connect/disconnect button wired to a stub.
//!
//! Deliberately not real yet. `core::state`'s `DeviceState` / `TunnelHandle`
//! traits (Week 3) get actual desktop implementations in Weeks 5 and 9, once
//! there is a real handshake (Week 6) and a trained policy wired in
//! (Week 7) to drive them. This shell exists to prove the UI can drive Rust
//! state through Tauri's command layer before any of that lands — exactly
//! what the plan's Week 4 task asks for.

use std::sync::Mutex;

use serde::Serialize;
use tauri::State;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;

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
            connection_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
