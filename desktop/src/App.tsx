import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

// Mirrors the Rust `ConnectionStatus` enum (src-tauri/src/lib.rs). Kept as a
// plain union rather than a shared type generator — there's exactly one
// enum to keep in sync right now, and adding a codegen step for that is not
// worth it yet.
type ConnectionStatus = "disconnected" | "connecting" | "connected";

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  disconnected: "Disconnected",
  connecting: "Connecting…",
  connected: "Connected",
};

// Mirrors the Rust `DeviceSnapshot` struct (src-tauri/src/lib.rs) field for
// field — plain serde output, no rename_all, so the JSON keys are exactly
// the Rust field names.
type DeviceSnapshot = {
  cpu_load: number;
  ram_available_fraction: number;
  latency_ms: number | null;
  upload_rate_bytes_per_sec: number;
  connection_type: string;
};

const percent = (fraction: number) => `${Math.round(fraction * 100)}%`;

function App() {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [pending, setPending] = useState(false);
  const [device, setDevice] = useState<DeviceSnapshot | null>(null);
  const [deviceLoading, setDeviceLoading] = useState(false);

  // Read the real status on load rather than assuming "disconnected" — the
  // backend, not the frontend, owns this state.
  useEffect(() => {
    invoke<ConnectionStatus>("connection_status").then(setStatus);
  }, []);

  async function toggleConnection() {
    setPending(true);
    try {
      const next = await invoke<ConnectionStatus>(
        status === "connected" ? "disconnect" : "connect",
      );
      setStatus(next);
    } finally {
      setPending(false);
    }
  }

  // On-demand only, not polled — a real read takes ~200ms-1.2s (the CPU
  // sample window plus the ping timeout), so this stays a button rather
  // than a background timer until Week 8 wires the actual 5-second tick
  // loop the plan's architecture calls for.
  async function refreshDevice() {
    setDeviceLoading(true);
    try {
      setDevice(await invoke<DeviceSnapshot>("device_snapshot"));
    } finally {
      setDeviceLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>PQC VPN</h1>

      {/* Stub only — Week 9 wires this to a real handshake + WireGuard
          tunnel via core::state::TunnelHandle. Nothing here touches the
          network yet. */}
      <div className={`status status--${status}`}>
        <span className="status__dot" />
        {STATUS_LABEL[status]}
      </div>

      <button
        className="connect-button"
        onClick={toggleConnection}
        disabled={pending}
      >
        {status === "connected" ? "Disconnect" : "Connect"}
      </button>

      {/* Week 5: real device readings via vpn_core::state::DeviceState
          (desktop's sysinfo-backed implementation). Not yet fed into the
          RL agent's state vector — that needs the rekey timer and threat
          score too (Week 8). */}
      <section className="device-panel">
        <div className="device-panel__header">
          <h2>Device</h2>
          <button onClick={refreshDevice} disabled={deviceLoading}>
            {deviceLoading ? "Reading…" : "Refresh"}
          </button>
        </div>

        {device ? (
          <dl className="device-grid">
            <dt>CPU load</dt>
            <dd>{percent(device.cpu_load)}</dd>

            <dt>RAM available</dt>
            <dd>{percent(device.ram_available_fraction)}</dd>

            <dt>Latency</dt>
            <dd>
              {device.latency_ms === null
                ? "unreachable"
                : `${device.latency_ms.toFixed(0)} ms`}
            </dd>

            <dt>Upload rate</dt>
            <dd>{(device.upload_rate_bytes_per_sec / 1000).toFixed(1)} KB/s</dd>

            <dt>Connection</dt>
            <dd>{device.connection_type}</dd>
          </dl>
        ) : (
          <p className="device-panel__empty">
            Not read yet — click Refresh for live numbers.
          </p>
        )}
      </section>
    </main>
  );
}

export default App;
