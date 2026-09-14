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

function App() {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [pending, setPending] = useState(false);

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
    </main>
  );
}

export default App;
