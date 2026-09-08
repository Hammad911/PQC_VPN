# handshake-server

Server-side PQC handshake responder — reference implementation of
[`../PROTOCOL.md`](../PROTOCOL.md) v1. Member 2, Week 2.

All cryptography is `vpn_core::crypto` (Member 1's Rust port). This crate is
only the wire protocol around it: framing, the signed transcript, the key
schedule, the server identity, and a blocking TCP server.

## Layout

| module | role |
|---|---|
| `wire` | frame + message encode/decode (`PROTOCOL.md` §3–4) |
| `transcript` | the byte string the server signs (§5.1) |
| `kdf` | `hybrid_secret` → `{psk, confirm_key}` + the finish MACs (§5.3) |
| `identity` | long-term ML-DSA-65 key, persisted as its 32-byte seed (§5.2) |
| `handshake` | server: `ClientHello` → `ServerHello` |
| `client_handshake` | client side, for `test-client` and `emit-vectors` |
| `tunnel` | `PskInstaller` — `WgCli` runs `wg set` / `wg show … peers` / `… remove`; `DryRun` logs |
| `registry` | multi-peer registry: session ↔ wg peer ↔ tunnel address, persisted to `peers.json` (no PSKs); capacity limit; startup reconciliation against `wg0` |
| `server` | TCP accept loop + per-connection handler; updates the registry and installs the PSK; optional idle sweep |

## Binaries

```bash
# the server — dry-run (no WireGuard touched)
cargo run -p handshake-server --bin handshake-server -- \
    --bind 0.0.0.0:51821 --identity ./server-identity.seed

# the server — real: install PSKs into wg0 (needs CAP_NET_ADMIN)
cargo run -p handshake-server --bin handshake-server -- \
    --wg-interface wg0 --wg-port 51820 --tunnel-cidr 10.8.0.0/24 \
    --peers-file /var/lib/pqc-vpn/peers.json --max-peers 64 --idle-timeout-secs 0

# throwaway test client — full handshake, then prints a wg-quick client config
cargo run -p handshake-server --bin test-client -- \
    <handshake-host:port> <verifying-key-hex | @path> --algo 512|768|1024

# regenerate server/handshake-vectors.json
cargo run -p handshake-server --bin emit-vectors -- server/handshake-vectors.json
```

The server prints its verifying key on startup and also writes it next to the
identity file as `<identity>.pub` (hex). That hex is what a client pins.

## Tests

```bash
cargo test -p handshake-server        # 17 unit + 4 integration
```

`tests/handshake.rs` includes `over_tcp_full_handshake_returns_tunnel_params` —
a real `TcpListener` handshake for ML-KEM-512/768/1024 with both sides asserting
the same PSK, plus a dry-run PSK install and the returned tunnel params.

## Not yet done (later weeks)

- **Peers live in `wg0` + `peers.json`, not written into `wg0.conf`.** After a
  `wg0` restart the peers are gone from the interface but kept in the registry;
  their clients re-handshake. (Writing `wg0.conf` for true persistence is
  possible later; the registry + reconciliation covers the container case.)
- **No `RekeyRequest` CLI.** The server-side rekey path (`PROTOCOL.md` §6,
  including the `REKEY_ESCALATES` no-downgrade rule and the
  `(wg_pubkey, session_id)` lookup) is implemented and unit-tested; a
  `--rekey` flag on `test-client` lands Week 6.
- **Replay cache / rate limiting** — fields are in the protocol; enforcement is
  Week 7.
- **Idle eviction** is implemented (`--idle-timeout-secs`) but off by default;
  real use is Week 9/10.
- Blocking thread-per-connection is fine at this stage; revisit for the Week 10
  soak test.

## Deploy

**Primary (Week 4+):** `../deploy/deploy-docker.sh` — `docker compose up -d --build`
on the droplet. `../Dockerfile` (multi-stage) + `../docker-compose.yml`
(`network_mode: host`, `cap_drop: ALL` + `cap_add: NET_ADMIN`, `read_only`).

**Fallback:** `../deploy/deploy-handshake-server.sh` — release binary +
`handshake-server` systemd unit.

Either way the ML-DSA-65 identity seed lives at
`/var/lib/pqc-vpn/server-identity.seed` (a bind-mounted volume under Docker) and
survives redeploys, so the pinned client key never changes.
