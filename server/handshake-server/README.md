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
| `session` | in-memory session table |
| `server` | TCP accept loop + per-connection handler |

## Binaries

```bash
# the server
cargo run -p handshake-server --bin handshake-server -- \
    --bind 0.0.0.0:51821 --identity ./server-identity.seed

# throwaway test client — runs a full handshake, prints the derived PSK
cargo run -p handshake-server --bin test-client -- \
    <host:port> <verifying-key-hex | @path> --algo 512|768|1024

# regenerate server/handshake-vectors.json
cargo run -p handshake-server --bin emit-vectors -- server/handshake-vectors.json
```

The server prints its verifying key on startup and also writes it next to the
identity file as `<identity>.pub` (hex). That hex is what a client pins.

## Tests

```bash
cargo test -p handshake-server        # 13 unit + 4 integration
```

`tests/handshake.rs` includes `over_tcp_full_handshake` — a real `TcpListener`
handshake for ML-KEM-512/768/1024 with both sides asserting the same PSK.

## Week 2 scope / not yet done

- **No `wg set`.** The server derives and records the PSK; where it would run
  `wg set … preshared-key` it logs `would install/swap PSK …`. PSK injection is
  Week 3.
- **No `RekeyRequest` CLI.** The server-side rekey path (`PROTOCOL.md` §6,
  including the `REKEY_ESCALATES` no-downgrade rule) is implemented and
  unit-tested; a `--rekey <session-id>` flag on `test-client` lands Week 6.
- **Replay cache / rate limiting** — fields are in the protocol; enforcement is
  Week 7.
- Blocking thread-per-connection is fine at this stage; revisit for the Week 10
  soak test.

## Deploy

`../deploy/deploy-handshake-server.sh` builds a release binary and installs it
on the droplet as the `handshake-server` systemd service
(`../deploy/handshake-server.service`). The identity seed lives at
`/var/lib/pqc-vpn/server-identity.seed` and survives redeploys.
