# PQC-VPN Handshake Wire Protocol — v1 (FROZEN)

**Status:** FROZEN v1 — drafted by Member 2 for the end-of-Week-1 sync
(`TEAM_TIMELINE_PROPOSAL.md` §2 item 1), signed off by the team at the Week 4
checkpoint. All §8 questions are resolved. Any incompatible change from here
bumps `VER` (§9).
**Owner:** Member 2 (server). **Reviewers:** Member 1 (Rust client in `core/protocol/`), Member 3.

This document defines the bytes exchanged between a VPN **client** and the
**PQC handshake server** to establish (or rotate) the quantum-safe secret that
is then installed as a WireGuard Pre-Shared Key. It is deliberately independent
of client platform — the server makes no assumption about whether the peer is
desktop, mobile, or a test script. That property is what makes "mobile-ready
server" checkable rather than aspirational.

Source of truth for the crypto this protocol carries:
`core/src/crypto/hybrid_kem.rs`, `core/src/crypto/auth.rs`,
`contracts/algo_registry.json`.

## Change log (for the Week 4 protocol-stability sync with Member 1)

Everything below has been stable and implemented in `server/handshake-server/`
since Week 2. `VER` is still `0x01`; no incompatible change has been made.

| When | Change | Compatibility |
|---|---|---|
| Week 1 | Initial v1 draft: framing, 6 messages, transcript, key schedule, rekey rule | — |
| Week 2 | Transcript definition pinned to include `client_nonce ‖ server_nonce` (§5.1) — this is the still-open ask to fold back into `core::build_handshake_transcript` (§8 Q1) | additive to `core`'s current function; wire format unchanged |
| Week 3 | `ServerFinish` gained `server_wg_pubkey (32)`, `assigned_ip (4)`, `wg_port (2)` (§4.5) so the client needs no out-of-band tunnel config | **append-only** to one message; a `ServerFinish` reader written to the Week 2 layout would need updating, but no client existed yet |
| Week 4 | No protocol change (containerisation only). This section added. | — |
| Week 5 | New `Error` code `0x09 CAPACITY` (§4.7). Rekey lookup clarified: by `(client_wg_pubkey, session_id)` pair, not `session_id` alone (§4.6). `allowed-ips` always sent on `wg set` (§4.5). | additive error code; no wire-format change |
| Week 4 checkpoint sign-off | Status DRAFT → **FROZEN v1**. §8 resolved: nonces and HKDF now live in `vpn_core` and the server uses them (Q1/Q2), identity seed persistence in `vpn_core` (Q7), `REKEY_ESCALATES` approved (Q6), so §4.6 / §6 accept an equal-or-stronger rekey `algo`. | no wire-format change; every frame, transcript, KDF and MAC value in `handshake-vectors.json` unchanged (only its `protocol` status string updated) |

**For Member 1:** build the client side of `core/protocol/` against this document
and verify it against `server/handshake-vectors.json`. The transcript
(`vpn_core::crypto::hybrid_kem::build_handshake_transcript{,_from_bytes}`) and the
key schedule (`vpn_core::crypto::derive_session_keys`) are already shared with
the server, so only framing and the finish MACs need writing client-side.

---

## 1. Relationship to WireGuard

WireGuard is unchanged and does its own Noise handshake on `51820/udp`. This
protocol runs **beside** it as a separate service and only produces the 32-byte
value that both ends load with:

```
wg set <iface> peer <client-wg-pubkey> preshared-key <file>
```

```
          this protocol (51821/tcp)                     WireGuard (51820/udp)
client ───────── ClientHello ─────────▶ server
client ◀──────── ServerHello ───────── server
client ───────── ClientFinish ───────▶ server
                                        both install PSK ──▶ WG tunnel comes up
```

The PSK mixes into WireGuard's handshake, so a quantum adversary who breaks
X25519 (WireGuard's key exchange) still cannot derive the session keys without
also breaking ML-KEM. This is the Mullvad / Tailscale / PQ-WireGuard model.

---

## 2. Transport

| | |
|---|---|
| Protocol | **TCP** |
| Port | **51821/tcp** (proposed; WireGuard keeps 51820/udp) |
| TLS | none — this protocol is self-authenticating (§5). No CA, no cert. |
| Framing | length-prefixed messages, §3 |
| Timeouts | client abandons a handshake after 10 s; server drops a connection idle > 10 s or after `ServerFinish` |
| Connection lifetime | one handshake (or one rekey) per TCP connection, then close |

TCP rather than UDP because `ServerHello` carries a 3309-byte ML-DSA-65
signature plus an ML-KEM ciphertext (768–1568 B) — 4–5 KB total, past a safe
single-datagram size, and the exchange is a strict request/response that gains
nothing from UDP.

---

## 3. Message framing

Every message on the wire:

```
 0        1        2        3        4                         4+len
 +--------+--------+--------+--------+--------------------------+
 | VER    | TYPE   | LENGTH (u16, big-endian)                  | PAYLOAD (LENGTH bytes)
 +--------+--------+--------+--------+--------------------------+
```

| Field | Size | Notes |
|---|---|---|
| `VER` | 1 | protocol version. `0x01` for this document. Receiver rejects any other value with `Error(UNSUPPORTED_VERSION)`. |
| `TYPE` | 1 | message type, §4 |
| `LENGTH` | 2 | payload length in bytes, big-endian, ≤ 8192 |
| `PAYLOAD` | `LENGTH` | type-specific, §4 |

Payload fields that vary in size (public keys, ciphertext, signature) are each
written as `u16 length ‖ bytes`. Fixed-size fields (nonces, session id, the
1-byte algorithm code) are written raw. All integers big-endian.

`TYPE` values:

| Value | Message | Direction |
|---|---|---|
| `0x01` | `ClientHello` | client → server |
| `0x02` | `ServerHello` | server → client |
| `0x03` | `ClientFinish` | client → server |
| `0x04` | `ServerFinish` | server → client |
| `0x05` | `RekeyRequest` | client → server |
| `0xEE` | `Error` | either |

---

## 4. Messages

### 4.1 Algorithm codes

One byte, matching `contracts/algo_registry.json` `action_index`:

| Code | Algorithm |
|---|---|
| `0x00` | ML-KEM-512 |
| `0x01` | ML-KEM-768 |
| `0x02` | ML-KEM-1024 |

`action_index 3` (`rekey-now`) is **not** an on-wire algorithm — a rekey re-runs
the handshake at the algorithm already in force or a stronger one, never a
weaker one (§6). HQC-256 is reserved for a
future `0x03` if `algo_registry` enables it.

### 4.2 `ClientHello` (0x01)

| Field | Size | Meaning |
|---|---|---|
| `algo` | 1 | requested algorithm code (§4.1) |
| `client_nonce` | 32 | fresh random, this handshake only |
| `client_wg_pubkey` | 32 | the peer's WireGuard public key — tells the server which peer this secret is for |
| `client_x25519_pub` | `u16 ‖ 32` | classical half |
| `client_mlkem_pub` | `u16 ‖ N` | ML-KEM encapsulation key (`N` = 800 / 1184 / 1568 by `algo`) |

### 4.3 `ServerHello` (0x02)

| Field | Size | Meaning |
|---|---|---|
| `algo` | 1 | algorithm the server accepted — MUST equal the requested `algo` or the message is an `Error` instead (no silent downgrade) |
| `session_id` | 16 | server-assigned, used to correlate a later `RekeyRequest` |
| `server_nonce` | 32 | fresh random |
| `server_x25519_pub` | `u16 ‖ 32` | server's ephemeral classical half |
| `mlkem_ciphertext` | `u16 ‖ N` | ML-KEM encapsulation to `client_mlkem_pub` (`N` = 768 / 1088 / 1568) |
| `signature` | `u16 ‖ 3309` | ML-DSA-65 signature over the transcript (§5) |

### 4.4 `ClientFinish` (0x03)

| Field | Size | Meaning |
|---|---|---|
| `session_id` | 16 | echoed from `ServerHello` |
| `client_tag` | 32 | `HMAC-SHA256(confirm_key, "client finished" ‖ transcript)` — proves the client derived the same hybrid secret |

On a valid `client_tag` the server installs the PSK for `client_wg_pubkey` and
replies `ServerFinish`.

### 4.5 `ServerFinish` (0x04)

| Field | Size | Meaning |
|---|---|---|
| `session_id` | 16 | |
| `server_tag` | 32 | `HMAC-SHA256(confirm_key, "server finished" ‖ transcript)` |
| `server_wg_pubkey` | 32 | the server's WireGuard public key (raw, not base64) |
| `assigned_ip` | 4 | IPv4 address the server assigned this peer inside the tunnel |
| `wg_port` | 2 | UDP port the server's WireGuard listens on (big-endian) |

Only after `server_tag` verifies does the client trust the tunnel parameters.
It then brings its WireGuard interface up with `Address = assigned_ip`,
`Peer.PublicKey = server_wg_pubkey`, `Peer.PresharedKey = psk` (§5.3),
`Peer.Endpoint = <handshake host>:wg_port`. The PSK is **never** on the wire —
both sides derived it. Both sides close the TCP connection.

The server installs the peer on its side before sending `ServerFinish`
(`wg set <iface> peer <client_wg_pubkey> preshared-key … allowed-ips
assigned_ip/32`, where `client_wg_pubkey` came from `ClientHello` §4.2). The
`allowed-ips` argument is always sent — on a rekey it re-sets the same value, a
no-op.

**Server-side peer/session tracking (Week 5).** The server keeps a registry
keyed by `client_wg_pubkey`: one `Peer` = `{assigned_ip, session_id, algo,
timestamps, rekey count}`. A fresh `ClientHello` from a **known** pubkey
*replaces* that peer's session (new `session_id`, new PSK, same `assigned_ip`) —
this is how an algorithm change §6 is done. The registry is persisted (no PSKs)
and, on startup, reconciled against the live interface: `wg` peers the registry
does not know are removed.

### 4.6 `RekeyRequest` (0x05)

Identical body to `ClientHello`, plus a leading `session_id` (16). The server
looks the peer up by **`client_wg_pubkey`** and checks the supplied `session_id`
matches that peer's current session; if the pubkey is unknown or the
`session_id` is stale → `Error(UNKNOWN_SESSION)`. It then responds with
`ServerHello` / expects `ClientFinish` as normal, and swaps the peer's PSK
**without** removing the peer, so the tunnel does not drop (§6).

`algo` rule (`REKEY_ESCALATES`, approved — §8 Q6):
- **downgrade is always rejected** — `algo` weaker than the session's in-force
  algorithm → `Error(ALGO_MISMATCH)`. A step down in strength is a full new
  handshake (`ClientHello`), never a rekey.
- `algo` **equal to** in-force: allowed.
- `algo` **stronger than** in-force: allowed. The handshake runs at the
  requested level and the session's in-force algorithm is raised to it.

### 4.7 `Error` (0xEE)

| Field | Size | Meaning |
|---|---|---|
| `code` | 1 | see below |
| `message` | `u16 ‖ ≤256` | UTF-8, human-readable, non-authoritative |

| Code | Meaning |
|---|---|
| `0x01` | `UNSUPPORTED_VERSION` |
| `0x02` | `MALFORMED` |
| `0x03` | `ALGO_REJECTED` (server policy will not do the requested level) |
| `0x04` | `UNKNOWN_SESSION` (rekey for a session the server has no record of) |
| `0x05` | `ALGO_MISMATCH` (rekey `algo` ≠ in-force algo) |
| `0x06` | `AUTH_FAILED` (client tag did not verify) |
| `0x07` | `RATE_LIMITED` |
| `0x08` | `INTERNAL` |
| `0x09` | `CAPACITY` (server at `--max-peers`; new peer refused — Week 5) |

---

## 5. Authentication & key schedule

### 5.1 Transcript

Byte string both sides compute identically, via the one shared function
`vpn_core::crypto::hybrid_kem::build_handshake_transcript_from_bytes` (the typed
client-side `build_handshake_transcript` delegates to it):

```
transcript =
    "PQC-VPN-HANDSHAKE-v1"            (20 bytes, ASCII, no NUL)
  ‖ algo_name                        ("ML-KEM-512" | "ML-KEM-768" | "ML-KEM-1024")
  ‖ client_nonce                     (32)
  ‖ server_nonce                     (32)
  ‖ client_x25519_pub                (32)
  ‖ client_mlkem_pub                 (800 | 1184 | 1568)
  ‖ server_x25519_pub                (32)
  ‖ mlkem_ciphertext                 (768 | 1088 | 1568)
```

### 5.2 Server identity

The server has one long-term **ML-DSA-65** identity keypair, generated once
(`ServerAuthenticator::generate`). Its public key (1952 B) is **pinned in the
client** — shipped in the installer / config file, never fetched at runtime,
never TOFU. `signature` in `ServerHello` is `sign(identity_sk, transcript)`.

The client MUST verify `signature` against the pinned key **before** deriving
any secret or sending `ClientFinish`. This is the whole MitM defence: an
attacker can relay or substitute `server_x25519_pub` / `mlkem_ciphertext`, but
cannot produce a signature over the modified transcript.

### 5.3 Hybrid secret → PSK

```
hybrid_secret = SHA256(x25519_shared ‖ mlkem_shared)      // core, unchanged, 32 B

ikm  = hybrid_secret
salt = client_nonce ‖ server_nonce
psk         = HKDF-SHA256(ikm, salt, info = "pqc-vpn wg psk",   L = 32)
confirm_key = HKDF-SHA256(ikm, salt, info = "pqc-vpn confirm",  L = 32)
```

`psk` is what goes to `wg set ... preshared-key`. `confirm_key` keys the
`ClientFinish` / `ServerFinish` tags. Deriving two independent keys via HKDF
rather than reusing the raw secret keeps the confirmation MAC from ever exposing
anything about the PSK.

---

## 6. Rekey semantics

Base rule (`INTERFACE_FREEZE_PROPOSAL.md` §2.1): `rekey-now` means **rotate key
material** — the client sends `RekeyRequest` with the existing `session_id`, the
server runs a fresh handshake, derives a new `psk`, and updates the existing
peer's preshared key. The server MUST NOT `wg set ... remove` the peer or force
the tunnel down; WireGuard picks up the new PSK on its next handshake.

**Amendment — `REKEY_ESCALATES` (`contracts/DECISIONS.md` Decision 1, approved
at the Week 4 checkpoint):** Member 3's Week 3 work found the RL policy asks for
`rekey-now` on ~75% of high-threat states, some of which actually need a
stronger KEM. The client therefore rekeys at the *stronger* of {in-force
algorithm, the policy's top choice} — never weaker. A rekey can legitimately
arrive at a higher algorithm than the session started with.

Server rule: accept `RekeyRequest.algo >= in_force`, do the handshake at the
requested level, swap the PSK, and raise the session's in-force algorithm to the
new level (`registry.rs::Registry::rekey`). A **downgrade** is never a rekey, and
"when to rotate" vs "how strong" remain the client's decisions — the server just
responds.

A downgrade is a full new handshake: new `ClientHello`, new `session_id`, then
the client points its WireGuard at the new PSK.

---

## 7. Replay & rate limiting (fields defined now, enforcement in Week 7)

- `client_nonce` / `server_nonce` make every transcript unique, so a recorded
  `ServerHello` cannot be replayed into a new handshake (its signature is bound
  to the client's fresh nonce and ephemeral keys).
- Week 7 adds: a server-side cache of recently seen `client_nonce` values
  (reject duplicates within a 2-minute window) and a per-source-IP handshake
  rate limit → `Error(RATE_LIMITED)`.
- No protocol change is needed to turn these on; they are server-local.

---

## 8. Resolved questions (Week 4 checkpoint sign-off)

Kept as a record of what was asked at the freeze and what was decided.

1. **Nonces in the transcript — yes, done.** `vpn_core`'s transcript builder
   takes `client_nonce ‖ server_nonce` right after the algorithm name (Member 1,
   Week 4). The server builds its transcript through the same function
   (`build_handshake_transcript_from_bytes`), so the two cannot drift.
2. **HKDF location — in `vpn_core`, done.**
   `vpn_core::crypto::derive_session_keys(hybrid_secret, client_nonce,
   server_nonce)`; the server's `kdf::derive` now calls it. The finish MACs stay
   in the server crate until `core::protocol` (Week 6) needs them client-side.
3. **ML-KEM key ownership — keep the code.** The client holds the ephemeral
   ML-KEM keypair, the server encapsulates, and the server signs the whole
   transcript (covering both ephemeral X25519 keys, the ML-KEM public key and the
   ciphertext). Strictly stronger than "the server signs its ML-KEM public key";
   the proposal wording is corrected in `RL_PQC_VPN_Proposal_v2.md`.
4. **Port — `51821/tcp` approved.**
5. **Client identification — the WireGuard public key is the peer identity** for
   this quarter. A separate enrolment step is out of scope (future work).
6. **`REKEY_ESCALATES` — approved** (`contracts/DECISIONS.md` Decision 1). A rekey
   may carry an equal or stronger algorithm, never a weaker one (§4.6, §6).
7. **Identity persistence in `core` — done.**
   `ServerAuthenticator::{to_seed_bytes, from_seed_bytes}` exist; the server's
   `identity` module wraps `ServerAuthenticator` and no longer uses `ml-dsa`
   directly. Persisted seeds are unchanged, so the pinned key on the droplet
   stays valid.

## Reference implementation

The Week 2 server-side implementation of this document lives in
`server/handshake-server/` (built on `vpn_core::crypto`). Deterministic test
fixtures — framing, transcript, KDF, MAC — are in `server/handshake-vectors.json`
(generated by `cargo run -p handshake-server --bin emit-vectors`), for Member 1
to verify the client side of `core/protocol/` against the same bytes, the same
way `contracts/*_vectors.json` work.

---

## 9. What is frozen (signed off at the Week 4 checkpoint)

- Message types, header format, field order and encoding (§3, §4).
- The transcript definition and signing rule (§5.1, §5.2).
- The key schedule (§5.3) and rekey rule (§6).

**Not frozen:** the Week 7 replay/rate-limit tuning constants, the `Error`
message strings, and the server's algorithm-acceptance policy (which levels it
is willing to serve) — those are server config, not wire format.

Versioned: any incompatible change bumps `VER` to `0x02` and gets a
`PROTOCOL.md` v2. Member 2 maintains this document (`TEAM_TIMELINE_PROPOSAL.md`
Week 11).
