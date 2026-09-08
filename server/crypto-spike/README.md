# crypto-spike

Throwaway Week 1 validation binary for Member 2. **This is not the server.**

## What it does

Member 1 already ported the cryptographic layer to Rust in [`core/src/crypto/`](../../core/src/crypto/)
— pure-Rust `ml-kem` / `ml-dsa` / `x25519-dalek`, **no liboqs**. This binary:

1. Exercises `core::crypto` the way the handshake responder will:
   client keygen → `server_encapsulate` → server signs the transcript →
   client verifies → `client_decapsulate` → assert both secrets match, for
   ML-KEM-512 / 768 / 1024.
2. Also drives `core::crypto::hybrid_kem::run_local_authenticated_handshake`.
3. Prints the **encoded byte size** of every value that crosses the wire
   (X25519 publics, ML-KEM public, ML-KEM ciphertext, ML-DSA-65 signature,
   full signed transcript) — the inputs to message framing in `PROTOCOL.md`.

## Run

```bash
# from the workspace root
cargo run -p crypto-spike
```

No extra system packages — the crypto is pure Rust. (`server/deploy/wsl-setup.sh`
still installs cmake/clang/WireGuard for later weeks.)

## Notes carried into the wire-protocol draft

- The signed transcript layout comes straight from
  `core::crypto::hybrid_kem::build_handshake_transcript`:
  `b"PQC-VPN-HANDSHAKE-v1" ‖ algo_name ‖ client_x25519_pub ‖ client_mlkem_pub ‖ server_x25519_pub ‖ ciphertext`.
- In the current port the **client** owns the ML-KEM keypair and the server
  encapsulates; the server's only asymmetric value is its ephemeral X25519 key,
  which is what the ML-DSA-65 signature (over the whole transcript) protects.
  The proposal's wording ("server signs its ML-KEM public key") does not match
  this — worth a line in the freeze meeting, though the transcript-signing
  approach is stronger than signing a single key.
