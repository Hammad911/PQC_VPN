# syntax=docker/dockerfile:1
#
# PQC-VPN handshake server (server/handshake-server).
# Build context is the repo root:  docker build -f server/Dockerfile .
#
# Multi-stage: a Rust builder, then a slim Debian runtime that carries only the
# binary plus `wireguard-tools` (the server shells out to `wg set` / `wg show`).

# ---- builder ----------------------------------------------------------------
FROM rust:1-slim-bookworm AS builder
WORKDIR /build

# Copy the whole workspace. cargo needs every member manifest to resolve the
# workspace, but only `handshake-server` and its deps are compiled.
COPY . .

# BuildKit cache mounts keep the crate registry and target dir out of the image
# layer and make rebuilds fast. Copy the binary out before the mounts unmount.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release -p handshake-server --bin handshake-server \
    && cp target/release/handshake-server /handshake-server

# ---- runtime --------------------------------------------------------------
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends wireguard-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /handshake-server /usr/local/bin/handshake-server

# Identity seed + derived .pub live here; mount a volume so they survive
# container recreation (the .pub is pinned in every client).
VOLUME ["/var/lib/pqc-vpn"]

EXPOSE 51821/tcp
ENTRYPOINT ["/usr/local/bin/handshake-server"]
CMD ["--bind", "0.0.0.0:51821", \
     "--identity", "/var/lib/pqc-vpn/server-identity.seed", \
     "--wg-interface", "wg0", \
     "--wg-port", "51820", \
     "--tunnel-cidr", "10.8.0.0/24"]
