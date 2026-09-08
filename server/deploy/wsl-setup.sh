#!/usr/bin/env bash
# server/deploy/wsl-setup.sh
#
# One-time setup for the Member 2 dev environment (WSL2 Ubuntu, and later the
# Ubuntu VPS). Installs the Rust toolchain and the C build dependencies the
# `oqs` crate needs to compile liboqs from source. WireGuard is installed now
# too since later weeks need it.
#
# Run once, from the repo root:
#     bash server/deploy/wsl-setup.sh
#
# Safe to re-run. Asks for your sudo password once (for apt).

set -euo pipefail

echo "==> Installing build dependencies via apt (needs sudo)"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    clang \
    libclang-dev \
    pkg-config \
    libssl-dev \
    git \
    curl \
    ca-certificates \
    wireguard \
    wireguard-tools

if command -v rustc >/dev/null 2>&1; then
    echo "==> Rust already installed: $(rustc --version)"
else
    echo "==> Installing Rust via rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path
fi

# Make cargo available in this shell and future ones.
if ! grep -qs '.cargo/env' "$HOME/.bashrc"; then
    echo '. "$HOME/.cargo/env"' >> "$HOME/.bashrc"
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"

echo
echo "==> Installed versions:"
rustc --version
cargo --version
cmake --version | head -n1
clang --version | head -n1
wg --version 2>/dev/null || echo "wg: (WireGuard tools installed; kernel module loads on first use)"

echo
echo "Setup complete."
echo "Next:  cd server/crypto-spike && cargo run"
