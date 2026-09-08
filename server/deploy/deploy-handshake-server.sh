#!/usr/bin/env bash
# server/deploy/deploy-handshake-server.sh
#
# Build the handshake server (release, x86_64-linux) and deploy it to the
# droplet as a systemd service. Run from the repo root in WSL:
#
#     bash server/deploy/deploy-handshake-server.sh [DROPLET_IP] [SSH_USER]
#
# Defaults: 139.59.62.4  member2
#
# Idempotent. The server's ML-DSA-65 identity is generated on first start and
# persisted at /var/lib/pqc-vpn/server-identity.seed — it survives redeploys.

set -euo pipefail

IP="${1:-139.59.62.4}"
SSH_USER="${2:-member2}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# shellcheck disable=SC1090
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

echo "==> building release binary"
cargo build --release -p handshake-server --bin handshake-server

echo "==> uploading to $SSH_USER@$IP"
scp -o BatchMode=yes -q target/release/handshake-server "$SSH_USER@$IP:/tmp/handshake-server"
scp -o BatchMode=yes -q server/deploy/handshake-server.service "$SSH_USER@$IP:/tmp/handshake-server.service"

echo "==> installing service on the droplet"
ssh -o BatchMode=yes "$SSH_USER@$IP" 'sudo bash -s' <<'REMOTE'
set -euo pipefail
id pqcvpn >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin pqcvpn
install -Dm755 /tmp/handshake-server /opt/pqc-vpn/handshake-server
install -Dm644 /tmp/handshake-server.service /etc/systemd/system/handshake-server.service
rm -f /tmp/handshake-server /tmp/handshake-server.service
systemctl daemon-reload
systemctl enable --now handshake-server
systemctl restart handshake-server
sleep 1
systemctl --no-pager --full status handshake-server | sed -n '1,10p'
echo
echo "==> pinned verifying key (clients need this):"
cat /var/lib/pqc-vpn/server-identity.pub; echo
REMOTE

echo
echo "Deployed. Test from here with:"
echo "  VK=\$(ssh $SSH_USER@$IP 'sudo cat /var/lib/pqc-vpn/server-identity.pub')"
echo "  ./target/debug/test-client $IP:51821 \"\$VK\" --algo 768"
