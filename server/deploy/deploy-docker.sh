#!/usr/bin/env bash
# server/deploy/deploy-docker.sh
#
# Containerised deploy of the handshake server. Run from the repo root in WSL:
#
#     bash server/deploy/deploy-docker.sh [DROPLET_IP] [SSH_USER]
#
# Defaults: 139.59.62.4  member2
#
# Installs Docker on the droplet if needed, ships the current working tree as the
# build context (works with uncommitted changes), stops the systemd service, and
# brings the container up with `docker compose`. The ML-DSA-65 identity is a
# bind-mounted volume (/var/lib/pqc-vpn) so it is unchanged across the switch.

set -euo pipefail

IP="${1:-139.59.62.4}"
SSH_USER="${2:-member2}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_SRC=/opt/pqc-vpn-src

ssh_do() { ssh -o BatchMode=yes "$SSH_USER@$IP" "$@"; }

echo "==> ensuring Docker on the droplet"
ssh_do 'command -v docker >/dev/null || {
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2
    sudo systemctl enable --now docker
}'

echo "==> packaging build context"
tar czf /tmp/pqc-ctx.tar.gz -C "$REPO" \
    --exclude='./.git' --exclude='./target' --exclude='*/target' \
    --exclude='./venv' --exclude='*/node_modules' --exclude='./node_modules' \
    --exclude='./client/rl_agent/models' --exclude='./notebooks' \
    --exclude='./dashboard' --exclude='*.pyc' \
    .

echo "==> uploading + extracting to $REMOTE_SRC"
scp -o BatchMode=yes -q /tmp/pqc-ctx.tar.gz "$SSH_USER@$IP:/tmp/"
rm -f /tmp/pqc-ctx.tar.gz
ssh_do "sudo rm -rf $REMOTE_SRC && sudo mkdir -p $REMOTE_SRC \
    && sudo tar xzf /tmp/pqc-ctx.tar.gz -C $REMOTE_SRC && rm /tmp/pqc-ctx.tar.gz \
    && sudo chown -R $SSH_USER:$SSH_USER $REMOTE_SRC"

echo "==> stopping the systemd service (Docker takes over)"
ssh_do 'sudo systemctl disable --now handshake-server 2>/dev/null || true'

# The container runs as root but with cap_drop: ALL, so uid 0 no longer bypasses
# file permissions (no CAP_DAC_OVERRIDE). The identity dir was created by the
# systemd service as user `pqcvpn`; hand it to root so the container can read it.
echo "==> handing /var/lib/pqc-vpn to root for the container"
ssh_do 'sudo mkdir -p /var/lib/pqc-vpn && sudo chown -R root:root /var/lib/pqc-vpn && sudo chmod 700 /var/lib/pqc-vpn'

echo "==> docker compose up --build  (first build compiles Rust; a few minutes)"
ssh_do "cd $REMOTE_SRC/server && sudo docker compose up -d --build"

echo
echo "==> status"
ssh_do 'sudo docker ps --filter name=pqc-handshake-server --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
echo
ssh_do 'sudo docker logs --tail 15 pqc-handshake-server'
