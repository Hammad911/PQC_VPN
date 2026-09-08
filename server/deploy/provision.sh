#!/usr/bin/env bash
# server/deploy/provision.sh
#
# Run ONCE on a fresh DigitalOcean Ubuntu 24.04 droplet, as root:
#
#     bash provision.sh <new-username>
#
# Creates a non-root sudo user (copying root's authorized SSH keys), hardens
# sshd, enables the firewall, turns on automatic security updates, installs
# WireGuard, and enables IPv4 forwarding.
#
# Idempotent: safe to re-run.

set -euo pipefail

NEW_USER="${1:-}"
if [[ -z "$NEW_USER" ]]; then
    echo "usage: bash provision.sh <new-username>" >&2
    exit 1
fi
if [[ "$(id -u)" -ne 0 ]]; then
    echo "run this as root" >&2
    exit 1
fi

echo "==> Creating user '$NEW_USER'"
if ! id "$NEW_USER" &>/dev/null; then
    adduser --disabled-password --gecos "" "$NEW_USER"
fi
usermod -aG sudo "$NEW_USER"
echo "$NEW_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-$NEW_USER"
chmod 440 "/etc/sudoers.d/90-$NEW_USER"

echo "==> Copying SSH keys to '$NEW_USER'"
install -d -m 700 -o "$NEW_USER" -g "$NEW_USER" "/home/$NEW_USER/.ssh"
if [[ -f /root/.ssh/authorized_keys ]]; then
    install -m 600 -o "$NEW_USER" -g "$NEW_USER" \
        /root/.ssh/authorized_keys "/home/$NEW_USER/.ssh/authorized_keys"
else
    echo "WARNING: /root/.ssh/authorized_keys not found - make sure you can log in as $NEW_USER" >&2
fi

echo "==> Hardening sshd (no root login, no passwords)"
install -d /etc/ssh/sshd_config.d
cat > /etc/ssh/sshd_config.d/99-hardening.conf <<'EOF'
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
systemctl restart ssh

echo "==> Firewall (ufw): SSH + WireGuard 51820/udp + handshake 51821/tcp"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y ufw >/dev/null
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 51820/udp comment 'WireGuard'
ufw allow 51821/tcp comment 'PQC handshake'
ufw --force enable

echo "==> Automatic security updates"
DEBIAN_FRONTEND=noninteractive apt-get install -y unattended-upgrades >/dev/null
dpkg-reconfigure -f noninteractive unattended-upgrades

echo "==> Installing WireGuard"
DEBIAN_FRONTEND=noninteractive apt-get install -y wireguard wireguard-tools >/dev/null

echo "==> Enabling IPv4 forwarding"
cat > /etc/sysctl.d/99-wireguard-forward.conf <<'EOF'
net.ipv4.ip_forward = 1
EOF
sysctl --system >/dev/null

echo "==> Ensuring a swap file (1 GB droplet - Rust linking needs headroom)"
if ! swapon --show | grep -q '/swapfile'; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
    swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo
echo "Provision complete."
echo "Open a NEW terminal and verify:  ssh $NEW_USER@<this-host>"
echo "Then run wireguard-baseline.sh as that user (with sudo)."
