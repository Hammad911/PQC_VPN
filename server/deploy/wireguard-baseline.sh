#!/usr/bin/env bash
# server/deploy/wireguard-baseline.sh
#
# Run ONCE on the droplet after provision.sh, with sudo:
#
#     sudo bash wireguard-baseline.sh
#
# Brings up a plain (non-PQC) WireGuard server on wg0:
#   - subnet 10.8.0.0/24, server = 10.8.0.1
#   - NAT / masquerade out the public interface
#   - listen on 51820/udp
#   - starts on boot (wg-quick@wg0)
#
# No peers are added here - use add-peer.sh for that.
# Idempotent: re-running regenerates config from the existing server key.

set -euo pipefail
if [[ "$(id -u)" -ne 0 ]]; then echo "run with sudo" >&2; exit 1; fi

WG_DIR=/etc/wireguard
WG_IFACE=wg0
WG_SUBNET_CIDR=10.8.0.0/24
WG_SERVER_ADDR=10.8.0.1/24
WG_PORT=51820

# Public-facing interface (the one with the default route).
PUB_IFACE="$(ip -4 route show default | awk '{print $5; exit}')"
echo "==> Public interface detected: $PUB_IFACE"

install -d -m 700 "$WG_DIR"

if [[ ! -f "$WG_DIR/server_private.key" ]]; then
    echo "==> Generating server keypair"
    umask 077
    wg genkey | tee "$WG_DIR/server_private.key" | wg pubkey > "$WG_DIR/server_public.key"
fi
SERVER_PRIV="$(cat "$WG_DIR/server_private.key")"
SERVER_PUB="$(cat "$WG_DIR/server_public.key")"

# Preserve any [Peer] blocks already in a previous wg0.conf.
PEER_BLOCKS=""
if [[ -f "$WG_DIR/$WG_IFACE.conf" ]]; then
    PEER_BLOCKS="$(awk '/^\[Peer\]/{p=1} p{print}' "$WG_DIR/$WG_IFACE.conf")"
fi

echo "==> Writing $WG_DIR/$WG_IFACE.conf"
cat > "$WG_DIR/$WG_IFACE.conf" <<EOF
[Interface]
Address    = $WG_SERVER_ADDR
ListenPort = $WG_PORT
PrivateKey = $SERVER_PRIV
PostUp   = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT; iptables -t nat -A POSTROUTING -s $WG_SUBNET_CIDR -o $PUB_IFACE -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT; iptables -t nat -D POSTROUTING -s $WG_SUBNET_CIDR -o $PUB_IFACE -j MASQUERADE
EOF

if [[ -n "$PEER_BLOCKS" ]]; then
    echo "" >> "$WG_DIR/$WG_IFACE.conf"
    echo "$PEER_BLOCKS" >> "$WG_DIR/$WG_IFACE.conf"
    echo "==> Preserved existing [Peer] blocks"
fi
chmod 600 "$WG_DIR/$WG_IFACE.conf"

echo "==> Enabling wg-quick@$WG_IFACE"
systemctl enable "wg-quick@$WG_IFACE" >/dev/null 2>&1 || true
systemctl restart "wg-quick@$WG_IFACE"

echo
echo "Baseline WireGuard is up."
echo "  server public key : $SERVER_PUB"
echo "  listening on      : $WG_PORT/udp"
wg show "$WG_IFACE"
echo
echo "Next: sudo bash add-peer.sh <name>"
