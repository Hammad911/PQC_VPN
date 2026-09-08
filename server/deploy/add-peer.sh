#!/usr/bin/env bash
# server/deploy/add-peer.sh
#
# Run on the droplet with sudo:
#
#     sudo bash add-peer.sh <peer-name>
#
# Generates a keypair for one client, appends it as a [Peer] to wg0, and prints
# a complete client config to stdout. Save that output on the client as
# <peer-name>.conf and `wg-quick up` it.
#
# This is the *static PSK* baseline. From Week 2 the PSK is replaced per session
# by the value the PQC handshake server derives (see ../PROTOCOL.md); this
# script stays useful for quick connectivity tests.

set -euo pipefail
if [[ "$(id -u)" -ne 0 ]]; then echo "run with sudo" >&2; exit 1; fi

PEER_NAME="${1:-}"
[[ -z "$PEER_NAME" ]] && { echo "usage: sudo bash add-peer.sh <peer-name>" >&2; exit 1; }

WG_DIR=/etc/wireguard
WG_IFACE=wg0
WG_CONF="$WG_DIR/$WG_IFACE.conf"
PEER_DIR="$WG_DIR/peers"
[[ -f "$WG_CONF" ]] || { echo "run wireguard-baseline.sh first" >&2; exit 1; }

install -d -m 700 "$PEER_DIR"
umask 077

# Next free address in 10.8.0.0/24 (server holds .1).
USED="$(grep -oE '10\.8\.0\.[0-9]+' "$WG_CONF" || true)"
NEXT=2
while grep -q "10.8.0.$NEXT\b" <<<"$USED"; do NEXT=$((NEXT+1)); done
PEER_ADDR="10.8.0.$NEXT"

PEER_PRIV="$(wg genkey)"
PEER_PUB="$(wg pubkey <<<"$PEER_PRIV")"
PEER_PSK="$(wg genpsk)"
echo "$PEER_PRIV" > "$PEER_DIR/$PEER_NAME.key"
echo "$PEER_PSK"  > "$PEER_DIR/$PEER_NAME.psk"

SERVER_PUB="$(cat "$WG_DIR/server_public.key")"
SERVER_ENDPOINT="$(curl -s4 ifconfig.me || hostname -I | awk '{print $1}'):51820"

cat >> "$WG_CONF" <<EOF

[Peer]
# $PEER_NAME
PublicKey    = $PEER_PUB
PresharedKey = $PEER_PSK
AllowedIPs   = $PEER_ADDR/32
EOF

# Apply without dropping existing peers.
wg syncconf "$WG_IFACE" <(wg-quick strip "$WG_IFACE")

echo "==> Added peer '$PEER_NAME' as $PEER_ADDR" >&2
echo "==> Client config below - save as $PEER_NAME.conf on the client:" >&2
echo >&2

cat <<EOF
[Interface]
PrivateKey = $PEER_PRIV
Address    = $PEER_ADDR/24
DNS        = 1.1.1.1

[Peer]
PublicKey    = $SERVER_PUB
PresharedKey = $PEER_PSK
Endpoint     = $SERVER_ENDPOINT
AllowedIPs   = 0.0.0.0/0
PersistentKeepalive = 25
EOF
