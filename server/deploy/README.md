# server/deploy — VPS provisioning (Member 2, Week 1)

Gets a fresh DigitalOcean droplet from nothing to a hardened Ubuntu host running
a **plain (non-PQC) WireGuard tunnel** — the known-good baseline the PQC
handshake server is added onto in Week 2+.

Scripts here are also the start of the Week 11 "deployment automation"
deliverable, so every manual step gets captured as code as we go.

```
deploy/
├── README.md              this file
├── wsl-setup.sh           local dev-environment setup (run in WSL, not on the VPS)
├── provision.sh           run once on a fresh droplet as root
├── wireguard-baseline.sh  run once after provision.sh — brings up wg0
└── add-peer.sh            run per client — prints a ready client config
```

---

## 1. Manual steps (you, in a browser — ~10 min)

1. **Get DigitalOcean credit.** GitHub Student Developer Pack
   (`education.github.com/pack`) → DigitalOcean offer → **$200 for 12 months**.
2. **Add your SSH key to DigitalOcean.** If you don't have one yet, in WSL:
   ```bash
   ssh-keygen -t ed25519 -C "pqc-vpn"        # accept defaults
   cat ~/.ssh/id_ed25519.pub                  # paste this into DO → Settings → Security → SSH Keys
   ```
3. **Create the droplet.** Create → Droplets:
   - Image: **Ubuntu 24.04 (LTS) x64**
   - Type: **Basic → Regular → $6/mo** (1 GB / 1 vCPU — 512 MB is too tight to `cargo build` on the box)
   - Region: closest to you (**BLR1 / Bangalore** from Islamabad)
   - Authentication: **SSH key** (the one from step 2) — not a password
   - Hostname: `pqc-vpn-server`
4. **Note the public IPv4** shown after creation. Everything below calls it `$DROPLET_IP`.

---

## 2. Provision (run on the droplet as root)

From WSL:

```bash
DROPLET_IP=xxx.xxx.xxx.xxx        # from step 1.4
NEW_USER=member2                  # the non-root account to create

scp server/deploy/provision.sh root@$DROPLET_IP:/root/
ssh root@$DROPLET_IP "bash /root/provision.sh $NEW_USER"
```

`provision.sh` does: create `$NEW_USER` with sudo + your SSH key, disable root
SSH login and password auth, enable the firewall (SSH + `51820/udp` +
`51821/tcp`), turn on automatic security updates, install WireGuard, and enable
IPv4 forwarding.

After it finishes, **open a new terminal and confirm you can log in as the new
user before closing the root session:**

```bash
ssh member2@$DROPLET_IP        # should work
ssh root@$DROPLET_IP           # should now be refused
```

---

## 3. Bring up the baseline WireGuard tunnel

```bash
ssh member2@$DROPLET_IP
sudo bash /path/to/wireguard-baseline.sh      # scp it over the same way, or git clone the repo on the box
```

This generates the server keypair, writes `/etc/wireguard/wg0.conf` (subnet
`10.8.0.0/24`, server `10.8.0.1`), sets up NAT to the public interface, and
enables `wg-quick@wg0`. Check:

```bash
sudo wg show                  # interface listed, no peers yet
```

---

## 4. Add your laptop as a peer and test

On the droplet:

```bash
sudo bash add-peer.sh laptop
```

It prints a complete client config. Save it in WSL as `wg-laptop.conf`.

**Smoke test (split tunnel — safe for WSL).** Full-tunnel routing (`AllowedIPs
= 0.0.0.0/0`) inside WSL2 can disturb WSL's own networking, so for a quick
check use a copy with `AllowedIPs = 10.8.0.0/24` and no `DNS` line:

```bash
sed -e 's#^AllowedIPs .*#AllowedIPs   = 10.8.0.0/24#' -e '/^DNS /d' \
    wg-laptop.conf > wg-test.conf
sudo wg-quick up ./wg-test.conf     # local sudo prompts for your WSL password
ping -c4 10.8.0.1                   # tunnel works
sudo wg show                        # 'latest handshake' + non-zero transfer
sudo wg-quick down ./wg-test.conf
```

**Full-tunnel test** — use the real `wg-laptop.conf` with the WireGuard app on
Windows (not WSL): import it, toggle on, then `curl ifconfig.me` should show
`$DROPLET_IP`.

That's the Week 1 target: a real WireGuard tunnel, no PQC yet. Week 2 adds the
handshake server ([`../PROTOCOL.md`](../PROTOCOL.md)) that replaces the peer's
static PSK with a derived quantum-safe one.

---

## Notes

- The proposal names Ubuntu 22.04; 24.04 LTS is used here (newer kernel and
  WireGuard userspace, supported to 2029). Either is fine — nothing in the
  stack depends on the difference.
- Docker is intentionally **not** installed yet — that's the Week 4
  containerisation task. The baseline runs WireGuard straight on the host so
  there's a simple reference point before adding a container boundary.
