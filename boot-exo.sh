#!/bin/zsh
# boot-exo.sh — start the Exo daemon on this node and kick off auto-placement.
# Identical on every host in the cluster. Invoked by the com.jeweled.exo
# LaunchDaemon (RunAtLoad + KeepAlive), so it runs at BOOT (no login required)
# and is restarted if the wrapper exits.
#
# This wrapper:
#   1. launches auto-place-model.sh in the background (it waits for the API,
#      then registers the model card and requests the 2-node shard — idempotent,
#      deterministic single-placer election so both nodes running it is safe), and
#   2. execs exo in the foreground so launchd's KeepAlive applies to exo itself.
set -euo pipefail

EXO_DIR="${EXO_DIR:-$HOME/exo}"
LOG_DIR="$HOME/.exo/exo_log"
mkdir -p "$LOG_DIR"

log() { print -r -- "[$(date "+%Y-%m-%d %H:%M:%S")] $(hostname -s): $*"; }

# Refuse to double-start: if something is already listening on the API port,
# assume a prior instance is still settling and exit cleanly so KeepAlive
# doesn't spawn a fighter.
pid_on_port=$(lsof -tiTCP:52415 -sTCP:LISTEN 2>/dev/null | head -1 || true)
if [[ -n "$pid_on_port" ]]; then
  log "API port 52415 already held by pid=$pid_on_port; not starting a second instance"
  exit 0
fi

cd "$EXO_DIR"

# Env shared across the cluster.
export EXO_MEMORY_THRESHOLD="${EXO_MEMORY_THRESHOLD:-0.92}"
export EXO_MACMON_PATH="${EXO_MACMON_PATH:-/opt/homebrew/bin/macmon}"
# Prefer the direct Thunderbolt 5 link between the two Macs over the 25G ethernet
# path through the switch for ring (pipeline) send/recv, since TB5 is ~4x faster.
# Only affects pairs that have BOTH a thunderbolt and an ethernet socket path.
export EXO_RING_LINK_PRIORITY="${EXO_RING_LINK_PRIORITY:-thunderbolt,maybe_ethernet,ethernet,wifi,unknown}"

# Explicit zenoh peer over Thunderbolt 5. Link-local 169.254 TB IPs change
# across reboots, so resolve the peer dynamically via arp on the TB bridge
# (en5). Fall back to a static zenoh-peer.env if arp finds nothing. Peering
# over TB avoids the flaky IPv6 link-local discovery path. See
# rust/networking/src/lib.rs:cfg().
if [[ -z "${EXO_ZENOH_CONNECT:-}" ]]; then
  _tb_peer=""
  # Try dynamic discovery first
  if [[ -x /usr/sbin/arp ]]; then
    _tb_peer=$(arp -an -i en5 2>/dev/null | awk -F'[ ()]' '/169\.254/{print $2; exit}')
  fi
  if [[ -n "$_tb_peer" ]]; then
    export EXO_ZENOH_CONNECT="tcp/${_tb_peer}:52414"
    log "zenoh peer resolved via arp: $EXO_ZENOH_CONNECT"
  elif [[ -f "$EXO_DIR/zenoh-peer.env" ]]; then
    source "$EXO_DIR/zenoh-peer.env"
    export EXO_ZENOH_CONNECT
    log "zenoh peer from static zenoh-peer.env: $EXO_ZENOH_CONNECT"
  fi
fi
export HOME="${HOME:-/Users/jeweled}"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Kick off auto-placement in the background. It self-throttles (waits for the
# API, waits for peers, elects a single placer) so it is safe to start before
# exo is fully up. Logged to its own file for debugging.
#
# Detach with setsid (available on macOS via util-linux/homebrew) so the child
# survives independent of this script's session; fall back to plain background
# with redirection if setsid isn't present. We deliberately do NOT use nohup
# here — under launchd there is no controlling terminal and nohup fails with
# "can't detach from console: Inappropriate ioctl for device".
if [[ -x "$EXO_DIR/auto-place-model.sh" ]]; then
  log "launching auto-place-model.sh in background"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$EXO_DIR/auto-place-model.sh" > "$LOG_DIR/auto-place.log" 2>&1 < /dev/null &
  else
    "$EXO_DIR/auto-place-model.sh" > "$LOG_DIR/auto-place.log" 2>&1 < /dev/null &
    disown 2>/dev/null || true
  fi
fi

log "starting exo: .venv/bin/exo --fast-synch --zenoh-port 52414 --discovery-port 52413"
exec "$EXO_DIR/.venv/bin/exo" --fast-synch --zenoh-port 52414 --discovery-port 52413
