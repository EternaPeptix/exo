#!/bin/zsh
# boot-exo.sh — start the Exo daemon on this node.
# Identical on every host in the cluster. Invoked by the com.jeweled.exo
# LaunchAgent (RunAtLoad + KeepAlive), so it runs at boot/login and is restarted
# if it exits. Self-contained — does not depend on start-exo.sh (which has
# historically carried mangled comments that commented out its exec line).
set -euo pipefail

EXO_DIR="${EXO_DIR:-$HOME/exo}"
LOG_DIR="$HOME/.exo/exo_log"
mkdir -p "$LOG_DIR"

log() { print -r -- "[$(date "+%Y-%m-%d %H:%M:%S")] $(hostname -s): $*"; }

# Refuse to double-start: if something is already listening on the API port,
# assume a prior instance is still settling (or launchd is restarting us) and
# exit cleanly so KeepAlive doesn't spawn a fighter.
pid_on_port=$(lsof -tiTCP:52415 -sTCP:LISTEN 2>/dev/null | head -1 || true)
if [[ -n "$pid_on_port" ]]; then
  log "API port 52415 already held by pid=$pid_on_port; not starting a second instance"
  exit 0
fi

cd "$EXO_DIR"

# Env shared across the cluster.
export EXO_MEMORY_THRESHOLD="${EXO_MEMORY_THRESHOLD:-0.92}"
export EXO_MACMON_PATH="${EXO_MACMON_PATH:-/opt/homebrew/bin/macmon}"

log "starting exo: .venv/bin/exo --fast-synch --zenoh-port 52414 --discovery-port 52413"
exec "$EXO_DIR/.venv/bin/exo" --fast-synch --zenoh-port 52414 --discovery-port 52413
