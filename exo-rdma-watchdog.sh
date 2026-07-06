#!/bin/zsh
# exo-rdma-watchdog.sh — auto-recover from stale JACCL RDMA queue-pair state.
#
# The macOS/Apple-JACCL RDMA layer occasionally leaves queue pairs in a stale
# state (surfaced as RunnerFailed with "[jaccl] Changing queue pair to RTR
# failed" or "[jaccl] Recv failed"). exo's built-in retry just re-runs
# mx.distributed.init against the same stale QP, so it crash-loops forever and
# the cluster stays down until a manual reboot.
#
# This watchdog detects that failure via /state and recovers without a reboot:
#   A. rdma_ctl disable && enable  — reset the RDMA layer in place (~2s)
#   B. launchctl kickstart exo      — full process restart if A didn't work
#
# Invoked by a LaunchDaemon (com.jeweled.exo.rdma-watchdog), so it survives exo
# crashes — which is the whole point, since it must restart exo for path B.
set -uo pipefail   # NOT -e: a failing curl/grep must not kill the watchdog
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

API="${EXO_API:-http://127.0.0.1:52415}"
POLL_SECS="${EXO_WATCHDOG_POLL:-15}"
A_RECOVER_SECS="${EXO_WATCHDOG_A_RECOVER:-240}"
A_MAX_ATTEMPTS="${EXO_WATCHDOG_A_MAX:-2}"
B_RECOVER_SECS="${EXO_WATCHDOG_B_RECOVER:-360}"
COOLDOWN_SECS="${EXO_WATCHDOG_COOLDOWN:-600}"
LOG="${HOME}/.exo/exo_log/rdma-watchdog.log"
mkdir -p "${HOME}/.exo/exo_log"

log() { print -ru2 -- "[$(date "+%Y-%m-%d %H:%M:%S")] $(hostname -s): $*"; }

# fetch_status prints machine-readable key=value lines, e.g.:
#   status=ok runners=2 ready=2 failed=0 jaccl_qp=0
# or "status=api_down" if the API is unreachable.
fetch_status() {
  curl -sf --max-time 5 "$API/state" 2>/dev/null | python3 -c '
import json, re, sys
JACCL_QP_RE = r"(\[jaccl\].*(changing queue pair|recv failed)|errno 96)"
try:
    d = json.load(sys.stdin)
except Exception:
    print("status=api_down"); sys.exit()
runners = d.get("runners") or {}
n_ready = n_failed = n_jaccl_qp = 0
for v in runners.values():
    if not isinstance(v, dict): continue
    if "RunnerReady" in v: n_ready += 1
    if "RunnerFailed" in v:
        n_failed += 1
        msg = str(v.get("RunnerFailed", {}).get("error_message", ""))
        if re.search(JACCL_QP_RE, msg, re.IGNORECASE):
            n_jaccl_qp += 1
print(f"status=ok runners={len(runners)} ready={n_ready} failed={n_failed} jaccl_qp={n_jaccl_qp}")
' 2>/dev/null || echo "status=api_down"
}

# Extract a numeric field from the status line (0 if absent).
get_field() {
  local line="$1" key="$2"
  print -r -- "$line" | grep -oE "$key=[0-9]+" | head -1 | cut -d= -f2 || print 0
}

# True if a JACCL QP failure is currently indicated (jaccl_qp > 0).
is_jaccl_qp_failure() {
  local line; line="$(fetch_status)"
  [[ "$(get_field "$line" jaccl_qp)" -gt 0 ]]
}

# Path A: reset the RDMA layer in place, then delete the failed instance so the
# auto-place loop re-places against the freshly-reset RDMA layer.
do_rdma_reset() {
  log "A: rdma_ctl disable"
  sudo -n rdma_ctl disable 2>&1 | while read -r ln; do log "  $ln"; done
  sleep 2
  log "A: rdma_ctl enable"
  sudo -n rdma_ctl enable 2>&1 | while read -r ln; do log "  $ln"; done
  sleep 3
  for iid in $(curl -sf --max-time 5 "$API/state" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    for iid, iv in (d.get("instances") or {}).items():
        for body in (iv.values() if isinstance(iv, dict) else []):
            runners = (body.get("shardAssignments", {}) or {}).get("runnerToShard", {}) or {}
            if runners: print(iid); break
except Exception: pass
' 2>/dev/null); do
    log "A: deleting instance $iid to trigger re-placement"
    curl -sf -X DELETE "$API/instance/$iid" --max-time 10 >/dev/null 2>&1 || true
  done
}

# Path B: restart the whole exo process (LaunchDaemon will bring it back).
do_exo_restart() {
  log "B: restarting exo via launchctl kickstart -k system/com.jeweled.exo"
  sudo -n launchctl kickstart -k system/com.jeweled.exo 2>&1 | while read -r ln; do log "  $ln"; done
}

a_attempts=0
last_action=0
log "watchdog started (poll=${POLL_SECS}s, A_max=${A_MAX_ATTEMPTS}, cooldown=${COOLDOWN_SECS}s)"

while true; do
  sleep "$POLL_SECS"
  line="$(fetch_status)"

  if [[ "$(get_field "$line" status)" != "ok" ]]; then
    # API down (exo not up yet, or restarting). Not a jaccl failure — wait.
    continue
  fi

  jaccl_qp="$(get_field "$line" jaccl_qp)"
  if [[ "$jaccl_qp" -le 0 ]]; then
    if (( a_attempts != 0 )); then
      a_attempts=0
      log "healthy ($line); reset A attempt counter"
    fi
    continue
  fi

  # --- JACCL QP failure detected ---
  now=$(date +%s)
  if (( now - last_action < COOLDOWN_SECS )); then
    log "jaccl QP failure detected but in cooldown; waiting ($line)"
    continue
  fi

  if (( a_attempts < A_MAX_ATTEMPTS )); then
    a_attempts=$((a_attempts + 1))
    last_action=$now
    log "jaccl QP failure detected; A-path attempt $a_attempts/$A_MAX_ATTEMPTS ($line)"
    do_rdma_reset
    log "waiting up to ${A_RECOVER_SECS}s for recovery after A-path..."
    ok=0
    for i in $(seq 1 $((A_RECOVER_SECS / POLL_SECS))); do
      sleep "$POLL_SECS"
      l2="$(fetch_status)"
      jq2="$(get_field "$l2" jaccl_qp)"
      ready="$(get_field "$l2" ready)"
      if [[ "$jq2" -le 0 && "$ready" -gt 0 ]]; then
        log "A-path recovered: $l2"; ok=1; break
      fi
    done
    if (( ok )); then
      a_attempts=0
      log "recovered via A-path; resetting attempt counter"
      continue
    fi
    log "A-path did not recover within ${A_RECOVER_SECS}s"
  fi

  # A-path exhausted -> escalate to B.
  last_action=$(date +%s)
  log "escalating to B-path (full exo restart) — A-path did not recover"
  do_exo_restart
  a_attempts=0
  log "waiting ${B_RECOVER_SECS}s for exo to come back and re-shard before resuming..."
  sleep "$B_RECOVER_SECS"
done
