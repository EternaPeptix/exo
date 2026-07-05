#!/bin/zsh
# auto-place-model.sh — after the local Exo API is up, register the GLM-5.2 custom
# model card and request a 2-node tensor/MlxJaccl shard of it.
#
# Identical on every host. To avoid a double-placement race when both nodes boot
# at the same time, only ONE node performs the placement: the node whose own
# node-id is the lexicographically smallest among all visible peers. Every node
# computes this deterministically from /state, so the choice is unanimous and
# needs no cross-node coordination. If the elected placer has not placed the
# model within a generous window, the other node takes over as a fallback.
#
# Invoked by the com.jeweled.exo LaunchAgent as a RunAtLoad companion task.
set -euo pipefail
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

EXO_DIR="${EXO_DIR:-$HOME/exo}"
API="http://127.0.0.1:52415"
AUTO_MODEL_ENV="${EXO_DIR}/auto-model.env"
[[ -f "$AUTO_MODEL_ENV" ]] && source "$AUTO_MODEL_ENV"
EXO_AUTO_LOAD_MODEL="${EXO_AUTO_LOAD_MODEL:-1}"
EXO_AUTO_MODEL_ID="${EXO_AUTO_MODEL_ID:-pipenetwork/GLM-5.2-MLX-8bit}"
EXO_AUTO_MODEL_SHARDING="${EXO_AUTO_MODEL_SHARDING:-Tensor}"
EXO_AUTO_MODEL_META="${EXO_AUTO_MODEL_META:-MlxJaccl}"
EXO_AUTO_MODEL_MIN_NODES="${EXO_AUTO_MODEL_MIN_NODES:-2}"

log() { print -r -- "[$(date "+%Y-%m-%d %H:%M:%S")] $(hostname -s): $*"; }

wait_for_api() {
  local tries="${1:-90}" i
  for i in $(seq 1 "$tries"); do
    curl -sf --max-time 3 "$API/state" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

self_node_id() {
  curl -sf --max-time 5 "$API/node_id" 2>/dev/null | tr -d '"\n'
}

topology_node_ids() {
  curl -sf --max-time 5 "$API/state" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
t = d.get("topology") or {}
for n in (t.get("nodes") or []):
    nid = n.get("nodeId") if isinstance(n, dict) else n
    if nid:
        print(nid)
' 2>/dev/null
}

# Requires N peers (including self) to be visible in the local topology view.
wait_for_peers() {
  local need="${1:-2}" tries="${2:-120}" i got
  for i in $(seq 1 "$tries"); do
    got=$(curl -sf --max-time 3 "$API/state" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    t = d.get("topology") or {}
    print(len(t.get("nodes") or []))
except Exception:
    print(0)
' 2>/dev/null || echo 0)
    [[ "$got" -ge "$need" ]] && return 0
    sleep 3
  done
  return 1
}

model_already_loaded() {
  curl -sf --max-time 5 "$API/state" 2>/dev/null | EXO_CHECK_MODEL_ID="$1" python3 -c '
import json, os, sys
model_id = os.environ["EXO_CHECK_MODEL_ID"]
try:
    state = json.load(sys.stdin)
except Exception:
    sys.exit(2)
for inst in state.get("instances", {}).values():
    for body in inst.values():
        sa = body.get("shardAssignments") or body.get("shard_assignments") or {}
        mid = sa.get("modelId") or sa.get("model_id") or ""
        if mid == model_id:
            sys.exit(0)
sys.exit(1)
'
}

register_card() {
  local model_id="$1"
  # Idempotent: re-registering a known card is harmless. Makes the model appear
  # in /v1/models so clients can select it.
  curl -sf --max-time 30 -X POST "$API/models/add" \
    -H 'Content-Type: application/json' \
    -d "{\"model_id\":\"${model_id}\"}" >/dev/null 2>&1 || true
}

# Purge the macOS disk cache so exo's memory accounting sees reclaimable pages as
# available. Safe and non-hanging (unlike `memory_pressure -l warn`). Best-effort:
# ignore failures (no sudo, not on macOS, etc.).
reclaim_memory() {
  if command -v purge >/dev/null 2>&1; then
    sudo -n purge 2>/dev/null || true
  fi
}

# Returns 0 (true) if THIS node is the elected placer (smallest node-id among
# all visible peers), 1 otherwise.
am_elected_placer() {
  local self peers smallest
  self="${1:-}"
  [[ -z "$self" ]] && return 1
  peers=$(topology_node_ids | sort -u)
  smallest=$(echo "$peers" | head -1)
  [[ "$self" == "$smallest" ]]
}

if [[ "$EXO_AUTO_LOAD_MODEL" != "1" ]]; then
  log "auto model load disabled (EXO_AUTO_LOAD_MODEL != 1 in auto-model.env)"
  exit 0
fi

log "waiting for local Exo API..."
wait_for_api 90 || { log "API never came up; giving up"; exit 1; }
log "API up"

register_card "$EXO_AUTO_MODEL_ID"
log "model card registered: $EXO_AUTO_MODEL_ID"

if model_already_loaded "$EXO_AUTO_MODEL_ID"; then
  log "model already placed by a peer; nothing to do"
  exit 0
fi

log "waiting for $EXO_AUTO_MODEL_MIN_NODES peers before placing..."
wait_for_peers "$EXO_AUTO_MODEL_MIN_NODES" 120 || log "peers did not appear in time; will still attempt placement"

# Drop disk cache so the placement planner sees the most available memory
# possible. Especially helpful right after boot when the model files were just
# read from disk and macOS is caching them as "active".
reclaim_memory

# Phase 1: only the elected (smallest node-id) node places. Others wait and
# watch for the model to appear. This guarantees a single placer even when both
# nodes boot simultaneously.
SELF=$(self_node_id)
log "self node_id: $SELF"

FALLBACK_AFTER=${EXO_AUTO_PLACE_FALLBACK_SECONDS:-420}
deadline=$(( $(date +%s) + FALLBACK_AFTER ))
placed=0
while [[ $(date +%s) -lt $deadline ]]; do
  if model_already_loaded "$EXO_AUTO_MODEL_ID"; then
    log "model is placed (by self or peer); done"
    placed=1
    break
  fi
  if am_elected_placer "$SELF"; then
    log "I am the elected placer; calling place_instance"
    # place_instance returns 200 immediately (command accepted). The placement
    # itself happens asynchronously in the master and can fail transiently
    # (insufficient reported memory right after boot, RDMA topology not yet
    # settled). The model_already_loaded check at the top of the loop is the
    # source of truth; retry place_instance until it appears or we time out.
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -X POST "$API/place_instance" \
        -H 'Content-Type: application/json' \
        -d "{\"model_id\":\"${EXO_AUTO_MODEL_ID}\",\"sharding\":\"${EXO_AUTO_MODEL_SHARDING}\",\"instance_meta\":\"${EXO_AUTO_MODEL_META}\",\"min_nodes\":${EXO_AUTO_MODEL_MIN_NODES}}}" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      log "place_instance accepted (http 200) for $EXO_AUTO_MODEL_ID; waiting for it to appear"
      # Give the master time to actually create the instance before re-checking.
      sleep 20
    else
      log "place_instance returned http $code; will retry"
      sleep 15
    fi
  else
    sleep 10
  fi
done

if [[ "$placed" -eq 0 ]] && ! model_already_loaded "$EXO_AUTO_MODEL_ID"; then
  log "elected placer did not finish within ${FALLBACK_AFTER}s; attempting placement as fallback"
  curl -s -o /dev/null --max-time 20 -X POST "$API/place_instance" \
      -H 'Content-Type: application/json' \
      -d "{\"model_id\":\"${EXO_AUTO_MODEL_ID}\",\"sharding\":\"${EXO_AUTO_MODEL_SHARDING}\",\"instance_meta\":\"${EXO_AUTO_MODEL_META}\",\"min_nodes\":${EXO_AUTO_MODEL_MIN_NODES}}}" >/dev/null 2>&1 || true
fi

log "auto-place-model.sh done"
