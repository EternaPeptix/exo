#!/bin/zsh
# 512S1 coordinator: wait for Thunderbolt, bootstrap libp2p over TB, start RDMA Exo on both nodes.
set -euo pipefail
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

EXO_DIR="${EXO_DIR:-$HOME/exo}"
LOG_DIR="$HOME/.exo/exo_log"
LOCK_FILE="$HOME/.exo/exo-boot.lock"
LIBP2P_PORT_S1=50085
LIBP2P_PORT_S2=50086
S2_HOST_SHORT=512s2.local
AUTO_MODEL_ENV="${EXO_DIR}/auto-model.env"
[[ -f "$AUTO_MODEL_ENV" ]] && source "$AUTO_MODEL_ENV"
EXO_AUTO_LOAD_MODEL="${EXO_AUTO_LOAD_MODEL:-1}"
EXO_AUTO_MODEL_ID="${EXO_AUTO_MODEL_ID:-pipenetwork/GLM-5.2-MLX-8bit}"
EXO_AUTO_MODEL_SHARDING="${EXO_AUTO_MODEL_SHARDING:-Tensor}"
EXO_AUTO_MODEL_META="${EXO_AUTO_MODEL_META:-MlxJaccl}"
EXO_AUTO_MODEL_MIN_NODES="${EXO_AUTO_MODEL_MIN_NODES:-2}"

mkdir -p "$LOG_DIR"

log() {
  print -r -- "[$(date "+%Y-%m-%d %H:%M:%S")] $*" >&2
}

on_s1() {
  [[ "$(hostname -s)" == "512S1" ]]
}

require_s1() {
  if ! on_s1; then
    log "boot-exo-cluster.sh must run on 512S1 (coordinator), not $(hostname -s)"
    exit 1
  fi
}

exo_env() {
  export EXO_MEMORY_THRESHOLD=0.92
  export EXO_MACMON_PATH=/opt/homebrew/bin/macmon
  export EXO_KV_BITS=8
  export EXO_KV_GROUP_SIZE=64
}

SSH_S2=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

kill_local_exo() {
  pkill -f "/Applications/EXO.app" 2>/dev/null || true
  pkill -f ".venv/bin/exo" 2>/dev/null || true
  for pid in $(lsof -tiTCP:52415 -sTCP:LISTEN 2>/dev/null); do
    kill "$pid" 2>/dev/null || true
  done
}

kill_remote_exo() {
  local s2_tb="$1"
  "${SSH_S2[@]}" "jeweled@${s2_tb}" 'pkill -f ".venv/bin/exo" 2>/dev/null || true' || true
}

local_tb_ip() {
  ifconfig en5 2>/dev/null | awk '/inet 169\.254/{print $2; exit}'
}

discover_s2_tb_ip() {
  local ip
  for ip in $(dscacheutil -q host -a name "$S2_HOST_SHORT" 2>/dev/null | awk '/ip_address:/{print $2}' | grep '^169\.254'); do
    if "${SSH_S2[@]}" -o ConnectTimeout=5 "jeweled@${ip}" true 2>/dev/null; then
      print -r -- "$ip"
      return 0
    fi
  done
  for ip in $(arp -a 2>/dev/null | grep -i '512s2' | sed -n 's/.* (\([0-9.]*\)).*/\1/p' | grep '^169\.254'); do
    if "${SSH_S2[@]}" -o ConnectTimeout=5 "jeweled@${ip}" true 2>/dev/null; then
      print -r -- "$ip"
      return 0
    fi
  done
  return 1
}

wait_for_thunderbolt() {
  local i s1_tb s2_tb
  log "Waiting for Thunderbolt link (en5 169.254.x) and 512S2..."
  for i in {1..72}; do
    s1_tb=$(local_tb_ip)
    if [[ -n "$s1_tb" ]]; then
      s2_tb=$(discover_s2_tb_ip || true)
      if [[ -n "${s2_tb:-}" ]]; then
        print -r -- "$s1_tb" "$s2_tb"
        return 0
      fi
    fi
    sleep 5
  done
  log "ERROR: Timed out waiting for TB link to 512S2"
  return 1
}

write_jaccl_hostfile() {
  local s1_tb="$1" s2_tb="$2"
  local hf="$HOME/.jaccl_hostfile"
  log "Writing jaccl hostfile (TB IPs: $s1_tb <-> $s2_tb)"
  cat >"$hf" <<EOF
{
    "backend": "jaccl",
    "envs": [],
    "hosts": [
        {
            "ssh": "512s1.local",
            "ips": ["${s1_tb}"],
            "rdma": [null, "rdma_en5"]
        },
        {
            "ssh": "512s2.local",
            "ips": ["${s2_tb}"],
            "rdma": ["rdma_en5", null]
        }
    ]
}
EOF
  "${SSH_S2[@]}" "jeweled@${s2_tb}" \
    "cat > ~/.jaccl_hostfile" <"$hf"
}

start_exo_background() {
  local port="$1"
  ./.venv/bin/exo --fast-synch --libp2p-port "$port" >>"$LOG_DIR/nohup.out" 2>&1 &
  disown
}

wait_for_tb_enumeration() {
  local i
  log "Waiting for Thunderbolt enumeration (exo iface map)..."
  for i in {1..24}; do
    if system_profiler SPThunderboltDataType 2>/dev/null | grep -qiE "Device connected|Link Status: 0x2"; then
      sleep 10
      return 0
    fi
    sleep 5
  done
  log "WARN: Thunderbolt enumeration slow; continuing anyway"
}

start_s2_exo() {
  local s1_tb="$1" s2_tb="$2" s1_id="$3"
  local bootstrap="/ip4/${s1_tb}/tcp/${LIBP2P_PORT_S1}/p2p/${s1_id}"
  log "Starting 512S2 exo (bootstrap -> S1)"
  "${SSH_S2[@]}" "jeweled@${s2_tb}" \
    "cd ${EXO_DIR} && export EXO_MEMORY_THRESHOLD=0.92 EXO_MACMON_PATH=/opt/homebrew/bin/macmon EXO_KV_BITS=8 EXO_KV_GROUP_SIZE=64 EXO_BOOTSTRAP_PEERS='${bootstrap}' && ./.venv/bin/exo --fast-synch --libp2p-port ${LIBP2P_PORT_S2} >> ${LOG_DIR}/nohup.out 2>&1 & disown"
}

wait_for_api() {
  local url="$1" tries="${2:-30}"
  local i
  for i in $(seq 1 "$tries"); do
    if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

verify_cluster() {
  local s2_tb="$1"
  local topo
  topo=$(curl -sf http://127.0.0.1:52415/state | python3 -c \
    'import sys,json; s=json.load(sys.stdin); t=s.get("topology",{}); print(len(t.get("nodes",[])), list(s.get("nodeIdentities",{}).values()))' 2>/dev/null) || true
  log "S1 topology: ${topo:-unknown}"
  "${SSH_S2[@]}" "jeweled@${s2_tb}" \
    "curl -sf http://127.0.0.1:52415/state | python3 -c 'import sys,json; s=json.load(sys.stdin); t=s.get(\"topology\",{}); print(len(t.get(\"nodes\",[])), list(s.get(\"nodeIdentities\",{}).values()))'" \
    2>/dev/null | while read -r line; do log "S2 topology: $line"; done || true
}

model_already_loaded() {
  local model_id="$1"
  curl -sf http://127.0.0.1:52415/state | EXO_CHECK_MODEL_ID="$model_id" python3 -c '
import json, os, sys
model_id = os.environ["EXO_CHECK_MODEL_ID"]
state = json.load(sys.stdin)
for inst in state.get("instances", {}).values():
    for body in inst.values():
        sa = body.get("shardAssignments") or body.get("shard_assignments") or {}
        mid = sa.get("modelId") or sa.get("model_id") or ""
        if mid == model_id:
            sys.exit(0)
sys.exit(1)
' 2>/dev/null
}

auto_load_model() {
  local model_id="$EXO_AUTO_MODEL_ID"
  local i nodes

  if [[ "$EXO_AUTO_LOAD_MODEL" != "1" ]]; then
    log "Auto model load disabled (EXO_AUTO_LOAD_MODEL=$EXO_AUTO_LOAD_MODEL)"
    return 0
  fi

  log "Waiting for API before loading $model_id..."
  wait_for_api "http://127.0.0.1:52415/state" 90 || {
    log "ERROR: API not ready; skipping auto model load"
    return 1
  }

  log "Waiting for 2-node cluster..."
  nodes=0
  for i in {1..60}; do
    nodes=$(curl -sf http://127.0.0.1:52415/state | python3 -c \
      'import sys,json; print(len(json.load(sys.stdin).get("topology",{}).get("nodes",[])))' 2>/dev/null || echo 0)
    if [[ "$nodes" -ge 2 ]]; then
      break
    fi
    sleep 5
  done
  if [[ "$nodes" -lt 2 ]]; then
    log "ERROR: cluster has $nodes nodes; skipping auto model load"
    return 1
  fi

  if model_already_loaded "$model_id"; then
    log "Model already loaded: $model_id"
    return 0
  fi

  log "Placing $model_id ($EXO_AUTO_MODEL_SHARDING, $EXO_AUTO_MODEL_META, min_nodes=$EXO_AUTO_MODEL_MIN_NODES)"
  if ! curl -sf -X POST http://127.0.0.1:52415/place_instance \
    -H 'Content-Type: application/json' \
    -d "{\"model_id\":\"${model_id}\",\"sharding\":\"${EXO_AUTO_MODEL_SHARDING}\",\"instance_meta\":\"${EXO_AUTO_MODEL_META}\",\"min_nodes\":${EXO_AUTO_MODEL_MIN_NODES}}"; then
    log "ERROR: place_instance failed for $model_id"
    return 1
  fi
  log "place_instance accepted; waiting for runners (up to ~30 min)..."

  for i in {1..180}; do
    if model_already_loaded "$model_id"; then
      log "Model loaded: $model_id"
      return 0
    fi
    sleep 10
  done
  log "WARN: $model_id not confirmed loaded after 30 min (may still be starting)"
  return 0
}

post_boot_tasks() {
  local s2_tb="$1"
  sleep 30
  verify_cluster "$s2_tb"
  auto_load_model
}

main() {
  require_s1
  if [[ -d "${LOCK_FILE}.d" ]]; then
    if [[ -f "${LOCK_FILE}.d/pid" ]] && kill -0 "$(<"${LOCK_FILE}.d/pid")" 2>/dev/null; then
      log "Another boot-exo-cluster run is in progress (pid $(<"${LOCK_FILE}.d/pid")); exiting"
      exit 0
    fi
    log "Removing stale boot lock"
    rm -rf "${LOCK_FILE}.d"
  fi
  mkdir "${LOCK_FILE}.d"
  print -r -- "$$" >"${LOCK_FILE}.d/pid"
  trap 'rm -rf "${LOCK_FILE}.d" 2>/dev/null || true' EXIT INT TERM

  log "=== Exo TB/RDMA cluster boot (512S1 coordinator) ==="
  sleep 15

  read -r s1_tb s2_tb < <(wait_for_thunderbolt)
  log "TB ready: S1=$s1_tb S2=$s2_tb"
  wait_for_tb_enumeration

  write_jaccl_hostfile "$s1_tb" "$s2_tb"

  kill_local_exo
  kill_remote_exo "$s2_tb"
  sleep 2

  exo_env
  cd "$EXO_DIR"

  log "Starting temporary S1 exo for peer ID discovery"
  start_exo_background "$LIBP2P_PORT_S1"
  wait_for_api "http://127.0.0.1:52415/node_id" 45 || {
    log "ERROR: S1 API did not start for peer discovery"
    exit 1
  }
  s1_id=$(curl -sf http://127.0.0.1:52415/node_id | tr -d '"')
  log "S1 node_id=$s1_id"

  start_s2_exo "$s1_tb" "$s2_tb" "$s1_id"
  sleep 10
  wait_for_api "http://${s2_tb}:52415/node_id" 45 || {
    log "ERROR: S2 API did not start"
    exit 1
  }
  s2_id=$("${SSH_S2[@]}" "jeweled@${s2_tb}" \
    "curl -sf http://127.0.0.1:52415/node_id | tr -d '\"'")
  log "S2 node_id=$s2_id"

  log "Restarting S1 with bootstrap -> S2"
  kill_local_exo
  sleep 2

  export EXO_BOOTSTRAP_PEERS="/ip4/${s2_tb}/tcp/${LIBP2P_PORT_S2}/p2p/${s2_id}"
  log "EXO_BOOTSTRAP_PEERS=$EXO_BOOTSTRAP_PEERS"

  (
    post_boot_tasks "$s2_tb"
  ) &

  log "exec exo on 512S1 (LaunchAgent will keep this process alive)"
  exec ./.venv/bin/exo --fast-synch --libp2p-port "$LIBP2P_PORT_S1"
}

main "$@"
