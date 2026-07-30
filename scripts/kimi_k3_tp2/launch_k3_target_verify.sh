#!/usr/bin/env bash
set -euo pipefail

# Run on rank 0. mlx.launch starts rank 1 from the authenticated four-rail
# JACCL hostfile supplied through K3_TP_HOSTFILE.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
verify_root="${K3_TARGET_VERIFY_ROOT:-${script_dir}}"
tp_tools_root="${K3_TP_TOOLS_ROOT:-${script_dir}}"
rank0_root="${K3_TP_RANK0_ROOT:?set K3_TP_RANK0_ROOT}"
rank1_root="${K3_TP_RANK1_ROOT:?set K3_TP_RANK1_ROOT}"
hostfile="${K3_TP_HOSTFILE:?set K3_TP_HOSTFILE}"
transport_contract="${K3_TP_TRANSPORT_CONTRACT:?set K3_TP_TRANSPORT_CONTRACT}"
artifact_root="${K3_TARGET_VERIFY_ARTIFACT_ROOT:-${verify_root}/artifacts}"
launcher="${K3_TP_LAUNCHER:-}"
if [[ -z "${launcher}" ]]; then
  launcher="$(command -v mlx.launch || true)"
fi
if [[ -z "${launcher}" ]]; then
  echo "mlx.launch was not found; set K3_TP_LAUNCHER" >&2
  exit 2
fi
pythonpath="${verify_root}:${tp_tools_root}"
if [[ -n "${K3_MLX_LM_ROOT:-}" ]]; then
  pythonpath="${pythonpath}:${K3_MLX_LM_ROOT}"
fi
prompt_tokens="${K3_TARGET_VERIFY_PROMPT_TOKENS:-64}"
runs="${K3_TARGET_VERIFY_RUNS:-3}"
warmups="${K3_TARGET_VERIFY_WARMUPS:-1}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact="${artifact_root}/k3-target-verify-${prompt_tokens}p-${stamp}.json"

arguments=(
  "${verify_root}/k3_target_verify.py"
  --rank-checkpoint "${rank0_root}"
  --rank-checkpoint "${rank1_root}"
  --artifact "${artifact}"
  --prompt-token-target "${prompt_tokens}"
  --runs "${runs}"
  --warmups "${warmups}"
)
if [[ "${K3_TARGET_VERIFY_WIRED_LIMIT:-1}" == "1" ]]; then
  arguments+=(--wired-limit)
else
  arguments+=(--no-wired-limit)
fi
if [[ "${K3_TARGET_VERIFY_FILE_HASHES:-0}" == "1" ]]; then
  arguments+=(--verify-file-hashes)
fi

mkdir -p "${artifact_root}"
exec "${launcher}" \
  --verbose \
  --backend jaccl-ring \
  --hostfile "${hostfile}" \
  --env "PYTHONPATH=${pythonpath}" \
  --env "K3_TP_TRANSPORT_CONTRACT=${transport_contract}" \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env EXO_MLX_K3_VOCAB_PARALLEL_HEAD=1 \
  --env EXO_MLX_K3_REQUANT_ROUTED_LATENT_MXFP4=0 \
  --env EXO_MLX_K3_REQUANT_ATTENTION_QKVG_MXFP4=0 \
  -- \
  "${arguments[@]}"
