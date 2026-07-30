#!/usr/bin/env bash
set -euo pipefail

# Run this wrapper on rank 0. The same script and Python environment must exist
# on both hosts; each rank uses its own checkpoint path.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tools_root="${K3_TP_TOOLS_ROOT:-${script_dir}}"
rank0_root="${K3_TP_RANK0_ROOT:?set K3_TP_RANK0_ROOT}"
rank1_root="${K3_TP_RANK1_ROOT:?set K3_TP_RANK1_ROOT}"
hostfile="${K3_TP_HOSTFILE:?set K3_TP_HOSTFILE}"
transport_contract="${K3_TP_TRANSPORT_CONTRACT:?set K3_TP_TRANSPORT_CONTRACT}"
artifact_root="${K3_TP_ARTIFACT_ROOT:-${tools_root}/artifacts}"
launcher="${K3_TP_LAUNCHER:-}"
if [[ -z "${launcher}" ]]; then
  launcher="$(command -v mlx.launch || true)"
fi
if [[ -z "${launcher}" ]]; then
  echo "mlx.launch was not found; set K3_TP_LAUNCHER" >&2
  exit 2
fi
pythonpath="${tools_root}"
if [[ -n "${K3_MLX_LM_ROOT:-}" ]]; then
  pythonpath="${pythonpath}:${K3_MLX_LM_ROOT}"
fi
prompt_tokens="${K3_TP_PROMPT_TOKENS:-2048}"
output_tokens="${K3_TP_OUTPUT_TOKENS:-32}"
runs="${K3_TP_RUNS:-3}"
prefill_step="${K3_TP_PREFILL_STEP_SIZE:-2048}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact="${artifact_root}/k3-tp2-jaccl-${prompt_tokens}p-${output_tokens}d-${stamp}.json"

benchmark_args=(
  "${tools_root}/tp2_benchmark.py"
  --rank-checkpoint "${rank0_root}"
  --rank-checkpoint "${rank1_root}"
  --artifact "${artifact}"
  --prompt-token-target "${prompt_tokens}"
  --output-tokens "${output_tokens}"
  --runs "${runs}"
  --prefill-step-size "${prefill_step}"
)
if [[ -n "${K3_TP_MAX_KV_SIZE:-}" ]]; then
  benchmark_args+=(--max-kv-size "${K3_TP_MAX_KV_SIZE}")
fi
if [[ "${K3_TP_VERIFY_FILE_HASHES:-0}" == "1" ]]; then
  benchmark_args+=(--verify-file-hashes)
fi
if [[ "${K3_TP_WIRED_LIMIT:-1}" == "1" ]]; then
  benchmark_args+=(--wired-limit)
else
  benchmark_args+=(--no-wired-limit)
fi
if [[ "${K3_TP_ASYNC_LOOKAHEAD:-1}" == "1" ]]; then
  benchmark_args+=(--async-lookahead)
else
  benchmark_args+=(--no-async-lookahead)
fi

mkdir -p "${artifact_root}"
exec "${launcher}" \
  --verbose \
  --backend jaccl-ring \
  --hostfile "${hostfile}" \
  --env "PYTHONPATH=${pythonpath}" \
  --env "K3_TP_TRANSPORT_CONTRACT=${transport_contract}" \
  --env MLX_METAL_FAST_SYNCH=1 \
  -- \
  "${benchmark_args[@]}"
