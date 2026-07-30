#!/usr/bin/env bash
set -euo pipefail

# Run on rank 0 after stopping EXO on both hosts. mlx.launch starts rank 1
# from the authenticated four-rail JACCL hostfile.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
diagnostic_root="${K3_TARGET_DIAGNOSTIC_ROOT:-${script_dir}}"
tp_tools_root="${K3_TP_TOOLS_ROOT:-${script_dir}}"
rank0_root="${K3_TP_RANK0_ROOT:?set K3_TP_RANK0_ROOT}"
rank1_root="${K3_TP_RANK1_ROOT:?set K3_TP_RANK1_ROOT}"
hostfile="${K3_TP_HOSTFILE:?set K3_TP_HOSTFILE}"
transport_contract="${K3_TP_TRANSPORT_CONTRACT:?set K3_TP_TRANSPORT_CONTRACT}"
artifact_root="${K3_TARGET_DIAGNOSTIC_ARTIFACT_ROOT:-${diagnostic_root}/artifacts}"
launcher="${K3_TP_LAUNCHER:-}"
if [[ -z "${launcher}" ]]; then
  launcher="$(command -v mlx.launch || true)"
fi
if [[ -z "${launcher}" ]]; then
  echo "mlx.launch was not found; set K3_TP_LAUNCHER" >&2
  exit 2
fi
pythonpath="${diagnostic_root}:${tp_tools_root}"
if [[ -n "${K3_MLX_LM_ROOT:-}" ]]; then
  pythonpath="${pythonpath}:${K3_MLX_LM_ROOT}"
fi
if [[ -n "${K3_MLX_CORE_OVERRIDE:-}" ]]; then
  pythonpath="${K3_MLX_CORE_OVERRIDE}:${pythonpath}"
fi
prompt_tokens="${K3_TARGET_DIAGNOSTIC_PROMPT_TOKENS:-128}"
width="${K3_TARGET_DIAGNOSTIC_WIDTH:-2}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact_name="k3-target-diagnostic-${prompt_tokens}p-w${width}-${stamp}.json"
artifact="${artifact_root}/${artifact_name}"

arguments=(
  "${diagnostic_root}/k3_target_diagnostic.py"
  --rank-checkpoint "${rank0_root}"
  --rank-checkpoint "${rank1_root}"
  --artifact "${artifact}"
  --prompt-token-target "${prompt_tokens}"
  --width "${width}"
)
if [[ "${K3_TARGET_DIAGNOSTIC_WIRED_LIMIT:-1}" == "1" ]]; then
  arguments+=(--wired-limit)
else
  arguments+=(--no-wired-limit)
fi
if [[ "${K3_TARGET_DIAGNOSTIC_FILE_HASHES:-0}" == "1" ]]; then
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
  --env MLX_LM_KIMI_K3_FUSED_EXPERTS=0 \
  -- \
  "${arguments[@]}"
