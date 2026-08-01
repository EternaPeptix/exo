#!/usr/bin/env bash
set -euo pipefail

# Run on rank 0 only after EXO has been stopped deliberately. mlx.launch starts
# rank 1 from the authenticated four-rail JACCL hostfile.

die() {
  echo "k3 current-stack decoder-layer stage localizer: $*" >&2
  exit 2
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "set ${name}"
}

require_absolute_path() {
  local name="$1"
  local value="${!name}"
  [[ "${value}" == /* ]] || die "${name} must be an absolute path"
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] ||
    die "${name} must not contain a newline"
}

require_python_path() {
  local name="$1"
  require_absolute_path "${name}"
  [[ "${!name}" != *:* ]] || die "${name} must not contain ':'"
}

required_paths=(
  K3_DECODER_LAYER_LOCALIZER_ROOT
  K3_TP_TOOLS_ROOT
  K3_TP_RANK0_ROOT
  K3_TP_RANK1_ROOT
  K3_TP_HOSTFILE
  K3_TP_TRANSPORT_CONTRACT
  K3_DECODER_LAYER_LOCALIZER_ARTIFACT_ROOT
  K3_TP_LAUNCHER
  K3_MLX_LM_ROOT
  K3_MLX_CORE_OVERRIDE
)
for required_path in "${required_paths[@]}"; do
  require_env "${required_path}"
  require_absolute_path "${required_path}"
done
for python_root in \
  K3_DECODER_LAYER_LOCALIZER_ROOT \
  K3_TP_TOOLS_ROOT \
  K3_MLX_LM_ROOT \
  K3_MLX_CORE_OVERRIDE; do
  require_python_path "${python_root}"
done

localizer_root="${K3_DECODER_LAYER_LOCALIZER_ROOT}"
tp_tools_root="${K3_TP_TOOLS_ROOT}"
rank0_root="${K3_TP_RANK0_ROOT}"
rank1_root="${K3_TP_RANK1_ROOT}"
hostfile="${K3_TP_HOSTFILE}"
transport_contract="${K3_TP_TRANSPORT_CONTRACT}"
artifact_root="${K3_DECODER_LAYER_LOCALIZER_ARTIFACT_ROOT}"
launcher="${K3_TP_LAUNCHER}"
mlx_lm_root="${K3_MLX_LM_ROOT}"
mlx_core_override="${K3_MLX_CORE_OVERRIDE}"

[[ -d "${localizer_root}" ]] ||
  die "K3_DECODER_LAYER_LOCALIZER_ROOT is not a directory"
[[ -f "${localizer_root}/k3_decoder_layer_stage_localizer.py" ]] ||
  die "K3_DECODER_LAYER_LOCALIZER_ROOT has no decoder-layer localizer"
[[ -f "${localizer_root}/k3_kda_stage_localizer.py" ]] ||
  die "K3_DECODER_LAYER_LOCALIZER_ROOT has no KDA companion localizer"
[[ -f "${localizer_root}/k3_target_diagnostic.py" ]] ||
  die "K3_DECODER_LAYER_LOCALIZER_ROOT has no target diagnostic"
[[ -d "${tp_tools_root}" ]] || die "K3_TP_TOOLS_ROOT is not a directory"
[[ -f "${tp_tools_root}/tp2_benchmark.py" ]] ||
  die "K3_TP_TOOLS_ROOT has no tp2_benchmark.py"
[[ -f "${tp_tools_root}/rank_local_loader.py" ]] ||
  die "K3_TP_TOOLS_ROOT has no rank_local_loader.py"
[[ -d "${rank0_root}" ]] || die "rank-0 checkpoint is not a local directory"
# Rank 1 validates its own checkpoint after distributed initialization.
[[ -f "${hostfile}" ]] || die "K3_TP_HOSTFILE is not a local file"
[[ -f "${transport_contract}" ]] ||
  die "K3_TP_TRANSPORT_CONTRACT is not a local file"
[[ -x "${launcher}" ]] || die "K3_TP_LAUNCHER is not executable"
[[ -d "${mlx_lm_root}/mlx_lm" ]] ||
  die "K3_MLX_LM_ROOT has no mlx_lm package"
[[ -d "${mlx_core_override}/mlx" ]] ||
  die "K3_MLX_CORE_OVERRIDE has no mlx package"

transport_mode="${K3_DECODER_LAYER_LOCALIZER_TRANSPORT_MODE:-mesh}"
file_hashes="${K3_DECODER_LAYER_LOCALIZER_FILE_HASHES:-0}"
[[ "${transport_mode}" == "mesh" ]] ||
  die "K3_DECODER_LAYER_LOCALIZER_TRANSPORT_MODE must be mesh"
[[ "${file_hashes}" == "0" || "${file_hashes}" == "1" ]] ||
  die "K3_DECODER_LAYER_LOCALIZER_FILE_HASHES must be 0 or 1"
[[ "${MLX_JACCL_RING+x}" != "x" ]] ||
  die "MLX_JACCL_RING must be unset for the authenticated mesh"
unset MLX_JACCL_RING

pythonpath="${mlx_core_override}:${mlx_lm_root}:${localizer_root}:${tp_tools_root}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact="${artifact_root}/k3-decoder-layer-stage-localizer-current-128p-w2-${stamp}-p$$-mesh.json"
arguments=(
  "${localizer_root}/k3_decoder_layer_stage_localizer.py"
  --rank-checkpoint "${rank0_root}"
  --rank-checkpoint "${rank1_root}"
  --artifact "${artifact}"
  --min-cosine 0.999
  --max-abs-error 1.0
)
if [[ "${file_hashes}" == "1" ]]; then
  arguments+=(--verify-file-hashes)
fi

mkdir -p "${artifact_root}"
[[ ! -e "${artifact}" ]] || die "refusing to overwrite ${artifact}"

exec "${launcher}" \
  --verbose \
  --backend jaccl \
  --hostfile "${hostfile}" \
  --env "PYTHONPATH=${pythonpath}" \
  --env "K3_TP_TRANSPORT_CONTRACT=${transport_contract}" \
  --env K3_TP_TRANSPORT_MODE=mesh \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env EXO_MLX_JACCL_FORCE_MESH=1 \
  --env EXO_MLX_K3_VOCAB_PARALLEL_HEAD=1 \
  --env EXO_MLX_K3_REQUANT_ROUTED_LATENT_MXFP4=0 \
  --env EXO_MLX_K3_REQUANT_ATTENTION_QKVG_MXFP4=0 \
  --env MLX_LM_KIMI_K3_FUSED_EXPERTS=1 \
  --env MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE=1 \
  --env MLX_LM_KIMI_K3_FUSED_ROUTER=1 \
  --env MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS=1 \
  --env MLX_LM_KIMI_K3_PACKED_KDA_SKINNY=1 \
  --env MLX_LM_KIMI_K3_PACKED_KDA_WIDE=1 \
  --env MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD=1 \
  --env MLX_LM_KIMI_K3_FUSED_POST_KDA_RMS_SIGMOID_GATE=0 \
  --env MLX_LM_KIMI_K3_COMPILED_DECODE=0 \
  --env MLX_LM_KIMI_K3_PACKED_MOE_FRONT=0 \
  --env MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1 \
  --env MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL=1 \
  --env MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES=laguna8 \
  --env MLX_LM_KIMI_K3_ASYNC_DECODE_STATE=hidden \
  --env MLX_LM_KIMI_K3_EXACT_SPECULATIVE_KDA=0 \
  -- \
  "${arguments[@]}"
