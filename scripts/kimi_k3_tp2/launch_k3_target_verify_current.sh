#!/usr/bin/env bash
set -euo pipefail

# Run on rank 0 only after the EXO ring has been stopped deliberately. This
# launcher is intentionally deployment-inventory-free: every executable,
# source, checkpoint, transport, and artifact path must be supplied explicitly.
# mlx.launch starts rank 1 from the authenticated four-rail JACCL hostfile.

die() {
  echo "k3 current-stack target verifier: $*" >&2
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
  K3_TARGET_VERIFY_ROOT
  K3_TP_TOOLS_ROOT
  K3_TP_RANK0_ROOT
  K3_TP_RANK1_ROOT
  K3_TP_HOSTFILE
  K3_TP_TRANSPORT_CONTRACT
  K3_TARGET_VERIFY_ARTIFACT_ROOT
  K3_TP_LAUNCHER
  K3_MLX_LM_ROOT
  K3_MLX_CORE_OVERRIDE
)
for required_path in "${required_paths[@]}"; do
  require_env "${required_path}"
  require_absolute_path "${required_path}"
done
for python_root in \
  K3_TARGET_VERIFY_ROOT \
  K3_TP_TOOLS_ROOT \
  K3_MLX_LM_ROOT \
  K3_MLX_CORE_OVERRIDE; do
  require_python_path "${python_root}"
done

verify_root="${K3_TARGET_VERIFY_ROOT}"
tp_tools_root="${K3_TP_TOOLS_ROOT}"
rank0_root="${K3_TP_RANK0_ROOT}"
rank1_root="${K3_TP_RANK1_ROOT}"
hostfile="${K3_TP_HOSTFILE}"
transport_contract="${K3_TP_TRANSPORT_CONTRACT}"
artifact_root="${K3_TARGET_VERIFY_ARTIFACT_ROOT}"
launcher="${K3_TP_LAUNCHER}"
mlx_lm_root="${K3_MLX_LM_ROOT}"
mlx_core_override="${K3_MLX_CORE_OVERRIDE}"

[[ -d "${verify_root}" ]] || die "K3_TARGET_VERIFY_ROOT is not a directory"
[[ -f "${verify_root}/k3_target_verify.py" ]] ||
  die "K3_TARGET_VERIFY_ROOT has no k3_target_verify.py"
[[ -d "${tp_tools_root}" ]] || die "K3_TP_TOOLS_ROOT is not a directory"
[[ -f "${tp_tools_root}/tp2_benchmark.py" ]] ||
  die "K3_TP_TOOLS_ROOT has no tp2_benchmark.py"
[[ -f "${tp_tools_root}/rank_local_loader.py" ]] ||
  die "K3_TP_TOOLS_ROOT has no rank_local_loader.py"
[[ -d "${rank0_root}" ]] || die "rank-0 checkpoint is not a local directory"
# The rank-1 checkpoint is deliberately not tested from rank 0. The remote
# process selects and validates it before model loading.
[[ -f "${hostfile}" ]] || die "K3_TP_HOSTFILE is not a local file"
[[ -f "${transport_contract}" ]] ||
  die "K3_TP_TRANSPORT_CONTRACT is not a local file"
[[ -x "${launcher}" ]] || die "K3_TP_LAUNCHER is not executable"
[[ -d "${mlx_lm_root}/mlx_lm" ]] ||
  die "K3_MLX_LM_ROOT has no mlx_lm package"
[[ -d "${mlx_core_override}/mlx" ]] ||
  die "K3_MLX_CORE_OVERRIDE has no mlx package"

prompt_tokens="${K3_TARGET_VERIFY_PROMPT_TOKENS:-128}"
runs="${K3_TARGET_VERIFY_RUNS:-3}"
warmups="${K3_TARGET_VERIFY_WARMUPS:-1}"
widths_csv="${K3_TARGET_VERIFY_WIDTHS:-1,2}"
wired_limit="${K3_TARGET_VERIFY_WIRED_LIMIT:-1}"
file_hashes="${K3_TARGET_VERIFY_FILE_HASHES:-0}"
transport_mode="${K3_TARGET_VERIFY_TRANSPORT_MODE:-mesh}"

[[ "${prompt_tokens}" =~ ^[1-9][0-9]*$ ]] ||
  die "K3_TARGET_VERIFY_PROMPT_TOKENS must be a positive integer"
[[ "${runs}" =~ ^[1-9][0-9]*$ ]] ||
  die "K3_TARGET_VERIFY_RUNS must be a positive integer"
[[ "${warmups}" =~ ^[0-9]+$ ]] ||
  die "K3_TARGET_VERIFY_WARMUPS must be a nonnegative integer"
[[ "${widths_csv}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] ||
  die "K3_TARGET_VERIFY_WIDTHS must be a comma-separated integer list"
[[ "${wired_limit}" == "0" || "${wired_limit}" == "1" ]] ||
  die "K3_TARGET_VERIFY_WIRED_LIMIT must be 0 or 1"
[[ "${file_hashes}" == "0" || "${file_hashes}" == "1" ]] ||
  die "K3_TARGET_VERIFY_FILE_HASHES must be 0 or 1"
case "${transport_mode}" in
  mesh)
    launch_backend="jaccl"
    force_mesh=1
    [[ "${MLX_JACCL_RING+x}" != "x" ]] ||
      die "MLX_JACCL_RING must be unset for mesh"
    ;;
  ring)
    launch_backend="jaccl-ring"
    force_mesh=0
    ;;
  *)
    die "K3_TARGET_VERIFY_TRANSPORT_MODE must be mesh or ring"
    ;;
esac
# Do not let local process state leak into mlx.launch. The jaccl-ring backend
# sets MLX_JACCL_RING=1 on both remote ranks; the jaccl backend leaves it unset.
unset MLX_JACCL_RING

IFS=',' read -r -a requested_widths <<<"${widths_csv}"
supported_widths=" 1 2 3 4 7 8 "
previous_width=0
width_arguments=()
for width in "${requested_widths[@]}"; do
  [[ "${supported_widths}" == *" ${width} "* ]] ||
    die "unsupported width ${width}; choose an ordered subset of 1,2,3,4,7,8"
  ((width > previous_width)) ||
    die "K3_TARGET_VERIFY_WIDTHS must be unique and strictly increasing"
  previous_width="${width}"
  width_arguments+=(--width "${width}")
done

pythonpath="${mlx_core_override}:${mlx_lm_root}:${verify_root}:${tp_tools_root}"
width_slug="${widths_csv//,/-}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact="${artifact_root}/k3-target-verify-current-${prompt_tokens}p-w${width_slug}-${stamp}-p$$-${transport_mode}.json"

arguments=(
  "${verify_root}/k3_target_verify.py"
  --rank-checkpoint "${rank0_root}"
  --rank-checkpoint "${rank1_root}"
  --artifact "${artifact}"
  --prompt-token-target "${prompt_tokens}"
  --runs "${runs}"
  --warmups "${warmups}"
  --min-logit-cosine 0.999
  --max-logit-abs-error 1.0
  "${width_arguments[@]}"
)
if [[ "${wired_limit}" == "1" ]]; then
  arguments+=(--wired-limit)
else
  arguments+=(--no-wired-limit)
fi
if [[ "${file_hashes}" == "1" ]]; then
  arguments+=(--verify-file-hashes)
fi

mkdir -p "${artifact_root}"
[[ ! -e "${artifact}" ]] || die "refusing to overwrite ${artifact}"

exec "${launcher}" \
  --verbose \
  --backend "${launch_backend}" \
  --hostfile "${hostfile}" \
  --env "PYTHONPATH=${pythonpath}" \
  --env "K3_TP_TRANSPORT_CONTRACT=${transport_contract}" \
  --env "K3_TP_TRANSPORT_MODE=${transport_mode}" \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env "EXO_MLX_JACCL_FORCE_MESH=${force_mesh}" \
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
  -- \
  "${arguments[@]}"
