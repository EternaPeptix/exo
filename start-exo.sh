#!/bin/zsh
set -euo pipefail
cd /Users/jeweled/exo
export EXO_MEMORY_THRESHOLD=0.92
export EXO_MACMON_PATH=/opt/homebrew/bin/macmon
# int8 quantize MLA latent KV after prefill chunks (indexer cache stays fp16)
# DISABLED for corruption investigation - int8 KV cache quantization may cause\n# DSA indexer top-k divergence with large contexts.\n# # export EXO_KV_BITS=8\n# export EXO_KV_GROUP_SIZE=64\nexec ./.venv/bin/exo "$@"
