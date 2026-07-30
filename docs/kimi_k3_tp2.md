# Kimi K3 2-bit UVMAX: rank-local TP2 on EXO

This procedure converts one pinned Kimi K3 checkpoint into two rank-local MLX
checkpoints, so each Mac loads only its tensor-parallel slice. The source
checkpoint is downloaded once by the conversion coordinator; each resulting
rank directory is then transferred only to its assigned host.

The implementation is deliberately narrow:

- two MLX ranks only (`TP=2`);
- text-model tensors only (vision tensors are excluded);
- one pinned model revision and one reviewed MLX-LM Kimi K3 contract;
- source-only tooling, with no host inventory or deployment credentials.

## Coordinated public branches

The cluster work is published under the same branch name in all three forks:

| Repository | Public branch | Scope |
| --- | --- | --- |
| EXO | [`EternaPeptix/exo`](https://github.com/EternaPeptix/exo/tree/experiment/kimi-k3-uvmax-optimization-stack-v3) | RDMA striping, rank-local checkpoints, TP2 placement/runtime integration, prompt-lookup integration, reproducible benchmarks, and target-divergence diagnostics |
| MLX-LM | [`EternaPeptix/mlx-lm`](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-uvmax-optimization-stack-v3) | Kimi K3 model support, deterministic generation control, vocabulary-parallel output head, exact expert/down/router/AttnRes/KDA paths, asynchronous decode, authoritative packing, and exact row-tiled KDA prefill |
| MLX | [`EternaPeptix/mlx`](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-uvmax-optimization-stack-v3) | Exact Darwin/JACCL lineage used by the benchmarked Mac runtime, plus eval-walk and gather-index CPU-overhead reductions |

This coordinated branch is experimental. Its exact path remains the reference
configuration. The opt-in segmented decode experiment improved median short
decode throughput from `12.1544` to `12.6206` tok/s on the two-Mac TP2 setup,
but changed the deterministic completion digest. It is published for
reproduction and further investigation, not enabled as a production default.
The opt-in `laguna8` hidden-state asynchronous schedule improved a matched
three-repetition exact run from `12.0465` to `12.9399` tok/s (`7.4%`) while
retaining the canonical completion digest and approximately `414 GB` peak
memory per rank.
Adding the authoritative packed MoE front to that exact schedule produced
`12.9576` tok/s versus its matched `12.8921` asynchronous control (`+0.51%`)
over five repetitions, retained the canonical digest, and removed
approximately `6.93 GiB` of persistent duplicate projection storage per rank.
Adding the exact fused-expert path to the authoritative-pack/row-4 stack
raised the canonical 575-prompt/128-decode median from `12.9824` to
`13.2985` tok/s (`+2.44%`) over five candidate repetitions. All repetitions
retained the canonical completion digest, and a separate 1,067-token coding
prompt reached `13.2637` tok/s while retaining its own reference digest.
Fusing the expert down projection, BF16 route multiplication, and exact
top-16 reduction then raised the canonical median to `13.5835` tok/s
(`+2.14%` versus fused experts alone, `+4.63%` cumulatively) while retaining
the same canonical digest in all five repetitions. The coding prompt reached
`13.5102` tok/s (`+1.86%` versus fused experts alone, `+4.50%`
cumulatively) with its reference digest unchanged.

The first fused-router prototype reached `13.7189` median decode tok/s, but
used a SIMD denominator reduction and changed all five canonical completion
digests from `c84d…` to `905f…`; that implementation was rejected. The
corrected router preserves MLX's sequential FP32 denominator fold and raised
the exact AttnRes stack from `13.6714` to `13.8160` tok/s (`+1.06%`).

Adding the exact zero-copy KDA skinny-projection pack then produced the current
best exact median: `14.0268` tok/s over five canonical repetitions (`+1.53%`
over the corrected-router stack and `+3.26%` over fused down/reduction alone).
The matched 1,067-token coding prompt reached `13.9971` tok/s (`+1.57%` over
the corrected-router stack). Every measured completion retained its reference
digest, and the latest change saved `1.088 ms/token`.

The row-tiled KDA prefill kernel is bit-exact and now also has a matched
full-model TP2 result. At 2K target context it reached `147.4168` prompt
tok/s versus `146.3746` (`+0.71%`). At 8K it reached `154.9942` prompt tok/s
versus a `153.2673` warmed control (`+1.13%`). Peak memory was effectively
unchanged: approximately `419.74 GB` at 2K and `426.92 GB` at 8K.

The sanitized run record, including per-repetition throughput, memory,
configuration pins, and completion digests, is published in
[`kimi_k3_tp2_benchmark_20260730.json`](kimi_k3_tp2_benchmark_20260730.json).
Earlier Spark CUDA/MoE experiments remain available on the separate
[`experiment/exo-mlx-inference-optimizations`](https://github.com/EternaPeptix/mlx/tree/experiment/exo-mlx-inference-optimizations)
branch; they were not replayed onto this newer JACCL base.

The companion width-2 diagnostic also found that batched target verification
first diverges in the first recurrent KDA layer. The sampled top-1 continuation
still agreed, but the strict `k3-tp2-target-verification/v3` numerical gate
remains `FAIL`; the diagnostic cannot promote that result.

Lossy Kimi K3 requantization experiments are not enabled or included in the
validated runtime. Their measured output digests differed from the exact
checkpoint, so they remain research evidence rather than production defaults.

## Reproducibility pins

| Component | Pin |
| --- | --- |
| Model | [`kernelpool/Kimi-K3-2bit-UVMAX`](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX) |
| Model revision | `edb5113218df612f4a92f95145680f3f8eacd375` |
| Darwin MLX/JACCL runtime | [`EternaPeptix/mlx`](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-uvmax-optimization-stack-v3) tested code commit `57b87fe47cfce34d6dc59d0e274d8ee36bfb9308` |
| Execution-time MLX-LM | [`EternaPeptix/mlx-lm`](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-uvmax-optimization-stack-v3) coordinated commit `2787e74691376dca045c0fd55bf503eb6499c05b`; live-tested code commit `5d20e13a73118642d1bb539c7a333ef60843af73` |
| Checkpoint converter / Kimi K3 model-support base | [upstream MLX-LM #1626](https://github.com/ml-explore/mlx-lm/pull/1626) commit `7d505c285b801108a52c23353c7fb6af07204717` |
| Converter schema | `k3-rank-local-tp/v1` |

Rank-local manifests retain the converter commit and Kimi K3 source-file digest
that produced their slices. The loader verifies those identities independently
from the imported execution-time `kimi_k3.py` digest and fails closed on either
a different sharding contract or a different execution source. The immutable
MLX-LM dependency commit pins the rest of the package; the single-file digest
does not attest the other MLX-LM modules. Install the EXO-compatible dependency
at its execution commit:

```bash
python -m pip install \
  "mlx-lm @ git+https://github.com/EternaPeptix/mlx-lm.git@2787e74691376dca045c0fd55bf503eb6499c05b"
```

## License and trust boundary

The model weights are governed by the Kimi K3 License, not EXO's Apache
License. Read the
[official Kimi K3 license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)
before downloading the checkpoint.

Place an authentic copy of that license in the metadata directory as
`LICENSE`, `LICENSE.md`, or `LICENSE.txt`. Conversion fails if it is absent and
copies it byte-for-byte into both rank checkpoints.

Only explicitly named model and tokenizer metadata files are copied. Unknown
files are ignored, symlinks are rejected, and existing destinations are never
overwritten; byte-identical files are reused. Model Python files in the
allowlist are third-party code: review them before executing any workflow that
enables remote model code.

## Prepare metadata

Create a metadata directory containing the pinned model's:

- `config.json`;
- `model.safetensors.index.json`;
- tokenizer/configuration files needed by the model; and
- Kimi K3 license file.

The converter verifies SHA-256 digests for the pinned config and weight index.
An inexpensive remote contract check downloads only those two files:

```bash
python scripts/kimi_k3_tp2/k3_tp_checkpoint.py audit-remote
```

You can also audit already downloaded metadata:

```bash
python scripts/kimi_k3_tp2/k3_tp_checkpoint.py audit \
  --config "$K3_METADATA/config.json" \
  --index "$K3_METADATA/model.safetensors.index.json"
```

## Convert once, produce both ranks

Choose separate local output directories. `--rank-dir` order defines rank 0
then rank 1:

```bash
python scripts/kimi_k3_tp2/k3_tp_checkpoint.py convert \
  --metadata-dir "$K3_METADATA" \
  --rank-dir "$K3_OUTPUT/rank0" \
  --rank-dir "$K3_OUTPUT/rank1" \
  --cache-dir "$K3_CACHE"
```

With no `--source-dir`, each pinned source shard is downloaded to the cache,
stream-converted into both ranks, validated, and removed after successful
publication. Add `--keep-source` if the coordinator should retain downloaded
source weights.

For a pre-existing read-only snapshot, add
`--source-dir "$K3_SOURCE"`. User-supplied source files are never deleted. For
resumable or distributed conversion, use `audit-shard` and `convert-shard`; the
latter validates both staged rank outputs before atomically publishing either
final shard and never modifies its source shard.

Each completed rank directory contains a rank-specific
`model.safetensors.index.json`, `tp_manifest.json`, the allowed metadata, the
Kimi K3 license, and only that rank's weight slices. Transfer `rank0` to the
rank-0 host and `rank1` to the rank-1 host.

## Configure EXO

Set the same loader path on both hosts and a rank-templated checkpoint path:

```bash
export EXO_MLX_RANK_LOCAL_LOADER="$EXO_REPO/scripts/kimi_k3_tp2/rank_local_loader.py"
export EXO_MLX_RANK_LOCAL_CHECKPOINT="/models/kimi-k3-tp2/rank{rank}"
export EXO_MLX_RANK_LOCAL_VERIFY_HASHES=1
export EXO_MLX_K3_VOCAB_PARALLEL_HEAD=1
export MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES=laguna8
export MLX_LM_KIMI_K3_ASYNC_DECODE_STATE=hidden
export MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1
export MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL=1
export MLX_LM_KIMI_K3_FUSED_EXPERTS=1
export MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE=1
export MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS=1
export MLX_LM_KIMI_K3_FUSED_ROUTER=1
export MLX_LM_KIMI_K3_PACKED_KDA_SKINNY=1
```

The EXO MLX worker resolves `{rank}` from the distributed rank, verifies the
manifest and model contract, shards the empty model structure, and loads only
that rank's local tensors. Setting `EXO_MLX_RANK_LOCAL_VERIFY_HASHES=1` also
checks every weight file against its manifest at startup. Disabling that
opt-in check can shorten a controlled profiling launch, but weakens corruption
detection.

The loader file is hash-pinned by EXO. `.gitattributes` forces LF line endings
so a checkout cannot silently change that hash.

The vocabulary-parallel option row-shards Kimi K3's untied output projection
after the rank-local weights are loaded. Each tensor rank computes half of the
vocabulary projection, then an all-gather reconstructs the standard full-logit
result. It is opt-in because the full-logit gather scales with the number of
prompt positions; measure representative long prompts before enabling it for
a latency-sensitive production workload.

The asynchronous schedule, authoritative packed MoE-front, row-tiled KDA
prefill, exact fused-expert/down/router, AttnRes/RMSNorm, and KDA skinny-pack
paths are also opt-in. The authoritative paths repoint original parameters to
zero-copy views instead of retaining unpacked duplicates. The row-tiled path
activates only for supported Metal prefill shapes of at least 128 tokens. The
decode fusions activate only for their released-checkpoint Metal geometry and
quantization contracts. Unsupported shapes use the reference path. Keep the
variables unset when reproducing a feature-off control.

### Prompt-lookup speculative decode

A compatible MLX-LM build can use repeated token sequences already present in
the prompt as draft continuations, without loading a separate draft model:

```bash
export EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS=7
export EXO_MLX_PROMPT_LOOKUP_MAX_NGRAM_SIZE=4
export EXO_MLX_PROMPT_LOOKUP_ROUND_TELEMETRY=1
```

The opt-in is disabled when `EXO_MLX_PROMPT_LOOKUP_NUM_TOKENS` is absent. EXO
supplies MLX-LM with the complete logical prompt as lookup-only history while
retaining its two-token post-prefill decode boundary. This is important for
both fresh and prefix-cache-hit requests: the history seeds only the n-gram
index and is not processed into the model cache a second time.

This path is experimental. Kimi K3's strict-v3 multi-token target-equivalence
gate currently fails, so a verified multi-token block can change a
deterministic greedy completion relative to sequential one-token decode. Keep
the feature off when exact output reproducibility is required.

The token count must be 1–7, the maximum n-gram size must be 2–64, and
telemetry must be exactly `0` or `1`. Companion variables without the enabling
token count are rejected. Pipeline-parallel and batch generation are also
rejected rather than silently running a different decode path. Tensor
parallelism remains supported. The MLX-LM checkout must expose the
`prompt_lookup_history` argument; enabling this option against an older build
fails at the generation boundary.

## Validate before the full model

Run the source-only unit suite:

```bash
python -m pytest -q scripts/kimi_k3_tp2/tests
```

Then run the tiny MLX two-rank equivalence check in an environment with MLX and
the distributed launcher available:

```bash
python scripts/kimi_k3_tp2/tiny_mlx_tp_equivalence.py --help
```

The tiny test compares distributed rank-local output with an unsharded
reference before committing hours to full checkpoint conversion.

The compatible MLX-LM branch also exercises dense and quantized output heads
under two local ranks:

```bash
mlx.launch -n 2 tests/model_parallel_tests.py
```

## Benchmark target-verification widths

Before integrating a speculative decoder, measure whether the unmodified
target can verify multiple proposed tokens more cheaply than sequential
one-token decode. The benchmark fixes widths to `1, 2, 3, 4, 7, 8` and gives
every timed call a fresh, fully materialized copy of the same post-prefill
cache. It validates the Kimi K3 mixed cache layout (69 recurrent
`ArraysCache` layers and 24 `KVCache` layers), preserves KV capacity and
offsets, and fails closed unless both ranks agree on finite logits, top-1
tokens, the known continuation, and numerical error limits.

Stage `scripts/kimi_k3_tp2/` at the same absolute path on both ranks. Copy
`transport-jaccl-tp2.example.json` to a private deployment file and replace
its coordinator and rail placeholders. The benchmark independently checks the
runtime JACCL state against this contract, so a stale or one-rail launch fails
closed. Supply the contract, hostfile, and rank-local checkpoint roots
explicitly, then launch from rank 0:

```bash
K3_TP_HOSTFILE=/path/to/hosts-jaccl-tp2.json \
K3_TP_TRANSPORT_CONTRACT=/same/path/on/both/ranks/transport-jaccl-tp2.json \
K3_TP_RANK0_ROOT=/path/on/rank0/to/checkpoint \
K3_TP_RANK1_ROOT=/path/on/rank1/to/checkpoint \
K3_TARGET_VERIFY_ROOT=/path/to/scripts/kimi_k3_tp2 \
K3_TP_TOOLS_ROOT=/path/to/scripts/kimi_k3_tp2 \
bash scripts/kimi_k3_tp2/launch_k3_target_verify.sh
```

The atomic JSON artifact reports critical-path target and sequential timing,
verified tokens per second, memory, cache attestation, source/runtime hashes,
transport identity, and a per-width PASS/FAIL equivalence record. This is a
measurement harness, not a claim that speculative decoding is already
implemented.

## Reference performance

On two 512 GB M3 Ultra systems connected by a four-rail JACCL fabric, the
development checkpoint produced the following representative text-only
measurements. They are hardware/configuration observations, not performance
guarantees:

| Prompt | Prefill | Decode |
| ---: | ---: | ---: |
| 8K | about 158 tok/s | about 11.7 tok/s |
| 64K | about 130 tok/s | about 10.4 tok/s |
| 128K | about 101 tok/s | about 9.2 tok/s |

For the vocabulary-parallel output-head A/B, a deterministic 575-token prompt
and 128-token greedy decode produced these three-repetition medians:

| TP2 configuration | Prefill | Decode |
| --- | ---: | ---: |
| Four-rail JACCL mesh | 113.69 tok/s | 12.0382 tok/s |
| Mesh + vocabulary-parallel head | 114.41 tok/s | 12.0866 tok/s |

The completion digest was identical. The measured decode improvement was
0.40% over the mesh baseline and 1.35% over the original ring baseline. This
short-context result does not establish the cost of the full-logit gather at
8K–1M prompt lengths.

Preserving one compiled model and reusing a compatible prefix cache matters
more at long context than converter changes. Benchmark cold prefill and cached
prefill separately, and record the exact quantization, generated-token count,
fabric rail count, EXO commit, MLX-LM commit, and manifest digests.

## Limitations

- This is not a general-purpose checkpoint converter.
- TP world sizes other than two are rejected.
- The separate PP2 transport code in this experimental branch is not part of
  the validated TP2 result. Its TCP control channel is unauthenticated and
  intended only for an isolated, trusted fabric; do not expose it to an
  untrusted network.
- The loader does not make an incompatible or newer MLX-LM sharding contract
  safe; update the pins, hash, and equivalence tests together.
- Vision tensors are intentionally omitted.
- Rank-local checkpoints are derivative model artifacts and must retain the
  copied Kimi K3 license.
