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
| EXO | [`EternaPeptix/exo`](https://github.com/EternaPeptix/exo/tree/experiment/kimi-k3-uvmax-optimization-stack-v6) | RDMA striping, rank-local checkpoints, TP2 placement/runtime integration, prompt-lookup telemetry, reproducible benchmarks, strict target verification, and layer/KDA divergence diagnostics |
| MLX-LM | [`EternaPeptix/mlx-lm`](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-uvmax-optimization-stack-v6) | Kimi K3 model support, deterministic generation control, vocabulary-parallel output head, exact fused expert/KDA paths, transactional prompt-lookup speculation, and eager asynchronous decode boundaries |
| MLX | [`EternaPeptix/mlx`](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-uvmax-optimization-stack-v6) | Exact Darwin/JACCL lineage used by the benchmarked Mac runtime, plus eval-walk and gather-index CPU-overhead reductions |

This coordinated v6 branch is an experimental research checkpoint. Its
default exact runtime is unchanged from the benchmark-accepted v5 stack. The
new EXO changes add fail-closed TP2 verification, live JACCL mesh attestation,
a real-checkpoint layer-0 KDA localizer, and first-round-inclusive speculative
throughput telemetry. The telemetry is unit-tested but has not been promoted
by a live prompt-lookup gate, so prompt lookup remains disabled by default.

The accepted stack combines the `laguna8` hidden-state asynchronous schedule
with the exact packed/fused MLX-LM paths. It produced a five-run median
`14.2375` decode tok/s on the canonical 575-token prompt and `14.2155` tok/s
on a separate 1,067-token coding prompt, retaining the deterministic digest
in every repetition and using approximately `414 GB` peak memory per rank.
Earlier Spark CUDA/MoE experiments remain available on the separate
[`experiment/exo-mlx-inference-optimizations`](https://github.com/EternaPeptix/mlx/tree/experiment/exo-mlx-inference-optimizations)
branch; they were not replayed onto this newer JACCL base.

The current width-2 diagnostic first observes full-layer drift at decoder
layer 0. A follow-up real-checkpoint localizer found the KDA attention stages
themselves exact when given the same prepared input, narrowing the unresolved
source to the surrounding decoder-layer preparation/wrapper path. The sampled
top-1 continuation still agreed, but the strict
`k3-tp2-target-verification/v3` numerical gate remains `FAIL`; neither result
promotes multi-token target verification.

The default-off exact speculative-KDA rewrite and tiled dual-source MLA
prototype remain on separate research branches. They are deliberately
excluded from v6 because they have not passed the strict full-model TP2 gate.

Lossy Kimi K3 requantization experiments are not enabled or included in the
validated runtime. Their measured output digests differed from the exact
checkpoint, so they remain research evidence rather than production defaults.

## Reproducibility pins

| Component | Pin |
| --- | --- |
| Model | [`kernelpool/Kimi-K3-2bit-UVMAX`](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX) |
| Model revision | `edb5113218df612f4a92f95145680f3f8eacd375` |
| Darwin MLX/JACCL runtime | [`EternaPeptix/mlx`](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-uvmax-optimization-stack-v6) tested code commit `57b87fe47cfce34d6dc59d0e274d8ee36bfb9308` |
| Execution-time MLX-LM | [`EternaPeptix/mlx-lm`](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-uvmax-optimization-stack-v6) accepted runtime commit `95fc8ad485e8d2568eda4e468c4169f6a556919a` |
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
  "mlx-lm @ git+https://github.com/EternaPeptix/mlx-lm.git@95fc8ad485e8d2568eda4e468c4169f6a556919a"
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

When round telemetry is enabled, the response reports decode elapsed time,
first-round-inclusive effective generation throughput, proposed/accepted/
rejected draft-token counts, and round acceptance. These counters describe
end-to-end speculation rather than the target model's batched-call throughput.

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
one-token decode. The v6 benchmark accepts an ordered subset of the audited
widths `1, 2, 3, 4, 7, 8` and gives every timed call a fresh, fully
materialized copy of the same post-prefill cache. It validates the Kimi K3
mixed cache layout (69 recurrent
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

For the accepted fused routed-up-add runtime, use
`launch_k3_target_verify_current.sh`. It requires the launcher, source,
MLX-LM, MLX core, checkpoint, transport, and artifact paths explicitly. Its
focused default is a 128-token prompt at widths 1 and 2; set
`K3_TARGET_VERIFY_WIDTHS=1,2,3,4,7,8` to run the full audited sequence after
the focused gate passes. It defaults to the live `mlx.launch --backend jaccl`
mesh with `MLX_JACCL_RING` absent and records that attestation in the artifact.
Set `K3_TARGET_VERIFY_TRANSPORT_MODE=ring` only for an explicit comparison with
the earlier `jaccl-ring` verifier results.

If width 2 fails, run `launch_k3_target_diagnostic_current.sh` to locate the
first decoder-layer and cache divergence. The companion
`launch_k3_kda_stage_localizer_current.sh` then compares the real
checkpoint's layer-0 KDA projection, normalization, convolution, recurrence,
gate, output, and cache stages. Both tools are diagnostic only: they reproduce
and localize a failure but never relax the strict promotion gate.

The decoder-layer localizer also has a fail-closed candidate selector for the
ordinary non-transactional width-2 short convolution. Run the accepted control
with `K3_DECODER_LAYER_LOCALIZER_EXACT_WIDE_SHORT_CONV=0`, then repeat with it
set to `1` against the matching MLX-LM candidate. A useful candidate artifact
must make the controlled direct-attention convolution/output/cache stages exact
while the separately measured history-producing transaction path remains exact;
the full-model layer-0 endpoint reproduction check is mandatory in both runs.

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
