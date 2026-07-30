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

## Reproducibility pins

| Component | Pin |
| --- | --- |
| Model | [`kernelpool/Kimi-K3-2bit-UVMAX`](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX) |
| Model revision | `edb5113218df612f4a92f95145680f3f8eacd375` |
| EXO-compatible MLX-LM | [`EternaPeptix/mlx-lm`](https://github.com/EternaPeptix/mlx-lm/tree/k3/exo-support) commit `4f16fb2bccba39f6c0bbc528b8c2955fd754846f` |
| Kimi K3 model-support base | [upstream MLX-LM #1626](https://github.com/ml-explore/mlx-lm/pull/1626) commit `7d505c285b801108a52c23353c7fb6af07204717` |
| Converter schema | `k3-rank-local-tp/v1` |

The loader verifies the exact upstream Kimi K3 implementation hash and fails
closed on a different sharding contract. Install the EXO-compatible dependency
at its immutable commit:

```bash
python -m pip install \
  "mlx-lm @ git+https://github.com/EternaPeptix/mlx-lm.git@4f16fb2bccba39f6c0bbc528b8c2955fd754846f"
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
```

The EXO MLX worker resolves `{rank}` from the distributed rank, verifies the
manifest and model contract, shards the empty model structure, and loads only
that rank's local tensors. Setting `EXO_MLX_RANK_LOCAL_VERIFY_HASHES=1` also
checks every weight file against its manifest at startup. Disabling that
opt-in check can shorten a controlled profiling launch, but weakens corruption
detection.

The loader file is hash-pinned by EXO. `.gitattributes` forces LF line endings
so a checkout cannot silently change that hash.

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
