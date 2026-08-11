# Kimi K3 deferred W3 async: current-lineage EXO freeze

This document freezes the EXO-side port for a separate canary review. It is
source and test evidence only. It does not enable the feature, authorize a
TP2 run, or add any throughput credit.

## Lineage and scope

- EXO base: `d30cc7b73460ff34d32545b5bc5118555c0716d5`
  (`experiment/k3-width4-receipt-tail-overlap-composed-v1`)
- Deferred-W3 donor: `2309421da607345d470b27949f38b5e4317e2c2d`
- Merge-base retained for provenance: `04fe574f993471ece70a53651e669d5ea1f6f6c0`
- MLX-LM source worktree: `56dc6fd6ef3be21fa08d25b741f6ab45a3254709`
- Feature is default-off, width-three-only, and requires the exact paired
  selectors `EXO_MLX_KIMI_K3_DEFERRED_ASYNC_WIDTH3=1` and
  `MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3=1`.
- Width-four receipt and deferred width-three async are mutually rejected.
  The current d30 width-four receipt and tail-overlap contracts remain intact.
- Deferred roots are validated while lazy, then submitted in tuple order only
  after target graph-build agreement. Tail-overlap submission remains after
  acceptance and before target commit. Both uncertain paths latch the same
  loaded target/JACCL poison; later requests refuse collectives locally.

## Content hashes

SHA-256 values below identify the frozen source and test inputs used for the
static and headless-safe checks. They are not model or runtime receipts.

| Artifact | SHA-256 |
| --- | --- |
| EXO `src/exo/worker/engines/mlx/generator/kimi_k3_dspark.py` | `cce9d0ba7890d776c0c0bc467bf9e15870b6ab1757cc15e24558f1380df036c7` |
| EXO `src/exo/worker/engines/mlx/generator/generate.py` | `479dfca5bf1182aaa449d2242e0ce120ef3ccb7e2c7ccfdffd9da203a49d19c3` |
| EXO `src/exo/worker/engines/mlx/builder.py` | `f49b88d7353da58abfcc9ad50137d9f7367ef1f4e0eee61f92f13b459d45263f` |
| EXO deferred DSpark tests | `bfc7d07192cb71e57bfb20c9c03e4dcf682731a2dd1cb5f9a94deda7c4eb35a4` |
| EXO tail-overlap interaction tests | `6cabb05628f5fdf1afe4cce62239837bbf2fbe62660c5c47a4fc8b599ec6210f` |
| MLX-LM `mlx_lm/models/kimi_k3.py` | `4cdc4f3212e57ee34539420d1bdad3c8cfacb457c3c1d18537d7f873e5886ede` |
| MLX-LM compiled-decode tests | `7ba00f0d27dbbef5d1957dee46647647dc92e4cf7a21bb7261b2be6cae7b6c07` |

## Verification boundary

The following checks passed before freezing this worktree:

```text
ruff check (EXO generator + deferred/tail tests): pass
Python AST parse (EXO generator + deferred/tail tests): pass
pytest --noconftest current DSpark + tail files: 201 passed
pytest --noconftest deferred selector/poison/materialization focus: pass
pytest test_greedy_vocab_parallel.py on Metal-capable runner: 69 passed
```

The ordinary pytest invocation with the MLX test `conftest.py` cannot import
`mlx.nn` in this headless session (`metal::load_device: No Metal device
available`). The headless-safe suite uses no conftest and exercises the fake
collective/transaction contracts; it is not evidence of TP2 or GPU behavior.

## Performance and promotion decision

Performance credit is **zero**. No benchmark, quality run, live host action,
or TP2 throughput claim is attached to this freeze. The accepted control
remains `22.517032167 tok/s`; reaching 25 tok/s still requires a fresh,
parity-checked canary. Promotion is blocked until both tensor-parallel ranks
run the exact paired source/runtime selectors with no ordinary fallback,
validated 12-root attestations, preserved W4/tail receipts, and independent
quality parity.
