# Kimi K3 width-four receipt and tail-overlap composition

This branch is an offline, strict-default-off composition candidate. It starts
from the authenticated width-four C1 source at EXO commit
`c122d175512d91e01a67b4bf58c6f1803a5eef25` and replays the promoted
tail-overlap changes, in order:

1. `691f55fb4b126e0655525e7f7f0eb9f38ad04350` — prelaunch the next draft under
   the target commit;
2. `80ea94ec479f3a535d9d8b30352186db32ccccc3` — fail-stop an asymmetric async
   submission without entering another collective;
3. `1b2f75fc91a3e9548fb4090d1538144cc20e6504` — seal the composition tests and
   provenance document.

The authenticated C1 source and promoted tail-overlap tip share merge base
`4205da9df81dc76895e6d1af08af72c8e09b682e`. All three cherry-picks applied
without conflicts. The resulting delta relative to `c122d175` is confined to
the DSpark generator, its focused integration tests, and this document; there
is no manual reconciliation of the state machines.

The authenticated loader/runtime chain remains byte-identical to `c122d175`:
the rank-local loader SHA-256 is
`9f10d5572f43c2dac34a9dfb2969118dad9d42326c36eace28d7b63a80897639`,
and it continues to pin MLX-LM commit
`591e11093b55b3b03f7cbc5018cd3b7d47abba4f` and the exact runtime source
digests documented in `kimi-k3-width4-runtime-pin.md`.

## Selector boundary

Tail overlap remains independently disabled unless
`EXO_MLX_KIMI_K3_DSPARK_TAIL_OVERLAP=1`. Its parser default is exactly `0`.
The width-four receipt remains independently disabled unless
`EXO_MLX_KIMI_K3_WIDTH4_DISPATCH_RECEIPT_LOG=1`, together with all of the exact
MLX-LM and native receipt selectors documented in
`kimi-k3-width4-dispatch-receipt.md`.

Therefore the ordinary/default path does not prelaunch a draft, collect a
width-four receipt, hash the native runtime, or reset receipt counters. Enabling
one candidate does not implicitly enable the other.

## Ordering and failure contract

On an eligible width-four round, tail overlap preserves the hardened order:

1. ranks agree on proposal and acceptance;
2. the draft cache builds its accepted prefix;
3. ranks build and agree on the next draft graph;
4. every rank submits the lazy context append and next proposal graph;
5. the target cache commits the accepted posterior;
6. the following round reuses the prelaunched draft only for the rank-agreed
   anchor.

If async submission or lazy draft finalization fails after any rank may have
submitted TP work, the engine poisons the request locally, cancels local draft
state, and raises `DSparkCollectivePoisonError` before target commit or any
later agreement. It never attempts ordinary fallback on a possibly asymmetric
collective stream. The terminal width-four receipt is captured only after the
public response, callback, terminal barrier, capture finalization, and packed
attestation have succeeded. Thus a poisoned request cannot emit a successful
terminal receipt.

## Offline evidence

- 273 focused tail-overlap, DSpark, greedy, rank-local loader, and loader-pin
  tests passed. They exercise proposal reuse, default-off behavior, width-four
  composition, target-commit ordering, anchor mismatch, fail-stop
  submission/finalization paths, and the exact EXO/MLX-LM/source pins.
- 43 isolated receipt tests passed, exercising strict selectors, native
  identity, transactional counters, terminal ordering, and rank agreement.
- 144 K3 TP2 and loader-pin tests passed.
- Ruff lint and format checks and Python byte-compilation passed. Strict
  basedpyright on the three production generator modules reported zero errors;
  its six warnings were only the expected missing-source warnings for
  third-party MLX/MLX-LM imports.

This is not live evidence. No cluster, model weights, network, service, quality
parity, memory, prefill, decode, or end-to-end throughput was exercised while
creating this composition. Promotion still requires a fresh, quiescent two-rank
request with an accepted bilateral receipt and the established A/B/A bracket.
