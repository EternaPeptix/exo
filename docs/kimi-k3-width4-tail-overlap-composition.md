# Kimi K3 width-four receipt and tail-overlap composition

This branch is an offline, strict-default-off composition candidate based on
EXO control `d30cc7b73460ff34d32545b5bc5118555c0716d5`. It retains the
authenticated width-four C1 source and promoted tail-overlap behavior already
present at that control; this port changes only current MLX-LM source identity
and the rank-local loader pin.

1. `691f55fb4b126e0655525e7f7f0eb9f38ad04350` — prelaunch the next draft under
   the target commit;
2. `80ea94ec479f3a535d9d8b30352186db32ccccc3` — fail-stop an asymmetric async
   submission without entering another collective;
3. `1b2f75fc91a3e9548fb4090d1538144cc20e6504` — seal the composition tests and
   provenance document.

The authenticated loader/runtime chain now pins current MLX-LM source
`cf61625caf5aa6dfad5c56eb2d71a02cf4c080ea` and remains byte-stable across all
W4/tail-overlap/DSpark/native modules. The rank-local loader SHA-256 is
`d5e532c1c053b8cced19f5e4d80fb41f6e2b5748e5097ea1895a6b9c88477141`,
and the exact runtime source digests are documented in
`kimi-k3-width4-runtime-pin.md`. The W3 packed-front selector is not enabled
globally and is not part of this tail-overlap composition.

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
