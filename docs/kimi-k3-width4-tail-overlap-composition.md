# Kimi K3 width-four receipt and tail-overlap composition

This branch is an offline, strict-default-off composition candidate. It starts
from the hardened width-four receipt stack at EXO commit
`4205da9df81dc76895e6d1af08af72c8e09b682e` and applies the existing tail
overlap changes, in order:

1. `226355a72601b727c5a94c65915d821d6c3f81cb` — prelaunch the next draft under
   the target commit;
2. `886b450752b5a9157b19d33fe49580e65365c90b` — fail-stop an asymmetric async
   submission without entering another collective.

The source commits share base
`04fe574f993471ece70a53651e669d5ea1f6f6c0`. Both cherry-picks applied without
conflicts. The resulting delta relative to the hardened tail tip is the receipt
stack plus the previously screened width-four DSpark admission; there is no
manual reconciliation of the state machines.

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

- Focused tail-overlap, DSpark, and greedy integration tests exercise proposal
  reuse, default-off behavior, width-four composition, target-commit ordering,
  anchor mismatch, and fail-stop submission/finalization paths.
- The isolated receipt suite exercises strict selectors, native identity,
  transactional counters, terminal ordering, and rank agreement.
- Rank-local loader tests preserve the exact EXO/MLX-LM/source pins.
- Ruff lint and format checks, Python byte-compilation, and strict basedpyright
  on the three production generator modules are required before sealing this
  branch. Third-party MLX/MLX-LM imports may report missing-source warnings;
  errors are not accepted.

This is not live evidence. No cluster, model weights, network, service, quality
parity, memory, prefill, decode, or end-to-end throughput was exercised while
creating this composition. Promotion still requires a fresh, quiescent two-rank
request with an accepted bilateral receipt and the established A/B/A bracket.
