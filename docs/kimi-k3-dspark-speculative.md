# Kimi K3 DSpark + ReplaySSM integration scaffold

This path is experimental, sequential-only, and disabled unless every opt-in
below is exact. EXO never resolves a remote model ID for the sidecar; each
target tensor-parallel rank loads the same absolute local directory.

Pinned sidecar contract:

- checkpoint: `RadixArk/Kimi-K3-DSpark`
- revision: `eb03982e58d4fb79bcfc099e902158f562e2e27b`
- local directory: `/Users/jeweled/.exo/models/RadixArk--Kimi-K3-DSpark-eb03982e58d4fb79bcfc099e902158f562e2e27b`
- `config.json` SHA256: `6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f`
- `model.safetensors`: 4,498,585,858 bytes
- `model.safetensors` SHA256: `29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495`
- target post-layer taps: direct MLX-LM layer ids `(7, 23, 51, 67, 83)`
- placement: replicated on every target tensor-parallel rank

For a later offline/canary launch, provide this environment configuration to
every rank:

```text
EXO_MLX_KIMI_K3_DSPARK_SPECULATIVE=1
EXO_MLX_KIMI_K3_DSPARK_CHECKPOINT=/Users/jeweled/.exo/models/RadixArk--Kimi-K3-DSpark-eb03982e58d4fb79bcfc099e902158f562e2e27b
MLX_LM_KIMI_K3_DSPARK_PROPOSER=1
MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE=1
EXO_MLX_KIMI_K3_DSPARK_ROUND_TELEMETRY=1
```

Omitting `EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH` selects the model-native
`gamma=7`, target width 8 path. A conservative canary may explicitly set
`EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH=3` (`gamma=2`); EXO warns because that
differs from the published production width. Other widths fail validation.

## Deferred width-three async boundaries

Width-three verification can opt into deferred asynchronous Metal submission
only when both ranks set both flags exactly:

```text
EXO_MLX_KIMI_K3_DEFERRED_ASYNC_WIDTH3=1
MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3=1
```

This candidate is default-off and requires
`EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH=3`. The MLX-LM target must return exactly
12 immutable hidden-state roots, each shaped `[1, 3, 7168]`, for the 93-layer
`laguna8` schedule. Missing, malformed, extra, or unexpectedly present roots
fail closed; the source-off path requires an absent or empty root payload.

Building the target graph never submits these roots. EXO binds the feature bit
into its rank-agreed load, prompt, and verification-plan contracts, validates
the complete root tuple while the graph is still lazy, and submits one root at
a time in tuple order only from `BuiltTargetVerification.materialize()`. That
method is entered only after the existing target graph-build agreement, so an
asymmetric build failure cancels cleanly without entering target TP
collectives.

That zero-deferred-submission invariant is scoped only to the new width-three
verifier. It deliberately does not alter the inherited `laguna8` `qlen=1`
ordinary-decode construction or its immediate Q1 asynchronous boundary; the
paired feature flags select identical Q1 configuration and behavior. A canary
benchmark arm is valid only when round telemetry shows no error-driven ordinary
fallback. Such a fallback rejects the arm instead of being counted as deferred
width-three performance (a planned final short max-token tail is reported
separately).

After all 12 submissions, EXO performs the normal final synchronous target
evaluation and cache validation. A successful request logs a rank-local
attestation with the enabled bit, validated/materialized round counts, exact
validated/submitted root totals, and the first/last target offsets. Source-off
controls log `enabled=0` with zero root and round counts.

The deferred selector is width-three-only and is rejected by the configuration
when the width-four receipt candidate is selected. The current d30 width-four
receipt and tail-overlap paths remain separate contracts. If tail overlap is
enabled alongside deferred W3, deferred roots submit during target
materialization before acceptance, while the next draft submits only after the
acceptance agreement and before target commit; both failure modes latch the
same loaded target/JACCL poison.

Once any deferred submission or the final synchronous evaluation fails, target
stream state is uncertain even if local cache cancellation appears to work.
EXO poisons the shared loaded target/JACCL group before attempting cleanup and
permanently rejects later speculative rounds, ordinary tails, and new requests
locally before any rank agreement or target graph. A cancellation failure that
masks the original deferred error follows the same fail-stop path. Recovery
requires restarting **both** tensor-parallel runners (not merely recreating the
request), because one peer may already be blocked in work that the other peer
did not submit. Malformed roots and other failures before submission retain the
existing rank-agreed rollback and do not poison the loaded group.

Every round constructs `[anchor, proposal_1, ..., proposal_gamma]`, agrees on
that block across ranks, and runs one target width-N verification. The accepted
count is the leading equality run between the proposals and target posterior
`[:-1]`. EXO then agrees on the acceptance boundary, commits `anchor + accepted`
target inputs through the ReplaySSM transaction, appends that target
hidden-state prefix to the replicated draft context, and emits the accepted
proposals plus one target posterior token. A proposal, verification, or
pre-commit agreement error rolls both transactions back before ordinary
one-token target decode is used. Target-commit failure or disagreement is a
distributed fail-stop because the authoritative cache may already have
advanced. Draft-commit failure or
disagreement disables speculation on every rank after returning the already
committed target result; subsequent rounds use ordinary target decode.

Do not add these values to a live runner opportunistically. Apply them only in
a separately scheduled offline/canary launch after the paired MLX-LM proposer
commit is pinned in EXO.
