# Kimi K3 W3 composition diagnostic receipt freeze

This worktree is an offline, default-off source-and-test freeze for the strict
`kimi-k3-w3-composition-receipt/v1` diagnostic. It contains no live inference,
host result, benchmark, or timing evidence. A positive receipt is diagnostic
evidence only: it does not grant performance credit or promotion eligibility.

## Lineage and authenticated runtime

- EXO branch: `codex/exo-k3-w3-composition-receipt-v1`
- frozen combined EXO parent: `893ed966ecc3260a238f42d65cacce7b660bfab6`
- MLX-LM source commit: `400134d0dd53ffc8e80ca7ae78ab6ecc4687145a`
- MLX-LM documentation HEAD: `4eee0a094555d5718d820ec63424c61ca9cfe201`
- MLX-LM tree: `6dd753d318f2c12a5253ba2a1ded4a2658e10d04`
- native counter implementation: `b2948c83ef59b28afb6c40dc0fb79abcc30ab491`
- native counter documentation HEAD: `755fe61a66948c106d6eb230cf26c578ac9fc3c9`

The loader and receipt require these exact source/image identities:

- `models/kimi_k3.py`:
  `7aff86896e70baf2b986808341b5f292f8c8ecd55d17a9693813cbe25b356c97`
- `kimi_k3_packed_moe_front.py`:
  `9be2130bc3afd754d4369882aea571bdc546d62ca1d67b4ba7658641f42e5510`
- `kimi_k3_w3_prework.py`:
  `f8fcba947dc0e52c335522bd0d152b57679818a64db9edc6e6ba8f58834cb254`
- EXO rank-local loader bridge:
  `f7059a2d45cffedf7614174e44a326f4f14372951231eac4159094eea8b3e67c`
- sealed counter-bearing `libmlx.dylib`:
  `91f742bfa20f3559c2fb85e6b3b5aad7a5b6264e1158d2cfc1a089a35d1b18cd`

The native image is opened without following symlinks, authenticated from a
retained file descriptor, required to be the already-loaded MLX image, and
rechecked at begin, finish, and publication. EXO and MLX source manifests and
the rank-common launch contract are independently hashed and rechecked at the
same boundaries. The startup and API phases must retain the same native inode
identity as well as the same content digest.

## Preregistered two-arm contract

The protected diagnostic admits exactly two arms:

| Arm | Packed W3 | KDA prework | Deferred W3 | Native affine8 Q3 | Tail overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0 | 0 | 0 | 1 | 1 |
| candidate | 1 | 1 | 1 | 1 | 1 |

Native affine8 Q3 dispatch receipt collection is also `1` on both arms. The
receipt rejects every mixed subset and any native-Q3/tail-overlap drift. This
is an anti-multiple-comparisons boundary, not a limitation of the underlying
default-off source: deferred-only, KDA-only, and other native-zero compositions
remain valid for source/unit testing outside this protected receipt.

The launch digest covers an explicit allowlist of rank-common behavioral
selectors and rejects unknown experiment keys in the protected namespaces.
Rank-local checkpoint/loader paths, coordinator/ring addresses, roles, secrets,
and credentials are excluded; deployment attestation must bind those
separately. `MLX_LM_KIMI_K3_PROJECTED_KV_CACHE_MAX_TOKENS` is required to be
the canonical decimal `32768`.

The allowlist also admits and authenticates every rank-common performance
selector used by the accepted 22.517 tok/s width-three control, including the
width-three fused-expert, MOK-overlap, ReplaySSM-commit, prefill-route, and
native affine companion selectors. They remain symmetric across control and
candidate and are bound by the externally pinned launch-contract digest; the
only protected arm deltas are still packed W3, KDA prework, and deferred W3.
The historical launcher omitted the inactive Q4 auxiliary-metallib variable,
and the receipt binds either absence or an explicit value rather than silently
dropping it from the launch map.

## Receipt proof boundary

Receipt admission is process-local, locked, and one-shot: a successful startup
receipt must pass the outer rank-agreed warmup validation before one API phase
is allowed. Failed, aborted, duplicate, concurrent, API-before-startup, and
identity-mismatched phases fail closed. Marker publication and phase completion
are rank-agreed side effects. The terminal decode close and Metal barrier occur
before MLX/native final snapshots.

The final numeric-only marker binds:

- exact MLX raw schema and strict scalar types for 92 sparse and 69 KDA layers;
- selector, request, setup, launch, EXO source, MLX source, and native identity;
- phase-local affine8-Q3 total/N4480/N6144/N10624 dispatch deltas;
- proposal, acceptance, target-committed output, visible output, and cache
  schedule digests;
- rank-agreed decode-start cache/anchor state, per-round cache continuity, and
  the fixed projected-cache capacity;
- 12 ordered, distinct deferred roots per eligible full round, exact initial
  and final offsets, and materialization attestation;
- causal tail-prelaunch submission/use/discard adjacency, context offsets, and
  one-way anchor fingerprints;
- fallback, poison, stale, invalidation, duplicate, abort, and cleanup state.

Raw prompt text and raw token IDs are never serialized. Token values are hashed
immediately into domain-separated request-local chains; only counts and digest
words enter the marker. The validator derives its algebra from observed full,
tail, and prefill geometry. It does not hardcode the canary schedule. The
focused fixture alone checks the canonical 45-full-round totals of 4,140 packed
W3 hits, 3,105 KDA successes, and 540 deferred roots.

Default-off requests retain no receipt object or marker field. The optional API
field is omitted when `None`; causal observer calls, receipt-only distinct-root
validation, and per-root receipt callbacks are gated before the hot loop. This
source freeze nevertheless makes no byte-identical timing claim. Any future
performance decision requires telemetry-off A/B/A on the same source.

## External controller obligations

The receipt does not replace deployment controls. A protected run must also:

1. pin the exact expected launch-contract SHA and all rank-specific deployment,
   topology, core, metallib, JACCL, checkpoint, and process identities;
2. keep `EXO_NO_BATCH=1` and a sequential process request ledger so unrelated
   Q3 work cannot enter the process-global reset/snapshot interval;
3. retain both rank logs, the completed response/lifecycle result, and the
   exact agreed marker bytes; and
4. reject stray/incomplete markers and any marker that lacks the matching
   successful startup-to-API lifecycle.

The Pydantic schema validates structure; it is not a cryptographic
authentication mechanism for a hand-edited marker. The controller must bind
the original bilateral evidence bytes.

## Offline verification

The final source was formatted and linted with Ruff, then verified in the
headless Python 3.13 environment without model loading or Metal execution:

```text
13 passed  # hermetic composition receipt scenarios
234 passed # existing DSpark, tail-overlap, and loader boundaries
18 passed  # chat-stream and Responses API serialization boundaries
```

The receipt scenarios cover default-off/orphan flags, exact selector admission,
startup/API lifecycle and concurrency, strict MLX schema types, q1/q3 and
noncontract prefill, control/candidate/subset rejection, canonical schedule
algebra, rank mismatch, abort/early close, poison/fallback, native counter and
image identity, begin/finish/publication mutation, root/cache/tail/anchor event
ordering, terminal visible-token binding, and projected-cache overflow.
