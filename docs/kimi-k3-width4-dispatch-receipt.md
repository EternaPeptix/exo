# Kimi K3 width-four request dispatch receipt

Date: 2026-08-13

This branch adds a strict, default-off, promotion-safe receipt for one isolated
Kimi K3 C1 request on exact TP2 DSpark with `verify_width=4`. Disabled mode does
not import the receipt runtime, inspect or hash `libmlx.dylib`, reset counters,
or change model routing.

## Required launch contract

Every worker must start with the same externally pinned session ID and every
candidate selector exactly `1`:

```text
EXO_MLX_KIMI_K3_WIDTH4_DISPATCH_RECEIPT_LOG=1
EXO_MLX_KIMI_K3_WIDTH4_RECEIPT_SESSION_ID=<1-128 ASCII [A-Za-z0-9._:-]>
EXO_MLX_KIMI_K3_WIDTH4_LIBMLX_SHA256=<exact lowercase candidate SHA-256>
EXO_MLX_KIMI_K3_DSPARK_PACKED_AGREEMENTS=1

MLX_LM_KIMI_K3_WIDTH4_DISPATCH_RECEIPT=1
MLX_LM_KIMI_K3_FUSED_EXPERTS=1
MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE=1
MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH4_EXACT=1
MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=1
MLX_METAL_K3_AFFINE6_Q4_DISPATCH_RECEIPT=1
MLX_METAL_K3_AFFINE6_Q4_QUAD=1
```

The normal runtime preflight remains responsible for authenticating the
configured auxiliary metallib. This bridge independently authenticates the
regular, non-symlink `lib/libmlx.dylib` sibling of the imported `mlx.core`, then
binds its native getter and reset symbols through `ctypes`.

Enabled workers are deliberately one-shot. The first service request consumes
the worker before validation or reset begins, whether it later passes or fails.
A second request fails before another reset or capture. Warmup is excluded from
that one-shot state.

## Counter and terminal boundaries

The bridge resets both the transactional MLX-LM receipt and native Q4 counter:

1. after warmup has completed and all ranks cross its barrier; and
2. again, under DSpark rank agreement, immediately before the single C1
   request enters prefill.

The second reset records the MLX-LM generation and native zero baseline. The
terminal capture requires that same generation and computes native
`final - baseline`; cumulative process counts are not accepted.

No terminal JSON is logged until all of these have completed:

1. public response construction and inner decode-generator closure;
2. the optional generation callback;
3. the terminal rank barrier;
4. confidence-capture finalization, when configured;
5. strict packed-agreement attestation (disabled or unreadable is fatal in
   receipt mode); and
6. rank agreement that local receipt capture succeeded on both ranks.

Each rank then writes one grep-safe `K3_WIDTH4_DISPATCH_RECEIPT` JSON record.
The one-shot path drains Loguru's enqueued sinks before yielding the terminal
response, so a returned C1 request cannot race its own receipt record.

## Reducer contract

An external TP2 reducer should require exactly two schema-v3 terminal records,
ranks `{0,1}`, with identical:

- `session_id`, `model_id`, model fingerprint, request fingerprint, and request
  fingerprint SHA-256;
- `world_size=2` and `verify_width=4`;
- exact selector maps;
- authenticated `libmlx_sha256`;
- MLX-LM receipt generation; and
- clean path-count and terminal-record structures.

Each record must independently prove:

- `switch_glu` has all-zero counts;
- `switch_glu_reduce.dispatched > 0`;
- `attempted = supported = dispatched` and `fallback = error = 0`;
- no fallback/error reason classes or stale completions;
- every terminal record is a supported `switch_glu_reduce/dispatched` outcome
  under the exact five MLX-LM selector values;
- native baseline `0`, final greater than zero, and delta equal to
  `final - baseline`; and
- every terminal boundary boolean, including `rank_capture_agreed`, is true and
  `rank_agreement_pending` is false.

The request fingerprint is the four-word setup fingerprint already agreed by
the DSpark ranks. In enabled mode it additionally binds the external session ID
and model ID. The model fingerprint is explicitly a domain-separated SHA-256 of
the model ID, not a claim that every checkpoint byte was re-hashed here.

This receipt proves service-path attribution. It does not by itself prove
output quality, performance, JACCL correctness, or a throughput target; those
remain separate protected-bracket gates.
