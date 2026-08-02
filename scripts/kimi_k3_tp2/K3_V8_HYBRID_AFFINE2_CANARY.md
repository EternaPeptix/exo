# Kimi K3 v8 hybrid and affine2 isolated canary

This recipe stages the combined candidate without modifying the retained EXO,
MLX, MLX-LM, model, or checkpoint trees. Build products, virtual environments,
logs, and coordinator state must live under a new canary root.

## Exact source candidates

| Repository | Branch | Commit |
| --- | --- | --- |
| MLX | `experiment/k3-v8-true-hybrid-affine2-v1` | `21d518384e845c1adf5c4be43d204152572fcce2` |
| MLX-LM | `experiment/k3-v8-affine2-elision-selective-down-v1` | `738df4ebd667d763e49a003cb0287fad12e759b3` |
| EXO | `experiment/k3-v8-affine2-elision-loader-pin-v1` | Use the audited branch tip reported with the canary |

The MLX and MLX-LM branches are rooted at the published v8 tips `2cfb8304`
and `bf378e3`; the EXO branch is rooted at `a45b8a5`. Every new optimization
is default-off.

## Isolated staging

Use a new directory such as
`/private/tmp/k3-v8-hybrid-affine2-canary`. Do not install into or replace the
retained runtime at `/Users/jeweled/.exo/mlx-overrides/57b87fe`.

1. Build MLX from the exact candidate commit into the canary root with
   `MLX_METAL_JIT=ON`. Build and run the standalone JACCL policy test with
   `JACCL_BUILD_TESTS=ON`.
2. Install the exact MLX-LM candidate into an isolated virtual environment.
3. Run EXO from the exact candidate worktree and point
   `EXO_MLX_RANK_LOCAL_LOADER` at that worktree's loader. Keep the existing
   rank-local checkpoint directories read-only.
4. Give the canary its own coordinator port, PID files, and logs. Verify the
   source pins on both ranks before model construction.

## Rank-consistent profiles

Keep the existing validated v8 flags identical between control and candidate.
Set the following values explicitly on both tensor ranks.

Control:

```bash
export MLX_JACCL_RING=1
export MLX_JACCL_TP2_HYBRID=0
export MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=0
export MLX_LM_KIMI_K3_ELIDE_AFFINE2_BIAS=0
```

Candidate:

```bash
export MLX_JACCL_RING=1
export MLX_JACCL_TP2_HYBRID=1
export MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=1
export MLX_LM_KIMI_K3_ELIDE_AFFINE2_BIAS=1
```

JACCL rejects a hybrid request mismatch before constructing the dedicated mesh
QP. MLX-LM rejects ambiguous affine2 values, an elision request without derive,
an unavailable affine2 core kernel, and any nonexact bias bank.

## Gates

Run the following in A-B-A order with identical topology, prompts, sampling,
and existing v8 flags:

1. The model-free TP2 trace: 186 BF16 all-sums using the K3 14,336- and
   21,504-byte reductions. Require exact digests and no more than 2% p50 or p95
   regression versus stock mesh.
2. Boundary and large-message collectives at 65,536 bytes, 1 MiB, 8 MiB, and
   32 MiB. Below 1 MiB must remain on stock mesh; 1 MiB and above must retain
   stock multi-rail ring behavior.
3. K3 deterministic decode at the short smoke prompt and canonical 575-token
   prompt, followed by 2K and 8K prompt-prefill checks. Require identical token
   and output digests.
4. Record per-rank peak active and wired memory before load, after strict load,
   after raw-reference release, and after first evaluation. Confirm that a
   forced strict-load failure retains the raw dictionary and skips cache clear
   and evaluation.

Stop the isolated processes after the gate. Rollback is simply returning to the
retained runtime and its original environment; this recipe makes no persistent
network or fleet-runtime changes.
