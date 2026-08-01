# Kimi K3 maintenance-window canary

This is a fail-closed operator plan for the two-node Kimi K3 TP2 service.  It
does not make a change by default.  Its ordinary invocation performs only
read-only SSH probes, hashes the staged source and checkpoint contracts, and
prints a deterministic JSON plan.  No command in this runbook was executed on
either Mac while this offline package was prepared.

The plan is rooted at the default-off EXO generator integration
`04d616d5dafc07dd0146f49a70cc39ee03ef3158`. It pins and source-attests the
consolidated MLX-LM candidate, but both DSpark actions remain disabled until a
real two-Mac TP2 Builder-to-`mlx_generate` hardware canary exercises reject and
full-accept rounds. No confirmation string can bypass that block. Segmented
SDPA and the new KDA decode tile remain explicitly off. The factorized-prefill
actions are also disabled because that is a separate canary with a separate
runtime.

## Fixed contract

The preflight accepts only this source and data set:

| Component | Required value |
| --- | --- |
| EXO DSpark generator candidate | `04d616d5dafc07dd0146f49a70cc39ee03ef3158` |
| Consolidated MLX-LM candidate | `bf378e33831e745715a88418a44ce20ab1075b9b` |
| Default-off MLX kernel candidate | `2cfb83040011c273377a25df8ed16def80c6646c` |
| MLX-LM `kimi_k3.py` SHA-256 | `3e283240117d298d95e33f7238cb49abc5606aafdd26f70062e841518059088b` |
| MLX-LM `kimi_k3_dspark.py` SHA-256 | `5ba010755e703f39b86aed1ad999576a18f2f93c041b502bbe3f57b197af2f01` |
| MLX-LM `gated_delta.py` SHA-256 | `44aef2791ed0cd5cfb84e31ef00cb4df3d40ae184e6c6852b7f0dba7406d2f78` |
| MLX-LM `kimi_k3_fused_expert.py` SHA-256 | `d51bf88fa603846f4ac9876faf724d273a99ca8b6643715f68df5993eb352d68` |
| MLX-LM `kimi_k3_fused_switch_glu.py` SHA-256 | `0aa226e32b992e5bb18a14b4a3ead225b6a6f531431249e1ae6b5bbb35ca29d1` |
| MLX-LM `kimi_k3_fused_down_reduce.py` SHA-256 | `2b9841394f8334e02044f2e6b418f0bc7ff41cc719a877464a311c5e8c899ee9` |
| MLX-LM `kimi_k3_packed_moe_front.py` SHA-256 | `82076bf9c0098f2fc72a0434a5482e72a6e022fc982574f05867dcce32645435` |
| Rollback-only MLX/JACCL decode runtime | `57b87fe47cfce34d6dc59d0e274d8ee36bfb9308` |
| Rollback-only factorized MLX core | `152f01807c8327ac154b8ed56dd9279a6f9506e6` |
| Factorized MLX-LM wiring ancestor | `53dbe04a0499ffb3e98ede90ff5a82f118f77f04` |
| DSpark revision | `eb03982e58d4fb79bcfc099e902158f562e2e27b` |
| DSpark model bytes | `4,498,585,858` |
| DSpark model SHA-256 | `29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495` |
| DSpark config SHA-256 | `6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f` |
| Accepted continuation SHA-256 | `c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71` |

Both ranks must expose only `en5` as active among `en3` through `en6`.  The
JACCL contract must contain exactly:

```json
[
  [null, ["rdma_en5"]],
  [["rdma_en5"], null]
]
```

The last observed rank-local checkpoint manifest hashes are
`2da4db586e81a61fbea990e95fcfa14d14a0563c6cdafc6767373f5a74e12b21`
for rank 0 and
`c91c25a9439471ba115d45ae3371341954bbb0453677549c1fb6eeb2ace0d23f`
for rank 1.  A node with less than 500 GiB physical RAM or 16 GiB free in its
artifact filesystem fails preflight.

Those hashes identify the legacy schema-v1 manifests and are retained only as
drift evidence. Candidate commit `04d616d` requires schema v2, which signs all
tokenizer/config/remote-code metadata and agrees its digest across ranks.
Before either DSpark start action can be enabled, rerun the converter to refresh
both manifests, inspect the resulting metadata inventory, and replace both
hashes in the inventory. Existing valid rank-sliced weight files are reused.

The exact host routes and intended staging paths are in
`k3_maintenance_canary.inventory.json`.  Four paths are staging contracts, not
a claim that deployment has occurred:

- `/Users/jeweled/exo-k3-dspark-runtime-04d616d` must resolve to the candidate
  EXO commit;
- `/Users/jeweled/mlx-lm-k3-maintenance-bf378e3` must resolve to the candidate
  MLX-LM commit and all seven source hashes above;
- `/Users/jeweled/.exo/mlx-overrides/2cfb830` must resolve to the default-off
  MLX kernel candidate;
- `/Users/jeweled/mlx-k3-factorized-152f018` must resolve to the rollback-only
  factorized MLX commit.

If any path has not been staged, the read-only preflight must fail.  Do not
weaken the expected commit to make it pass. Baseline and rollback actions are
explicitly rollback-only and continue to name the existing accepted
`/Users/jeweled/exo-k3-routed-up-add-pin-76a75fd` and
`/Users/jeweled/mlx-lm-k3-attnres-router-kda-wide-routed-add-95fc8ad`
deployment, not the candidate source.  The pinned one-rail addresses from the
last verified live state are `169.254.14.36` on 512S1 and
`169.254.199.64` on 512S2; any address change must produce a new inspected
plan rather than an ad-hoc command edit.

## Read-only preflight

Run from this repository worktree:

```text
rtk /usr/bin/python3 scripts/kimi_k3_tp2/k3_maintenance_canary.py \
  --facts-output-dir /private/tmp/k3-maintenance-facts \
  --output /private/tmp/k3-maintenance-plan.json
```

This reads source commits and clean-worktree state plus required files, hashes the full 4.5 GB DSpark
weight file plus its config and rank manifest, reads the transport JSON,
checks interface state, RAM, and free disk, source-attests `kimi_k3.py`,
`kimi_k3_dspark.py`, `gated_delta.py`, `kimi_k3_fused_expert.py`,
`kimi_k3_fused_switch_glu.py`, `kimi_k3_fused_down_reduce.py`, and
`kimi_k3_packed_moe_front.py`, and detects whether `generate.py` actually
selects the DSpark adapter. It does not import or load
either model, stop a process, launch a runner, place an instance, or access the
network beyond the two configured SSH routes.

The facts directory is created only when explicitly requested.  Existing
`rank0.json` or `rank1.json` files are never overwritten.  Reproduce the plan
without contacting either Mac by using the captured facts:

```text
rtk /usr/bin/python3 scripts/kimi_k3_tp2/k3_maintenance_canary.py \
  --facts-dir /private/tmp/k3-maintenance-facts \
  --output /private/tmp/k3-maintenance-plan-replay.json
```

Before any maintenance action, inspect all of these fields in the JSON:

- `preflight.pass` is `true`, except that a confirmed emergency `stop` is
  intentionally available after an ordinary attestation failure;
- both rank reports are clean and `adapter_wired_on_both_ranks` reflects the
  source actually probed;
- `pins`, `topology`, and `accepted_flags` match this runbook;
- `actions.<action>` contains the exact argv, environment, local harness hash,
  and reason that the plan digest binds;
- `action_readiness.<action>` is `true`;
- `plan_digest` and `confirmations.<action>` are copied exactly, without shell
  substitution.

Changing an action argv, environment, enabled state, reason, or required local
harness hash changes the digest and confirmation phrase.  Apply also rehashes
the local harness before the first step.  A changed harness requires an
inventory update and a newly inspected plan.

## Explicit apply protocol

Selecting an action without `--apply` is still a dry run.  A mutation requires
all four inputs in one invocation: a non-plan action, `--apply`, the complete
inspected plan digest, and that action's complete confirmation phrase.  For
example, with values manually copied from a fresh plan:

```text
rtk /usr/bin/python3 scripts/kimi_k3_tp2/k3_maintenance_canary.py \
  --action stop \
  --apply \
  --plan-digest REPLACE_WITH_ALL_64_HEX_CHARACTERS \
  --maintenance-confirm REPLACE_WITH_EXACT_STOP_CONFIRMATION
```

The invocation probes both nodes again.  If the new plan digest differs from
the inspected value, nothing executes.  The program rejects `--apply` when
`--facts-dir` is present; snapshots exist only for offline reproduction and
cannot authorize maintenance.

The enabled actions are:

- `stop`: one isolated-service stop.  This remains ready after an attestation
  failure so an operator can fail-stop a bad canary; the underlying harness
  still resolves the isolated two-node service before acting.
- `start-baseline`: rollback-only launch of the legacy accepted non-speculative
  stack and explicit one-rail TP2 placement. It is not candidate validation.
- `rollback`: stop and recreate that rollback-only stack. Steps stop on the
  first nonzero exit.

`start-dspark-width3`, `start-dspark-width8`, `start-factorized-off`, and
`start-factorized-on` are disabled.  Enabling one is a code-reviewed inventory
change, not an operator override.

## DSpark hardware canary before enablement

The candidate wires generator selection, fresh auxiliary-state prompt seeding,
transactional target verification, EOS/max-token boundaries, token provenance,
runtime telemetry, and fail-stop rank agreement. It rejects batch, pipeline,
prefix-cache, remote-prefill, quantized-KV, vision, logprobs, and stochastic or
otherwise unsupported sampling controls. Offline tests are not a substitute
for a real two-Mac TP2 run through Builder and `mlx_generate`; keep both start
actions disabled until that canary records at least one reject round and one
full-accept round with matching rank state. Retain this evidence order:

1. Run the exact target verifier at widths 2, 3, and 8.  Require bitwise
   logits, equal cache digests, the accepted completion digest, and minimum
   speedups of 1.20x, 1.45x, and 2.00x respectively.  The measured reference
   speedups were approximately 1.30x, 1.60x, and 2.22x.
2. Run conservative gamma 2 as width 3 in A-B-B-A order.  It is an explicit
   screening override, must accept at least 70%, emit at least 2.391 tokens per
   round on average, report no fallback/error/rank-disagreement rounds, and
   reach at least 17 tok/s.
3. Run native gamma 7 as width 8 in a separate A-B-B-A sequence.  It must
   accept at least 50%, emit at least 4.5 tokens per round on average, report
   no fallback/error/rank-disagreement rounds, and reach at least 17 tok/s.

Each B leg uses a 575-token prompt, 128-token greedy decode, and seed
`20260729`.  Every leg must produce the accepted completion digest.  Baseline
drift across each A-B-B-A block may not exceed 3%.  Do not mix gamma 2 and
gamma 7 samples into one promotion decision.

## Separate factorized-prefill canary

Factorized prefill is not a DSpark leg.  Keep speculation off and run its own
A-B-B-A sequence at Q=32 with contexts 512, 4096, and 8192.  The fast path may
be selected only when Q is greater than 8 and the supported short-context and
memory gates hold.  Require:

- at least one selected fast-path call in each supported on leg, zero in off
  legs, and zero fallback calls for supported cases;
- the accepted completion digest and unchanged decode behavior;
- prefill throughput at least 95% of its matched A baseline;
- decode regression no greater than 2%;
- peak-memory increase no greater than 256 MiB and zero score-memory slope per
  key token.

## Rollback boundary

Fail or roll back on the first digest mismatch, nonzero fallback/error/rank
disagreement count, acceptance or emitted-token miss, throughput miss,
baseline drift, topology change, source/checkpoint hash change, memory gate,
or selected-path telemetry mismatch.  Rollback means DSpark and factorized
flags are unset and the accepted `57b87fe` baseline is restored.  Record the
fresh plan, raw rank facts, action output, placement artifact, canary
telemetry, and rollback evidence together; never overwrite an earlier facts
snapshot.
