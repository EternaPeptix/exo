# Kimi K3 maintenance-window canary

This is a fail-closed operator plan for the two-node Kimi K3 TP2 service.  It
does not make a change by default.  Its ordinary invocation performs only
read-only SSH probes, hashes the staged source and checkpoint contracts, and
prints a deterministic JSON plan.  No command in this runbook was executed on
either Mac while this offline package was prepared.

The plan is rooted at the default-off EXO scaffold
`a583614bb18199fb63227217521d9c880b3466c3`.  That scaffold is deliberately
not connected to ordinary generation.  Accordingly, both DSpark actions are
disabled in the checked-in inventory; no confirmation string can bypass that
block.  The factorized-prefill actions are also disabled because that is a
separate canary with a separate runtime.

## Fixed contract

The preflight accepts only this source and data set:

| Component | Required value |
| --- | --- |
| EXO DSpark scaffold | `a583614bb18199fb63227217521d9c880b3466c3` |
| Consolidated MLX-LM DSpark head | `53dbe04a0499ffb3e98ede90ff5a82f118f77f04` |
| Accepted MLX/JACCL decode runtime | `57b87fe47cfce34d6dc59d0e274d8ee36bfb9308` |
| Factorized MLX core | `152f01807c8327ac154b8ed56dd9279a6f9506e6` |
| Factorized MLX-LM wiring ancestor | `e284ef8731ee30856925d2649b7d72a285510a3c` |
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

The rank-local checkpoint manifest hashes are
`2da4db586e81a61fbea990e95fcfa14d14a0563c6cdafc6767373f5a74e12b21`
for rank 0 and
`c91c25a9439471ba115d45ae3371341954bbb0453677549c1fb6eeb2ace0d23f`
for rank 1.  A node with less than 500 GiB physical RAM or 16 GiB free in its
artifact filesystem fails preflight.

The exact host routes and intended staging paths are in
`k3_maintenance_canary.inventory.json`.  Two paths are staging contracts, not
a claim that deployment has occurred:

- `/Users/jeweled/exo-k3-dspark-a583614` must resolve to the scaffold commit;
- `/Users/jeweled/mlx-k3-accepted-57b87fe` must resolve to the accepted MLX
  commit.

If either path has not been staged, the read-only preflight must fail.  Do not
weaken the expected commit to make it pass.  Baseline and rollback actions
continue to name the existing accepted `/Users/jeweled/exo-k3` and
`/Users/jeweled/mlx-lm-k3-overlay` deployment, not the candidate source.

## Read-only preflight

Run from this repository worktree:

```text
rtk /usr/bin/python3 scripts/kimi_k3_tp2/k3_maintenance_canary.py \
  --facts-output-dir /private/tmp/k3-maintenance-facts \
  --output /private/tmp/k3-maintenance-plan.json
```

This reads source commits and clean-worktree state plus required files, hashes the full 4.5 GB DSpark
weight file plus its config and rank manifest, reads the transport JSON,
checks interface state, RAM, and free disk, and detects whether
`generate.py` actually selects the DSpark adapter.  It does not import or load
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
- `start-baseline`: launch the accepted non-speculative stack, wait for the
  topology in the placement step, and submit the explicit one-rail TP2
  placement.
- `rollback`: stop, relaunch the accepted stack, and recreate its explicit
  one-rail placement.  Steps stop on the first nonzero exit.

`start-dspark-width3`, `start-dspark-width8`, `start-factorized-off`, and
`start-factorized-on` are disabled.  Enabling one is a code-reviewed inventory
change, not an operator override.

## DSpark canary after adapter repair

Do not enable DSpark on `a583614` as checked in.  A later adapter revision must
first fix the audited MLX-LM call signatures, zero-based layer mapping,
concrete replicated-draft lifecycle, generator selection, and collective
target/draft commit agreement.  It must also reject batch, pipeline,
prefix-cache, remote-prefill, quantized-KV, vision, and stochastic sampling
paths in the initial canary.

Once those fixes and their exact new EXO pin are reviewed, retain the plan's
three-stage evidence order:

1. Run the exact target verifier at widths 2, 3, and 8.  Require bitwise
   logits, equal cache digests, the accepted completion digest, and minimum
   speedups of 1.20x, 1.45x, and 2.00x respectively.  The measured reference
   speedups were approximately 1.30x, 1.60x, and 2.22x.
2. Run conservative gamma 2 as width 3 in A-B-B-A order.  It is an explicit
   screening override, must accept at least 70%, emit at least 2.39 tokens per
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
