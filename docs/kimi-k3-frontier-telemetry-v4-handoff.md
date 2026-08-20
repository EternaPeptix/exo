# Kimi K3 C1 frontier telemetry v4 sidecar / v5 receipt handoff

This is the exact source-to-executor contract for the default-off, same-process
sixteen-request telemetry interface. The source lineage is the C1 EXO parent
`20c747ec35c0b9858635823721892cc10f915ae2` on branch
`codex/k3-c1-frontier-telemetry-v3`. This package contains offline source and
tests only; it is not live-run evidence and it does not authorize deployment.

## Version boundary

- Default-off and startup receipts retain
  `kimi-k3-w3-composition-receipt/v2` exactly.
- Enabled API requests publish the reviewed combined
  `kimi-k3-w3-composition-receipt/v5`. The rank-local request and
  process-complete sidecars remain schema v4.
- `kimi-k3-w3-composition-receipt/v3` is already owned by the separate C1
  bookkeeping-timing bundle. This telemetry package never emits or redefines
  v3.
- `kimi-k3-w3-composition-receipt/v4` is the hash-only telemetry predecessor.
  This package does not overload v4 with the ordered schedule or timing.
- V5 is the strict reviewed superset that joins the v4 hash evidence with the
  ordered consumed/width schedule and integer timing required by the public
  executor contract.

## Request controls

The executor sends the unchanged request body to the bench chat-completions
endpoint with exactly one value for each header:

| Header | Exact contract |
| --- | --- |
| `X-EXO-K3-Frontier-Request-Index` | canonical decimal `1` through `16`; no leading zero |
| `X-EXO-K3-Frontier-Request-Nonce-SHA256` | unique lowercase hexadecimal SHA-256 |

The enabled process also requires these rank-local launch values:

| Environment name | Exact contract |
| --- | --- |
| `EXO_MLX_KIMI_K3_FRONTIER_TELEMETRY_V4` | literal `1` |
| `EXO_MLX_KIMI_K3_FRONTIER_TELEMETRY_DIR` | absolute, process-owned, non-symlink directory with exact mode `0700` |
| `EXO_MLX_KIMI_K3_FRONTIER_TELEMETRY_SESSION_SHA256` | lowercase hexadecimal SHA-256 shared by the intended session |

The API request indices are strictly monotonic in one process. With the one
startup receipt at request sequence 1, the executor must observe API receipt
sequences 2 through 17. A duplicate nonce, duplicate or skipped index, overlap,
retry after failure, or seventeenth API request poisons the telemetry sequence.

## Response marker

The v5 response marker contains the complete strict receipt-v2 C1 core with
only its external `schema` and `receipt_schema_version` advanced to v5. It
retains these exact v4 hash-join scalar fields:

- `frontier_telemetry_schema_version`
- `frontier_request_limit`
- `frontier_request_index`
- `frontier_reset_generation`
- `frontier_request_nonce_digest_word_0` through
  `frontier_request_nonce_digest_word_3`
- `frontier_receipt_v2_core_digest_word_0` through
  `frontier_receipt_v2_core_digest_word_3`
- `frontier_telemetry_file_digest_rank0_word_0` through
  `frontier_telemetry_file_digest_rank0_word_3`
- `frontier_telemetry_file_digest_rank1_word_0` through
  `frontier_telemetry_file_digest_rank1_word_3`
- `frontier_process_complete_digest_rank0_word_0` through
  `frontier_process_complete_digest_rank0_word_3`
- `frontier_process_complete_digest_rank1_word_0` through
  `frontier_process_complete_digest_rank1_word_3`

It then adds exactly:

- `frontier_round_schedule`, an ordered nonempty list of strict rows containing
  only integer `ordinal`, `consumed`, and `width`; ordinals start at one and
  are contiguous, width is exactly zero or two, and
  `0 <= consumed <= width`. The model boundary joins row count, width counts,
  proposed/accepted token sums, full/partial accept counts, and
  `emitted_tokens == row_count + sum(consumed)` back to the inherited v2 core;
- `frontier_target_commit_ns` and `frontier_prelaunch_ns`, the nonnegative
  integer rank-zero aggregates; and
- `frontier_target_commit_ns_rank1` and `frontier_prelaunch_ns_rank1`, the
  corresponding nonnegative integer rank-one aggregates.

Each four-word digest is the unmasked SHA-256 split into four unsigned 64-bit
big-endian words. The rank-local telemetry digest is the SHA-256 of the exact
request-file bytes. The process-completion digest is all zero words for API
requests 1 through 15 and the exact completion-file SHA-256 on request 16.
The marker contains no RSS or Metal byte values: those numeric measurements
remain only in the private v4 sidecars. The public marker joins those files by
hash and exposes only the bounded schedule and integer timing additions. It
never contains prompt, completion, token text, environment, hostname, or path
data.

## Rank-local request evidence

Each rank writes one canonical-JSON file named
`request-{request_index:02d}-rank{rank}.json` with exact mode `0600` and schema
`kimi-k3-frontier-request-telemetry/v4`. Its exact keys are:

```text
schema
schema_version
session_sha256
request_index
request_nonce_sha256
rank
world_size
pid
reset_generation
reset_fence_sha256
receipt_v2_core_sha256
receipt_request_sequence
receipt_request_token
replayssm_telemetry_revision_before
replayssm_telemetry_revision_after
replayssm_telemetry_revision_delta
replayssm_attempted_prepares_delta
replayssm_batched_prepares_delta
replayssm_batched_commits_delta
replayssm_identity_prepares_delta
replayssm_identity_commits_delta
replayssm_fallback_prepares_delta
replayssm_fallback_commits_delta
replayssm_batched_errors_delta
replayssm_identity_errors_delta
replayssm_layers_batched_delta
replayssm_layers_identity_committed_delta
process_rss_baseline_bytes
process_rss_final_bytes
process_rss_final_delta_bytes
process_rss_observed_peak_bytes
process_rss_observed_peak_delta_bytes
process_rss_lifetime_highwater_baseline_bytes
process_rss_lifetime_highwater_after_bytes
process_rss_lifetime_highwater_delta_bytes
rss_sample_interval_ns
rss_sample_count
metal_active_baseline_bytes
metal_active_final_bytes
metal_active_final_delta_bytes
metal_active_peak_bytes
metal_active_peak_delta_bytes
metal_cache_baseline_bytes
metal_cache_final_bytes
metal_cache_final_delta_bytes
metal_residency_baseline_bytes
metal_residency_final_bytes
metal_residency_final_delta_bytes
metal_residency_observed_peak_bytes
metal_residency_observed_peak_delta_bytes
metal_observation_count
rss_peak_semantics
rss_lifetime_highwater_semantics
metal_active_peak_semantics
metal_residency_peak_semantics
evidence_sha256
```

`evidence_sha256` authenticates the canonical object without that field.
`receipt_v2_core_sha256` authenticates the canonical object formed from schema
`kimi-k3-w3-composition-receipt/v2` and the complete agreed v2 marker payload.
The request marker authenticates the SHA-256 of the complete file bytes, which
is deliberately a different digest boundary.

The exact semantics strings are:

| Evidence key | Exact value |
| --- | --- |
| `rss_peak_semantics` | `bounded-10ms-request-sampler` |
| `rss_lifetime_highwater_semantics` | `process-lifetime-authoritative` |
| `metal_active_peak_semantics` | `mlx-reset-scoped-authoritative` |
| `metal_residency_peak_semantics` | `bounded-lifecycle-observed` |

The RSS observed peak is sampled every 10 ms by a bounded request-scoped
thread. The lifetime high-water fields come from `getrusage` and are named
separately. MLX active peak is the authoritative allocator counter following
the existing request reset. Residency is active plus cache and is observed at
bounded lifecycle points because MLX exposes no residency high-water counter.

## Executor projection crosswalk

The current executor projection vocabulary maps mechanically from each
rank-local v4 file:

| Executor field | Rank-local v4 field |
| --- | --- |
| `rss_baseline_bytes` | `process_rss_baseline_bytes` |
| `rss_sampled_peak_bytes` | `process_rss_observed_peak_bytes` |
| `rss_final_bytes` | `process_rss_final_bytes` |
| `rss_delta_bytes` | `process_rss_final_delta_bytes` |
| `metal_residency_baseline_bytes` | `metal_residency_baseline_bytes` |
| `metal_residency_sampled_peak_bytes` | `metal_residency_observed_peak_bytes` |
| `metal_residency_final_bytes` | `metal_residency_final_bytes` |
| `metal_residency_delta_bytes` | `metal_residency_final_delta_bytes` |
| `rss_sample_count` | `rss_sample_count` |
| `metal_observation_count` | `metal_observation_count` |

For the executor's reset projection,
`reset_generation_after = reset_generation`,
`reset_generation_before = reset_generation - 1`, and
`reset_generation_delta = 1`. The executor must additionally require
`frontier_reset_generation == request_index` in the marker and require the
rank-local `reset_fence_sha256` to match the file it authenticated. These are
derived projection names, not additional v4 marker or sidecar fields.

## Sixteen-request completion and file safety

Request 16 additionally writes `process-complete-rank{rank}.json` with schema
`kimi-k3-frontier-process-telemetry/v4` and exact keys:

```text
schema
schema_version
session_sha256
rank
world_size
pid
request_count
requests
request_chain_sha256
evidence_sha256
```

`requests` is an ordered sixteen-row list. Each row contains exactly
`request_index`, `request_nonce_sha256`, `request_file_sha256`,
`receipt_request_sequence`, and `receipt_request_token`.

Every write uses `O_EXCL`, exact mode `0600`, file and directory `fsync`, and a
readback through the same authenticated directory descriptor used for
creation. Request 1 requires an empty directory. Request N requires exactly the
N-1 files already authenticated by this live process, with matching inode,
owner, link count, mode, and byte SHA-256. The writer rechecks the exact
inventory immediately before and after creation, and request 16 requires the
exact sixteen-file inventory before publishing the completion file.

Creation records the exact device, inode, regular-file type, owner, mode, link
count, and size. Reopen must reproduce that complete identity. After a
post-create failure, cleanup keeps the original `O_EXCL` descriptor open,
authenticates the name through the already-open directory descriptor, fsyncs
the directory, unlinks, requires the original descriptor's link count to
transition from one to zero, and fsyncs the directory again. A name that no
longer resolves to the created inode is never removed; cleanup fails, the
sequence is poisoned, and exact inventory prevents reuse.

Neither evidence file retains a prompt, completion, raw request/response body,
token stream, environment map, hostname, or filesystem path.

## Bilateral lifecycle and timing boundary

Every failure-prone rank-local admission and setup stage is enclosed by a
fixed-order two-rank agreement. This includes source/native identity setup,
telemetry claim, model health, receipt begin, reset baseline capture, and
request setup. Any local or peer failure invokes cleanup and globally poisons
both telemetry and the DSpark materialization before either rank can proceed.
Cross-rank admission compares only authenticated selector, launch, source, and
native-library content digests. Device, inode, owner, mode, and timestamp
identity remains mandatory for startup-to-API continuity on each rank, but is
never compared between two different host filesystems.

Rank-local telemetry finalization, sampling, `O_EXCL` writes, readback, and
`fsync` are likewise one agreed stage before either file digest collective.
The receipt then remains explicitly `finalized-unpublished` while the terminal
response is yielded to the runner. The runner must parse it and complete its
outer rank-agreed, nonblocking event publication before it resumes the
generator. Only that resumption may complete the process-local lifecycle,
commit the telemetry state, and emit the rank-zero marker as the final side
effect, with no peer collective after the marker write. Parser, channel, peer,
abandonment, lifecycle, or commit failure therefore cannot leave a marker.
If final marker logging itself reports failure after bytes become visible,
those bytes remain an irrevocable record that every earlier bilateral gate
succeeded; the executor must still reject any subsequent request error rather
than treating a marker alone as benchmark success. This also applies to
request 16 after both process-complete files exist.

Authoritative decode elapsed time and effective generation TPS are captured
once at the terminal compute boundary, after the required terminal barrier but
before telemetry finalization, digest collectives, validation, or filesystem
durability work. Receipt I/O can therefore never inflate authoritative decode
time or reduce effective TPS. Per-round target-commit and prelaunch durations
are converted to bounded integer nanoseconds as each observed round closes and
are gathered without recapturing the decode clock.

## EXO source identity expansion

The EXO source digest now binds the exact API and forwarding path, including:

```text
src/exo/master/main.py
src/exo/shared/types/commands.py
src/exo/shared/types/tasks.py
src/exo/shared/types/text_generation.py
src/exo/worker/main.py
src/exo/worker/plan.py
src/exo/worker/runner/runner.py
src/exo/worker/runner/llm_inference/batch_generator.py
```

It also retains the generator, DSpark, telemetry, builder, API adapter/schema,
rank-local checkpoint, and configured rank-local loader identities. Focused
drift tests require every listed master/commands/tasks/worker forwarding file
to alter the digest independently.

## Join gate

The source is ready for offline integration, but a live frontier run must stay
ineligible until the executor and canary both:

1. send the two exact headers on all sixteen API requests without changing the
   body bytes;
2. validate the strict v5 response extension, exact ordered schedule, four
   integer timing fields, and reconstruct every four-word digest without
   truncation; reject v4/v5 schema substitution in both directions;
3. read only the file named by the authenticated rank-local digest, validate
   its exact key set and algebra, and apply the projection crosswalk above;
4. require zero process-completion digests for requests 1-15 and two valid
   completion digests on request 16; and
5. update the executor and canary source locks from their current v4 marker to
   this exact v5 contract before any live run. This source package does not by
   itself modify or authorize that separate executor package.
