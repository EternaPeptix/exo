# Kimi K3 C1 frontier telemetry v4 handoff

This is the exact source-to-executor contract for the default-off, same-process
sixteen-request telemetry interface. The source lineage is the C1 EXO parent
`20c747ec35c0b9858635823721892cc10f915ae2` on branch
`codex/k3-c1-frontier-telemetry-v3`. This package contains offline source and
tests only; it is not live-run evidence and it does not authorize deployment.

## Version boundary

- Default-off and startup receipts retain
  `kimi-k3-w3-composition-receipt/v2` exactly.
- Enabled API requests publish
  `kimi-k3-w3-composition-receipt/v4`.
- `kimi-k3-w3-composition-receipt/v3` is already owned by the separate C1
  bookkeeping-timing bundle. This telemetry package never emits or redefines
  v3.
- A future composition of bookkeeping v3 and telemetry v4 must use a reviewed
  v5 strict superset. It must not publish timing fields under the v4 URI.

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

The v4 response marker contains the complete strict receipt-v2 C1 core with
only its external `schema` and `receipt_schema_version` advanced to v4. It adds
these exact scalar fields:

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

Each four-word digest is the unmasked SHA-256 split into four unsigned 64-bit
big-endian words. The rank-local telemetry digest is the SHA-256 of the exact
request-file bytes. The process-completion digest is all zero words for API
requests 1 through 15 and the exact completion-file SHA-256 on request 16.
The marker contains no RSS or Metal byte values: it is numeric/hash-only and
joins the private files without exposing their contents in the response.

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

Neither evidence file retains a prompt, completion, raw request/response body,
token stream, environment map, hostname, or filesystem path.

## Join gate

The source is ready for offline integration, but a live frontier run must stay
ineligible until the executor and canary both:

1. send the two exact headers on all sixteen API requests without changing the
   body bytes;
2. validate the strict v4 response extension and reconstruct every four-word
   digest without truncation;
3. read only the file named by the authenticated rank-local digest, validate
   its exact key set and algebra, and apply the projection crosswalk above;
4. require zero process-completion digests for requests 1-15 and two valid
   completion digests on request 16; and
5. keep bookkeeping timing under receipt v3, or advance an explicitly reviewed
   combined interface to v5.
