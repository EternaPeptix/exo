# Kimi K3 current Q3 full packed-front EXO source pin

Date: 2026-08-16

Status: **offline source-authentication update; default-off; zero performance
credit.** No EXO service, checkpoint load, host, network, quality, memory, or
throughput action is included.

## Authenticated source contract

- EXO control: `d30cc7b73460ff34d32545b5bc5118555c0716d5`
- MLX-LM source: `cf61625caf5aa6dfad5c56eb2d71a02cf4c080ea`
- `mlx_lm/models/kimi_k3.py` SHA-256:
  `39c59837a4a5d900d40ab8b6bb84201e2feefc9aabb3687a4e623854251d9383`
- `mlx_lm/models/kimi_k3_packed_moe_front.py` SHA-256:
  `9dd1d75ca7022cc837165f9eae670df4df36263ef7011ccd989c9f015db9a159`
- rank-local loader SHA-256:
  `d5e532c1c053b8cced19f5e4d80fb41f6e2b5748e5097ea1895a6b9c88477141`
- checkpoint converter pins remain d30's `7d505c2` contract and source
  inventory; the execution-time source pins above are checked separately.

The loader continues to authenticate every existing W4 fused-expert,
width-four receipt, tail-overlap, DSpark, native, auxiliary, checkpoint, and
metadata source. Only the execution-time Kimi K3 and packed-front identities
move to the current MLX-LM port.

## Selector boundary

EXO does not set `MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT` or
`MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3`. The W3 full-pack path is
therefore default-off and cannot be inferred from this source pin. Existing W4
receipt and tail-overlap selectors retain their d30 defaults and behavior.

The loader rejects the prior execution source identities from MLX-LM `591e110`
(`c20fe402...` Kimi K3 and `82076bf9...` packed-front), as well as stale receipt
packed-front identities. A manifest produced by the d30 checkpoint converter
must still carry its original converter commit/hash fields; those are distinct
from the execution-time source checks.

## Verification boundary

Focused loader/checkpoint tests and current K3 transport/DSpark suites must
pass in the sealed local runtime before any freeze. This document records
identity only; it is not a benchmark or deployment receipt. Any later W3
promotion requires a separate current-lineage EXO composition, bilateral
source/selector agreement, exact completion and quality parity, memory parity,
and an authorized A/B/A measurement.
