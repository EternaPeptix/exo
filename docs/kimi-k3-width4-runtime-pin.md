# Kimi K3 width-four runtime authentication (current source)

Date: 2026-08-16

This branch binds EXO's rank-local checkpoint loader to the sealed width-four
MLX-LM integration. It is an authentication update, not a throughput result.

The execution source contract is:

- EXO control parent: `d30cc7b73460ff34d32545b5bc5118555c0716d5`;
- MLX-LM current source: `cf61625caf5aa6dfad5c56eb2d71a02cf4c080ea`;
- `mlx_lm/models/kimi_k3.py` SHA-256:
  `39c59837a4a5d900d40ab8b6bb84201e2feefc9aabb3687a4e623854251d9383`;
- `mlx_lm/models/kimi_k3_packed_moe_front.py` SHA-256:
  `9dd1d75ca7022cc837165f9eae670df4df36263ef7011ccd989c9f015db9a159`;
- `mlx_lm/models/kimi_k3_fused_expert.py` SHA-256:
  `584b4c94a623b583dae470a62851a8c882580dc7cb1519916a4019a631e7a3e5`;
- `mlx_lm/models/kimi_k3_width4_fused_expert.py` SHA-256:
  `5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832`;
- rank-local loader SHA-256:
  `d5e532c1c053b8cced19f5e4d80fb41f6e2b5748e5097ea1895a6b9c88477141`;
- rank-local manifest schema: `k3-rank-local-tp/v2`.

The loader authenticates the current Kimi K3 and packed-front modules before
the model opens a distributed transaction. The W4 fused-expert, tail-overlap,
DSpark, native, auxiliary, checkpoint, model-revision, metadata, and TP-contract
pins remain unchanged from d30. The W3 packed-front selector is not set by EXO
and remains default-off; this document carries source identity only.

Verification completed offline:

- 24/24 loader unit tests;
- 47/47 combined loader, loader-pin, and checkpoint-resolution tests;
- Ruff and `git diff --check`.

No Mac cluster host, checkpoint, JACCL collective, EXO service, or benchmark
was used. Deployment additionally requires the sealed native MLX runtime,
bilateral runtime/source preflight, mesh and ring C0, a real-checkpoint C1
load, and a protected A/B/A service bracket.
