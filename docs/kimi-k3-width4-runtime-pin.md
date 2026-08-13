# Kimi K3 width-four runtime authentication

Date: 2026-08-13

This branch binds EXO's rank-local checkpoint loader to the sealed width-four
MLX-LM integration. It is an authentication update, not a throughput result.

The execution source contract is:

- EXO parent: `d91b6ff7ef0f666ddeadee664fd3999fcd0394b0`;
- MLX-LM: `b9d09bde034328cf2a10817e4c05cb0533fc1fc2`;
- `mlx_lm/models/kimi_k3_fused_expert.py` SHA-256:
  `3e5e45337d521bf4e8f5e4b2578ab75cffe40d4fd7af39387acfb4c4e87bc95d`;
- `mlx_lm/models/kimi_k3_width4_fused_expert.py` SHA-256:
  `5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832`;
- rank-local loader SHA-256:
  `de9a611233eb70dbde747301bb2a07bbf9feac0b07dc843c7bc7656137cba5ec`;
- rank-local manifest schema: `k3-rank-local-tp/v2`.

The loader authenticates both changed MLX-LM modules before the model opens a
distributed transaction. The other execution-module, checkpoint, model
revision, metadata, and TP-contract pins are unchanged from the parent.

Verification completed offline:

- 24/24 loader unit tests;
- 47/47 combined loader, loader-pin, and checkpoint-resolution tests;
- Ruff and `git diff --check`.

No Mac cluster host, checkpoint, JACCL collective, EXO service, or benchmark
was used. Deployment additionally requires the sealed native MLX runtime,
bilateral runtime/source preflight, mesh and ring C0, a real-checkpoint C1
load, and a protected A/B/A service bracket.
