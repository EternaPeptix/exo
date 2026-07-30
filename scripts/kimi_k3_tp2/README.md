# Kimi K3 rank-local TP2 tools

This directory contains the source-only, text-only reproducibility tools for
running the pinned Kimi K3 2-bit UVMAX checkpoint as two rank-local MLX tensor
parallel checkpoints.

- `k3_tp_checkpoint.py` audits and converts the pinned checkpoint.
- `rank_local_loader.py` loads an already sliced checkpoint into the matching
  MLX-LM Kimi K3 model.
- `tiny_mlx_tp_equivalence.py` provides a small two-rank equivalence check.
- `tp2_benchmark.py` records a fail-closed, deterministic full-model TP2
  baseline on the authenticated four-rail JACCL topology.
- `launch_tp2_jaccl.sh` launches that baseline with explicit wired-memory and
  asynchronous-lookahead controls.
- `k3_target_verify.py` measures exact multi-token target-verification widths
  before a speculative decoder is integrated.
- `launch_k3_target_verify.sh` launches that benchmark with overridable,
  environment-derived paths.
- `launch_k3_target_verify_current.sh` is the fail-closed launcher for the
  accepted routed-up-add stack. It requires every deployment path explicitly,
  pins all accepted feature states, defaults to the live JACCL mesh transport,
  and initially checks widths 1 and 2.
- `k3_target_diagnostic.py` is a separate, non-promotional companion that
  localizes T>1 divergence across token positions, decoder layers, final
  hidden states, mixed cache state, and one subsequent T=1 rollout.
- `launch_k3_target_diagnostic.sh` launches the diagnostic at a small
  128-token prompt and width 2 by default. It does not alter or replace the
  target-verification v3 PASS/FAIL gate.
- `transport-jaccl-tp2.example.json` documents the explicit topology contract
  that both ranks must receive; it contains placeholders, not deployment
  inventory.
- `tests/` covers the converter and loader contracts without model weights.
- `THIRD_PARTY_NOTICES.md` records upstream code and model-license boundaries.

The full, sanitized procedure and exact dependency pins are in
[`../../docs/kimi_k3_tp2.md`](../../docs/kimi_k3_tp2.md).

This package intentionally excludes credentials, machine inventories,
hostfiles, generated benchmark artifacts, and model weights.
