# Kimi K3 rank-local TP2 tools

This directory contains the source-only, text-only reproducibility tools for
running the pinned Kimi K3 2-bit UVMAX checkpoint as two rank-local MLX tensor
parallel checkpoints.

- `k3_tp_checkpoint.py` audits and converts the pinned checkpoint.
- `rank_local_loader.py` loads an already sliced checkpoint into the matching
  MLX-LM Kimi K3 model.
- `tiny_mlx_tp_equivalence.py` provides a small two-rank equivalence check.
- `tests/` covers the converter and loader contracts without model weights.
- `THIRD_PARTY_NOTICES.md` records upstream code and model-license boundaries.

The full, sanitized procedure and exact dependency pins are in
[`../../docs/kimi_k3_tp2.md`](../../docs/kimi_k3_tp2.md).

This package intentionally excludes machine inventories, launch daemons,
hostfiles, benchmark artifacts, and model weights.
