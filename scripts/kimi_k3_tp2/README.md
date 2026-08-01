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
  target-verification v5 PASS/FAIL gate.
- `launch_k3_target_diagnostic_current.sh` runs that width-2 diagnostic with
  the accepted fused stack and the same fail-closed mesh/ring attestation as
  the current target verifier.
- `k3_kda_stage_localizer.py` compares real-checkpoint layer-0 KDA
  intermediates and cache state between one width-2 call and two sequential
  width-1 calls without weakening the target-verification gate.
- `launch_k3_kda_stage_localizer_current.sh` runs that localizer on the pinned
  accepted stack and independently attests both ranks and the selected JACCL
  transport.
- `k3_decoder_layer_stage_localizer.py` extends the real-checkpoint width-2
  comparison across layer 0's preparation, KDA cache, AttnRes/RMS, MoE, and
  residual boundaries. Controlled shared-input probes distinguish upstream
  propagation, the verifier's direct non-history KDA path, and the
  transaction's history-producing KDA path; complete-model layer-0 endpoint
  reproduction is mandatory before an artifact can be written.
- `launch_k3_decoder_layer_stage_localizer_current.sh` runs that checker only
  on the accepted v6 JACCL mesh. It rejects the superseded speculative-KDA
  rewrite and defaults the exact direct-wide short-convolution candidate off;
  set `K3_DECODER_LAYER_LOCALIZER_EXACT_WIDE_SHORT_CONV=1` for the explicit
  candidate run. The selector is validated as exactly `0` or `1` and is
  attested in the artifact.
- `k3_maintenance_canary.py` is the fail-closed, default-read-only two-rank
  maintenance planner pinned to the offline DSpark scaffold. Its local
  maintenance inventory binds exact staging paths, ordered action argv and
  environment, and the existing harness hash into an inspected plan digest;
  all candidate start actions remain disabled until their adapter/runtime
  gates are implemented. See `K3_MAINTENANCE_CANARY.md`.
- `transport-jaccl-tp2.example.json` documents the explicit topology contract
  that both ranks must receive; it contains placeholders, not deployment
  inventory.
- `tests/` covers the converter and loader contracts without model weights.
- `THIRD_PARTY_NOTICES.md` records upstream code and model-license boundaries.

The full, sanitized procedure and exact dependency pins are in
[`../../docs/kimi_k3_tp2.md`](../../docs/kimi_k3_tp2.md).

This package intentionally excludes credentials, hostfiles, generated
benchmark artifacts, and model weights. The maintenance-only inventory names
the two already-documented hosts and absolute staging paths but contains no
credential material.
