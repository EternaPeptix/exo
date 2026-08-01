# Kimi K3 DSpark + ReplaySSM integration scaffold

This path is experimental, sequential-only, and disabled unless every opt-in
below is exact. EXO never resolves a remote model ID for the sidecar; each
target tensor-parallel rank loads the same absolute local directory.

Pinned sidecar contract:

- checkpoint: `RadixArk/Kimi-K3-DSpark`
- revision: `eb03982e58d4fb79bcfc099e902158f562e2e27b`
- local directory: `/Users/jeweled/.exo/models/RadixArk--Kimi-K3-DSpark-eb03982e58d4fb79bcfc099e902158f562e2e27b`
- `config.json` SHA256: `6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f`
- `model.safetensors`: 4,498,585,858 bytes
- `model.safetensors` SHA256: `29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495`
- target post-layer taps: `(7, 23, 51, 67, 83)` (HF
  `output_hidden_states` indices `(8, 24, 52, 68, 84)`)
- placement: replicated on every target tensor-parallel rank

For a later offline/canary launch, provide this environment configuration to
every rank:

```text
EXO_MLX_KIMI_K3_DSPARK_SPECULATIVE=1
EXO_MLX_KIMI_K3_DSPARK_CHECKPOINT=/Users/jeweled/.exo/models/RadixArk--Kimi-K3-DSpark-eb03982e58d4fb79bcfc099e902158f562e2e27b
MLX_LM_KIMI_K3_DSPARK_PROPOSER=1
MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE=1
EXO_MLX_KIMI_K3_DSPARK_ROUND_TELEMETRY=1
```

Omitting `EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH` selects the model-native
`gamma=7`, target width 8 path. A conservative canary may explicitly set
`EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH=3` (`gamma=2`); EXO warns because that
differs from the published production width. Other widths fail validation.

Every round constructs `[anchor, proposal_1, ..., proposal_gamma]`, agrees on
that block across ranks, and runs one target width-N verification. The accepted
count is the leading equality run between the proposals and target posterior
`[:-1]`. EXO then agrees on the acceptance boundary, commits `anchor + accepted`
target inputs through the ReplaySSM transaction, crops the replicated draft to
the same new start, and emits the accepted proposals plus one target posterior
token. A proposal, verification, or agreement error rolls both transactions
back before ordinary one-token target decode is used.

Do not add these values to a live runner opportunistically. Apply them only in
a separately scheduled offline/canary launch after the paired MLX-LM proposer
commit is pinned in EXO.
