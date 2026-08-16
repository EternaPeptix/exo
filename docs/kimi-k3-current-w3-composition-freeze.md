# Kimi K3 current W3 composition freeze

This worktree is an offline, default-off source and test composition. It does
not promote a performance result and contains no new timing evidence.

## Lineage

- EXO control: `d30cc7b73460ff34d32545b5bc5118555c0716d5`
- deferred-W3 EXO donor: `c9eb00ef7ddf128197cb0edb8a363a5819df7ded`
  (replayed locally as `730bcd18`)
- packed-current loader donor: `4e85055e382a1117fcb5131710308334eaf47eff`
  (replayed locally as `067b29b0`)
- composed EXO source/tests freeze: `0e5ab6d30e73ff8942c95178ad7b553c5aa12404`
- combined MLX-LM HEAD: `72164e3521b8ee68d605963cfd3266a7341279e6`

The loader authenticates the combined MLX-LM source inventory with these
composition-critical digests:

- `models/kimi_k3.py`:
  `c111751d37a030ec16852b88cb20fb3e2aea537ca458521367fbc5e68b6b110c`
- `kimi_k3_packed_moe_front.py`:
  `9dd1d75ca7022cc837165f9eae670df4df36263ef7011ccd989c9f015db9a159`
- `kimi_k3_w3_prework.py`:
  `f8fcba947dc0e52c335522bd0d152b57679818a64db9edc6e6ba8f58834cb254`
- EXO loader bridge:
  `492a3a5f650b02fa668997bc05d23075e9af0e4a8eecb9068decef2c633da072`

The remaining loader and checkpoint pins are inherited unchanged from the EXO
control, including width four, tail overlap, DSpark checkpoints, native kernels,
auxiliary modules, and stale-source rejection.

## Selector contract

All candidate selectors remain globally default-off. The composed WIDTH3 path
uses the existing strict deferred pair plus these MLX-LM selectors:

- `MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT`
- `MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3`
- `MLX_LM_KIMI_K3_W3_PREWORK_HISTORY`
- `MLX_METAL_K3_AFFINE8_Q3_TRIPLET`

The two authoritative-packed selectors must be `0/0` or `1/1`. Packed WIDTH3
requires the independent native Q3 triplet selector to be `1`; deferred-only
and KDA-only configurations remain valid with native Q3 set to `0`. The
intended controlled A/B contract keeps native Q3 at `1` on both arms while the
deferred, packed, and KDA composition bits change together. Width-four receipt
mode remains mutually exclusive with all W3 candidate selectors.

Deferred, packed, KDA, and native states are bound into the rank verification
plan and request/setup fingerprints even when their values are zero. Native Q3
is also explicit in the checkpoint/source and loaded-model identities, so a
rank or setup mismatch fails before target graph construction or model calls.

## Local verification boundary

The sealed Python 3.13/Metal-capable local environment first ran the focused
loader, loader bridge, Kimi K3 DSpark, deferred/tail, and greedy integration
suites:

```text
303 passed, 9 warnings
```

The broader `scripts/kimi_k3_tp2/tests`, `src/exo/download/tests`, and
`src/exo/worker/tests/unittests/test_mlx` boundary then completed with:

```text
651 passed, 187 deselected, 9 warnings
```

The warnings are pre-existing test-double cleanup warnings from scenario round
engines without a `close` method. No network, host, service, live inference,
benchmark, or timing command was run.
