# Kimi K3 width-four terminal dispatch receipt

Date: 2026-08-13

This branch adds a strict, default-off service-terminal receipt for the Kimi K3
width-four candidate. It does not change routing when disabled.

Enable all three receipt selectors in a fresh C1 worker process:

```text
EXO_MLX_KIMI_K3_WIDTH4_DISPATCH_RECEIPT_LOG=1
MLX_LM_KIMI_K3_WIDTH4_DISPATCH_RECEIPT=1
MLX_METAL_K3_AFFINE6_Q4_DISPATCH_RECEIPT=1
```

At terminal response construction, after the output has been consumed, each
rank writes one grep-safe `K3_WIDTH4_DISPATCH_RECEIPT` JSON record. The record
binds hostname, process ID, TP rank, sibling `libmlx.dylib` path, native Q4
dispatch count, and the transactional MLX-LM W4 receipt. It fails closed on a
missing native symbol, impossible/nonterminal Python totals, a stale completion
from an earlier receipt generation, or any child selector not exactly `1`.

The counters are process-cumulative. Promotion-grade C1 therefore uses a fresh
worker process and exactly one isolated request. Both rank logs must show:

- `terminal_output_consumed=true`;
- the authenticated candidate `libmlx.dylib` path;
- native Q4 dispatch count greater than zero;
- Python `attempted = dispatched + fallback + error`;
- Python `dispatched > 0`, `fallback = 0`, and `error = 0`; and
- `stale_completions.total = 0`.

The log proves service-path attribution. It does not itself prove output
quality, performance, JACCL correctness, or a 25 tok/s result; those remain
separate protected bracket gates.
