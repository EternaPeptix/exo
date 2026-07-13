#!/usr/bin/env python3
"""Patch mlx_lm/models/deepseek_v32.py so the MTP (NextN) head weights are
preserved at load when EXO_MTP_SPECULATIVE=1, instead of being dropped by the
default sanitize().

The default sanitize() unconditionally discards every weight whose key matches
``model.layers.<n>.*`` with ``n >= num_hidden_layers``. For GLM-5.2
(``num_hidden_layers=78``, ``num_nextn_predict_layers=1``) that means layer 78
(the single MTP head) is stripped. When ``EXO_MTP_SPECULATIVE`` is set, we want
those weights to survive so the MTP draft head (see exo/.../mlx/mtp.py) can be
populated.

This patch rewrites the drop-loop in ``Model.sanitize`` so that:
  - When EXO_MTP_SPECULATIVE is OFF (default): behavior is byte-identical to
    upstream — layer-78+ weights are dropped. Zero risk to existing deployments.
  - When EXO_MTP_SPECULATIVE is ON: layer-78+ weights are KEPT in the returned
    dict (left under their original ``model.layers.78.*`` keys). The downstream
    expert-stacking loop only iterates ``range(num_hidden_layers)`` (0..77), so
    it ignores layer 78 — we let MTPHead.load_weights() consume them directly.

Idempotent: re-running detects the patched marker and exits 0.

Usage:
    python patch_deepseek_v32_mtp.py /path/to/mlx_lm/models/deepseek_v32.py
"""
from __future__ import annotations

import os
import sys

MARKER = "# EXO_MTP_PATCH_APPLIED"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_deepseek_v32_mtp.py <deepseek_v32.py>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if MARKER in content:
        print(f"[patch_dsv32_mtp] already patched: {path}")
        return 0

    # The exact current (unpatched) drop-loop in Model.sanitize:
    old = (
        "    def sanitize(self, weights):\n"
        "        # Remove multi-token prediction layers\n"
        "        mpt_layer = self.args.num_hidden_layers\n"
        "        new_weights = {}\n"
        "        for k, v in weights.items():\n"
        "            parts = k.split(\".\")\n"
        "            if len(parts) >= 3 and parts[1] == \"layers\" and int(parts[2]) >= mpt_layer:\n"
        "                continue\n"
        "            new_weights[k] = v\n"
        "        weights = new_weights\n"
    )

    new = (
        "    def sanitize(self, weights):\n"
        "        # EXO_MTP_PATCH_APPLIED\n"
        "        # Conditionally keep multi-token prediction (MTP / NextN) layers.\n"
        "        # Default behavior (EXO_MTP_SPECULATIVE unset) is byte-identical to\n"
        "        # upstream: drop every layer with index >= num_hidden_layers. When\n"
        "        # EXO_MTP_SPECULATIVE is set, those weights (e.g. layer 78 in\n"
        "        # GLM-5.2) are KEPT and renamed from ``model.layers.<n>.*`` to\n"
        "        # ``mtp_head.*`` so they populate the top-level MTPHead submodule on\n"
        "        # Model (built by patch_glm_moe_dsa_mtp.py). The expert-stacking /\n"
        "        # kv_b_proj reshape loops below only iterate range(num_hidden_layers),\n"
        "        # so they never touch the MTP layer; MTPHead does its own stacking in\n"
        "        # its load_weights. This keeps upstream body-processing unchanged.\n"
        "        import os as _os\n"
        "        _keep_mtp = _os.environ.get(\"EXO_MTP_SPECULATIVE\", \"\").lower() in (\"1\", \"true\", \"yes\")\n"
        "        mpt_layer = self.args.num_hidden_layers\n"
        "        new_weights = {}\n"
        "        for k, v in weights.items():\n"
        "            parts = k.split(\".\")\n"
        "            if len(parts) >= 3 and parts[1] == \"layers\" and int(parts[2]) >= mpt_layer:\n"
        "                if _keep_mtp:\n"
        "                    # model.layers.78.<rest> -> mtp_head.<rest>\n"
        "                    new_weights[\"mtp_head.\" + \".\".join(parts[3:])] = v\n"
        "                continue\n"
        "            new_weights[k] = v\n"
        "        weights = new_weights\n"
    )

    if old not in content:
        print(f"[patch_dsv32_mtp] ERROR: target drop-loop not found in {path}", file=sys.stderr)
        print("[patch_dsv32_mtp] The upstream sanitize() shape may have changed; "
              "inspect the file and update this patch.", file=sys.stderr)
        return 3

    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"[patch_dsv32_mtp] patched sanitize() in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
