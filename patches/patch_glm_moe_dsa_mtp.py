#!/usr/bin/env python3
"""Patch mlx_lm/models/glm_moe_dsa.py to attach an MTP (NextN) draft head when
EXO_MTP_SPECULATIVE=1.

This patch is deliberately small: it does NOT define the MTP head here (that
lives in exo at exo/worker/engines/mlx/mtp.py, so it can evolve without
re-patching the vendored file). Instead it:

  1. Imports ``MTPHead`` from the exo package (lazy, guarded by the env flag so
     the unpatched path is unaffected).
  2. In ``Model.__init__``, after building ``self.model``, conditionally builds
     ``self.mtp_head = MTPHead(config)`` (or ``None``).
  3. Exposes ``self.mtp_head`` so speculative_generate can reach it.
  4. Extends ``make_cache`` so the MTP head's attention gets a cache entry
     appended at the end of the list (after the 78 decoder-layer caches).

The MTP head weights are loaded by mlx-lm's loader because sanitize() now keeps
``model.layers.78.*`` keys (see patch_deepseek_v32_mtp.py); MTPHead.load_weights
remaps them onto its submodule tree. The head is intentionally NOT added to
``self.model.layers`` so the sharding loop in auto_parallel never touches it —
it stays replicated on every node (the "drafter stays local" property).

Idempotent via the marker comment.

Usage:
    python patch_glm_moe_dsa_mtp.py /path/to/mlx_lm/models/glm_moe_dsa.py
"""
from __future__ import annotations

import sys

MARKER = "# EXO_MTP_DSA_PATCH_APPLIED"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_glm_moe_dsa_mtp.py <glm_moe_dsa.py>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    if MARKER in content:
        print(f"[patch_glm_dsa_mtp] already patched: {path}")
        return 0

    # 1. Add the lazy MTPHead import near the top, after the existing imports.
    import_anchor = "from .deepseek_v32 import Model as DSV32Model\n"
    if import_anchor not in content:
        print(f"[patch_glm_dsa_mtp] ERROR: import anchor not found in {path}", file=sys.stderr)
        return 3
    import_addition = (
        import_anchor
        + "\n"
        + "# EXO_MTP_DSA_PATCH_APPLIED\n"
        + "# Lazy import of the MTP draft head. Lives in the exo package so it can\n"
        + "# evolve without re-patching this vendored file.\n"
        + "def _load_mtp_head_class():\n"
        + "    try:\n"
        + "        from exo.worker.engines.mlx.mtp import MTPHead\n"
        + "        return MTPHead\n"
        + "    except Exception:\n"
        + "        return None\n"
    )
    content = content.replace(import_anchor, import_addition, 1)

    # 1b. Add num_nextn_predict_layers to ModelArgs so it survives from_dict().
    #     The base ModelArgs (deepseek_v32) does not declare this field, so the
    #     config value is silently dropped — and Model.__init__'s MTP gate then
    #     sees 0. Insert at the END of the dataclass fields (after the last
    #     defaulted field) to respect dataclass ordering (defaults after non-defaults).
    args_anchor = "    index_skip_topk_offset: int = 2\n"
    if args_anchor not in content:
        print(f"[patch_glm_dsa_mtp] WARNING: ModelArgs anchor not found; "
              "num_nextn_predict_layers may not be added.", file=sys.stderr)
    else:
        args_addition = (
            args_anchor
            + "    num_nextn_predict_layers: Optional[int] = 0  # MTP/NextN layers (EXO_MTP patch)\n"
        )
        content = content.replace(args_anchor, args_addition, 1)

    # 2. Extend Model.__init__ to build mtp_head when the flag is set AND the
    #    checkpoint actually has MTP weights (num_nextn_predict_layers > 0).
    old_init = (
        "class Model(DSV32Model):\n"
        "    def __init__(self, config: ModelArgs):\n"
        "        super().__init__(config)\n"
        "        self.model = GlmMoeDsaModel(config)\n"
    )
    new_init = (
        "class Model(DSV32Model):\n"
        "    def __init__(self, config: ModelArgs):\n"
        "        super().__init__(config)\n"
        "        self.model = GlmMoeDsaModel(config)\n"
        "        # MTP (NextN) draft head for speculative decoding. Built only when\n"
        "        # EXO_MTP_SPECULATIVE is set AND the checkpoint carries an MTP layer\n"
        "        # (num_nextn_predict_layers > 0). Otherwise stays None and the whole\n"
        "        # speculative path is inert — zero behavior change vs upstream.\n"
        "        import os as _os\n"
        "        self.mtp_head = None\n"
        "        _has_mtp = bool(getattr(config, \"num_nextn_predict_layers\", 0))\n"
        "        if _os.environ.get(\"EXO_MTP_SPECULATIVE\", \"\").lower() in (\"1\", \"true\", \"yes\") and _has_mtp:\n"
        "            _MTPHead = _load_mtp_head_class()\n"
        "            if _MTPHead is not None:\n"
        "                # Locate the standalone MTP shard (model.mtp-head.safetensors).\n"
        "                # It's kept OUT of the main index so exo's download-integrity\n"
        "                # check doesn't wipe it. Explicit path wins; else search the\n"
        "                # standard exo model dirs one level deep.\n"
        "                _shard = _os.environ.get(\"EXO_MTP_SHARD\")\n"
        "                if not _shard:\n"
        "                    _models_dir = _os.path.expanduser(\"~/.exo/models\")\n"
        "                    if _os.path.isdir(_models_dir):\n"
        "                        for _d in _os.listdir(_models_dir):\n"
        "                            _p = _os.path.join(_models_dir, _d, \"model.mtp-head.safetensors\")\n"
        "                            if _os.path.isfile(_p):\n"
        "                                _shard = _p\n"
        "                                break\n"
        "                self.mtp_head = _MTPHead(config, shard_path=_shard)\n"
    )
    if old_init not in content:
        print(f"[patch_glm_dsa_mtp] ERROR: Model.__init__ anchor not found in {path}", file=sys.stderr)
        return 4
    content = content.replace(old_init, new_init, 1)

    # NOTE: make_cache is intentionally NOT modified. The MTP head's KV cache must
    # NOT be appended to the model's cache list, because auto_parallel's
    # patched_call walks cache[-1] and calls mx.depends(dep_cache.keys, logits) —
    # if the last entry is an untouched MTP cache, its .keys is None and that
    # raises TypeError. Instead, speculative_generate builds and manages the MTP
    # draft cache separately (MTPHead.make_draft_cache), keeping it out of the
    # model's distributed-cache iteration entirely.

    with open(path, "w") as f:
        f.write(content)
    print(f"[patch_glm_dsa_mtp] patched Model.__init__ in {path} (make_cache unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
