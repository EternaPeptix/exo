#!/usr/bin/env python3
"""Apply all MTP-speculative-decode patches for exo + GLM-5.2.

Run this on each cluster node (512S1 / 512S2) after pulling the exo repo. It is
idempotent — safe to re-run. Steps:

  1. Graft the MTP (NextN) head weights from the original zai-org/GLM-5.2
     checkpoint onto the served MLX model directory (if not already present).
     Requires huggingface_hub + safetensors + mlx (only on the cluster).
  2. Patch mlx_lm/models/deepseek_v32.py: sanitize() keeps + renames layer-78
     weights to mtp_head.* when EXO_MTP_SPECULATIVE=1.
  3. Patch mlx_lm/models/glm_moe_dsa.py: Model builds an MTPHead and make_cache
     appends a draft cache slot.

The vendored mlx-lm lives at <repo>/mlx-lm/mlx_lm (editable install on the
cluster). We discover it via ``import mlx_lm`` so the patch hits the real file.

Usage:
    python patches/apply_patches.py [--model-dir DIR] [--skip-graft] [--dry-run]

    --model-dir   Path to the MLX model to graft the MTP head into
                  (default: ~/.exo/models/pipenetwork--GLM-5.2-MLX-8bit).
    --skip-graft  Skip the weight graft (e.g. already done, or offline).
    --dry-run     Show what would happen without writing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[apply_patches] {msg}", file=sys.stderr, flush=True)


def _find_repo_root() -> Path:
    # apply_patches.py lives in <repo>/patches/
    return Path(__file__).resolve().parent.parent


def _find_mlx_lm_file(rel: str) -> Path | None:
    """Resolve a path inside the installed (editable) mlx_lm package."""
    try:
        import mlx_lm

        base = Path(mlx_lm.__file__).resolve().parent
        target = base / rel
        if target.is_file():
            return target
    except Exception as e:
        _log(f"could not import mlx_lm to locate {rel}: {e}")
    # Fallback: <repo>/mlx-lm/mlx_lm/<rel>
    repo = _find_repo_root()
    fb = repo / "mlx-lm" / "mlx_lm" / rel
    return fb if fb.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/.exo/models/pipenetwork--GLM-5.2-MLX-8bit"),
        help="MLX model directory to graft the MTP head into.",
    )
    ap.add_argument("--skip-graft", action="store_true", help="Skip the weight graft step.")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing.")
    args = ap.parse_args()

    repo = _find_repo_root()
    patches_dir = repo / "patches"
    rc = 0

    # ---- Step 1: graft MTP head weights ----------------------------------
    if not args.skip_graft:
        graft = patches_dir / "graft_mtp_head.py"
        target_idx = Path(args.model_dir) / "model.safetensors.index.json"
        if not target_idx.is_file():
            _log(f"SKIP graft: target index not found: {target_idx}")
        else:
            import json

            idx = json.loads(target_idx.read_text())
            already = any(k.startswith("model.layers.78.") or k.startswith("mtp_head.")
                          for k in idx.get("weight_map", {}))
            if already:
                _log("graft: layer 78 already present in target; skipping")
            else:
                _log(f"grafting MTP head into {args.model_dir} ...")
                cmd = [sys.executable, str(graft), "--target", args.model_dir]
                if args.dry_run:
                    cmd.append("--dry-run")
                ret = os.spawnvp(os.P_WAIT, cmd[0], cmd)
                if ret != 0:
                    _log(f"WARNING: graft returned {ret}; MTP path will fall back to n-gram")
                    rc = max(rc, ret)
                else:
                    _log("graft complete")
    else:
        _log("graft skipped (--skip-graft)")

    # ---- Step 2: patch deepseek_v32.py sanitize --------------------------
    dsv32 = _find_mlx_lm_file("models/deepseek_v32.py")
    if dsv32 is None:
        _log("ERROR: could not locate mlx_lm/models/deepseek_v32.py")
        rc = max(rc, 3)
    else:
        patch = patches_dir / "patch_deepseek_v32_mtp.py"
        _log(f"patching sanitize() in {dsv32} ...")
        cmd = [sys.executable, str(patch), str(dsv32)]
        ret = os.spawnvp(os.P_WAIT, cmd[0], cmd)
        if ret != 0:
            _log(f"ERROR: deepseek_v32 patch returned {ret}")
            rc = max(rc, ret)

    # ---- Step 3: patch glm_moe_dsa.py Model + make_cache -----------------
    glm_dsa = _find_mlx_lm_file("models/glm_moe_dsa.py")
    if glm_dsa is None:
        _log("ERROR: could not locate mlx_lm/models/glm_moe_dsa.py")
        rc = max(rc, 3)
    else:
        patch = patches_dir / "patch_glm_moe_dsa_mtp.py"
        _log(f"patching Model + make_cache in {glm_dsa} ...")
        cmd = [sys.executable, str(patch), str(glm_dsa)]
        ret = os.spawnvp(os.P_WAIT, cmd[0], cmd)
        if ret != 0:
            _log(f"ERROR: glm_moe_dsa patch returned {ret}")
            rc = max(rc, ret)

    if rc == 0:
        _log("all patches applied. Set EXO_MTP_SPECULATIVE=1 to enable MTP spec decode.")
    else:
        _log(f"finished with non-zero rc={rc}; see warnings above.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
