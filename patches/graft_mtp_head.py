#!/usr/bin/env python3
"""Graft the MTP (NextN) head from the original zai-org/GLM-5.2 checkpoint onto an
MLX-converted GLM-5.2 model that stripped it during conversion.

Background
----------
The MLX conversions on disk (e.g. ``pipenetwork/GLM-5.2-MLX-8bit``) were produced
by a converter that dropped every ``model.layers.<n>.`` weight with ``n >=
num_hidden_layers`` (78). The original ``zai-org/GLM-5.2`` checkpoint on HuggingFace
keeps layer 78 (the single ``num_nextn_predict_layers=1`` MTP head, ~791 weights,
bf16). This script fetches *only* those layer-78 weights from the original repo and
merges them into a target MLX model directory's safetensors shard + index, so the
patched ``deepseek_v32.sanitize`` (see ``patch_deepseek_v32_mtp.py``) can preserve
them for speculative decoding.

Usage
-----
    python graft_mtp_head.py --target ~/.exo/models/pipenetwork--GLM-5.2-MLX-8bit

Idempotent: re-running detects an already-grafted layer 78 and exits 0. Run on each
cluster node that loads the model (the served model dir is per-host).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

SRC_REPO = "zai-org/GLM-5.2"
MTP_LAYER = 78  # GLM-5.2: num_hidden_layers=78, num_nextn_predict_layers=1


def _log(msg: str) -> None:
    print(f"[graft_mtp] {msg}", file=sys.stderr, flush=True)


def _load_index(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_index(path: str, idx: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def _already_grafted(target_index: dict) -> bool:
    wm = target_index.get("weight_map", {})
    return any(k.startswith(f"model.layers.{MTP_LAYER}.") for k in wm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target",
        required=True,
        help="Path to the MLX model directory to graft the MTP head into.",
    )
    ap.add_argument(
        "--source",
        default=SRC_REPO,
        help=f"HuggingFace repo to fetch layer {MTP_LAYER} from (default {SRC_REPO}).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be grafted without writing anything.",
    )
    args = ap.parse_args()

    target_idx_path = os.path.join(args.target, "model.safetensors.index.json")
    if not os.path.isfile(target_idx_path):
        _log(f"ERROR: target index not found: {target_idx_path}")
        return 2

    target_idx = _load_index(target_idx_path)
    if _already_grafted(target_idx):
        _log(f"layer {MTP_LAYER} already present in target index; nothing to do")
        return 0

    # Lazy import — only needed on the cluster where these are installed.
    # Use mlx end-to-end so bf16 is preserved natively (no torch, no numpy
    # upcast to fp32). mlx has native safetensors load (mx.load) and save
    # (mx.save_safetensors), both of which handle bf16.
    try:
        import mlx.core as mx
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        _log(f"ERROR: required dependency missing: {e}")
        _log("Run this on a cluster node with the exo venv active.")
        return 3

    # 1. Fetch the source index and find which shards hold layer-78 weights.
    _log(f"fetching source index from {args.source} ...")
    src_idx_path = hf_hub_download(args.source, "model.safetensors.index.json")
    src_idx = _load_index(src_idx_path)
    src_wm = src_idx["weight_map"]
    mtp_keys = sorted(k for k in src_wm if k.startswith(f"model.layers.{MTP_LAYER}."))
    if not mtp_keys:
        _log(f"ERROR: source {args.source} has no layer {MTP_LAYER} weights")
        return 4
    _log(f"found {len(mtp_keys)} layer-{MTP_LAYER} weights in source")

    src_shards_needed: dict[str, list[str]] = defaultdict(list)
    for k in mtp_keys:
        src_shards_needed[src_wm[k]].append(k)
    _log(f"layer {MTP_LAYER} spans {len(src_shards_needed)} source shards: "
         f"{sorted(src_shards_needed)}")

    # 2. Pull the MTP tensors out of the (bf16) source shards as mlx arrays,
    #    preserving bf16 natively. We use mx.load() (mlx's native safetensors
    #    loader) rather than safetensors.safe_open, because the latter's mlx
    #    backend trips over numpy's lack of a bfloat16 dtype. mx.load reads the
    #    whole shard into a dict; we keep only the layer-78 keys we need.
    _log("extracting MTP tensors from source shards ...")
    mtp_tensors: dict[str, "mx.array"] = {}
    for shard_file, keys in sorted(src_shards_needed.items()):
        shard_path = hf_hub_download(args.source, shard_file)
        all_shard = dict(mx.load(shard_path))
        for k in keys:
            mtp_tensors[k] = all_shard[k]
        # Drop the rest so we don't hold the full shard in memory.
        del all_shard
        _log(f"  loaded {len(keys)} tensors from {shard_file}")

    # 3. The served MLX body is 8-bit quantized, but the MTP head is small
    #    (~1/78 of params) and runs only as a draft — keep it at the source bf16
    #    for simplicity and quality. Memory cost is a few hundred MB.
    new_shard_name = "model.mtp-head.safetensors"
    _log(f"will write {len(mtp_tensors)} tensors to new shard {new_shard_name}")

    if args.dry_run:
        _log("DRY RUN — not writing. Sample keys:")
        for k in mtp_keys[:8]:
            v = mtp_tensors[k]
            _log(f"  {k}: shape={v.shape} dtype={v.dtype}")
        return 0

    # 4. Write the grafted shard via mlx's native safetensors save (bf16 preserved).
    new_shard_path = os.path.join(args.target, new_shard_name)
    mx.save_safetensors(new_shard_path, mtp_tensors)
    _log(f"wrote {new_shard_path} ({os.path.getsize(new_shard_path) / 1e9:.2f} GB)")

    # 5. Update the target index: add all layer-78 keys pointing at the new shard,
    #    bump total_size, and record the new shard in weight_map / metadata.
    new_wm = dict(target_idx.get("weight_map", {}))
    for k in mtp_keys:
        new_wm[k] = new_shard_name

    # Recompute per-shard size map from the new weight_map.
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for k, s in new_wm.items():
        shard_to_keys[s].append(k)

    # total_size is the sum of all tensor bytes; recompute from what we know.
    # The body's total_size is preserved; add the MTP shard's contribution.
    added_bytes = sum(int(arr.size * arr.itemsize) for arr in mtp_tensors.values())
    prev_total = int(target_idx.get("metadata", {}).get("total_size", 0))
    new_total = prev_total + added_bytes

    new_idx = dict(target_idx)
    new_idx["weight_map"] = new_wm
    new_idx.setdefault("metadata", {})["total_size"] = new_total
    _save_index(target_idx_path, new_idx)
    _log(f"updated index: +{len(mtp_keys)} keys, total_size {prev_total} -> {new_total}")
    _log("graft complete. Set EXO_MTP_SPECULATIVE=1 to enable MTP speculative decode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
