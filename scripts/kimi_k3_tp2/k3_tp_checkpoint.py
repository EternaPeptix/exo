#!/usr/bin/env python3
"""Transactional, bounded-memory TP checkpoint converter for Kimi K3.

This module mirrors the tensor slicing performed by ``Model.shard`` in:

  ml-explore/mlx-lm@7d505c285b801108a52c23353c7fb6af07204717

It deliberately has no MLX, safetensors, or huggingface_hub dependency.  The
converter reads and writes the safetensors format directly and uses NumPy only
as a bounded-memory view/copy engine.

The intended production topology is:

* run once on a conversion coordinator;
* provide two distinct rank roots, local or mounted;
* download one pinned source shard;
* commit both rank-local outputs transactionally;
* delete the source shard only after both outputs are durable;
* continue with the next source shard.

This avoids two full Hugging Face downloads and never needs a complete 817 GB
source checkpoint on disk.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import mmap
import os
import re
import stat
import struct
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

import numpy as np

SOURCE_REPO = "kernelpool/Kimi-K3-2bit-UVMAX"
SOURCE_REVISION = "edb5113218df612f4a92f95145680f3f8eacd375"
MLX_LM_PR = "https://github.com/ml-explore/mlx-lm/pull/1626"
MLX_LM_COMMIT = "7d505c285b801108a52c23353c7fb6af07204717"
MLX_LM_KIMI_K3_SHA256 = (
    "3dd2e9db585190bca118d5812bcb5b103d1e7c6ec12187b20351992fed7e63cc"
)
SOURCE_CONFIG_SHA256 = (
    "d041003554810a367bb600d18733976bdd21041bb46e75cc1e27c7b15fe034d0"
)
SOURCE_INDEX_SHA256 = "ac65bcb3cd9e07cab3e7942ff455dde33879a9e02211bae40938e22fc204ae09"
SCHEMA = "k3-rank-local-tp/v2"
LEGACY_SCHEMA = "k3-rank-local-tp/v1"
JOURNAL_SCHEMAS = frozenset({LEGACY_SCHEMA, SCHEMA})
SHARD_AUDIT_SCHEMA = "k3-rank-local-tp-shard-audit/v1"
SHARD_CONVERSION_SCHEMA = "k3-rank-local-tp-shard-conversion/v1"
MANIFEST_UPGRADE_SCHEMA = "k3-rank-local-tp-manifest-upgrade/v1"
MANIFEST_UPGRADE_TRANSACTION_SCHEMA = "k3-rank-local-tp-manifest-upgrade-transaction/v1"
LEGACY_MANIFEST_SHA256 = {
    0: "2da4db586e81a61fbea990e95fcfa14d14a0563c6cdafc6767373f5a74e12b21",
    1: "c91c25a9439471ba115d45ae3371341954bbb0453677549c1fb6eeb2ace0d23f",
}
MANIFEST_UPGRADE_TRANSACTION_STATES = frozenset(
    {
        "prepared",
        "rank-0-published",
        "rank-1-published",
        "recovering-rollback",
        "rollback-rank-1-durable",
        "rollback-rank-0-durable",
        "rolled-back",
        "recovering-complete",
        "complete-rank-0-durable",
        "complete-rank-1-durable",
        "committed",
    }
)
MANIFEST_UPGRADE_STATE_TRANSITIONS = {
    "prepared": frozenset(
        {"rank-0-published", "recovering-rollback", "recovering-complete"}
    ),
    "rank-0-published": frozenset(
        {"rank-1-published", "recovering-rollback", "recovering-complete"}
    ),
    "rank-1-published": frozenset(
        {"committed", "recovering-rollback", "recovering-complete"}
    ),
    "recovering-rollback": frozenset(
        {
            "recovering-rollback",
            "recovering-complete",
            "rollback-rank-1-durable",
        }
    ),
    "rollback-rank-1-durable": frozenset(
        {
            "recovering-rollback",
            "recovering-complete",
            "rollback-rank-0-durable",
        }
    ),
    "rollback-rank-0-durable": frozenset(
        {"recovering-rollback", "recovering-complete", "rolled-back"}
    ),
    "rolled-back": frozenset({"recovering-rollback", "recovering-complete"}),
    "recovering-complete": frozenset(
        {
            "recovering-complete",
            "recovering-rollback",
            "complete-rank-0-durable",
        }
    ),
    "complete-rank-0-durable": frozenset(
        {
            "recovering-complete",
            "recovering-rollback",
            "complete-rank-1-durable",
        }
    ),
    "complete-rank-1-durable": frozenset(
        {"recovering-complete", "recovering-rollback", "committed"}
    ),
    "committed": frozenset({"recovering-rollback", "recovering-complete"}),
}
CONTRACT_VERSION = "mlx-lm-kimi-k3-shard@7d505c2"
DEFAULT_WORLD_SIZE = 2
TEXT_PREFIX = "language_model."
INTERNAL_CHECKPOINT_FILENAMES = frozenset(
    {"model.safetensors.index.json", "tp_manifest.json"}
)

# Exact non-weight inventory from SOURCE_REPO@SOURCE_REVISION.  These files are
# executable under ``trust_remote_code=True``; an allowlist of names alone is
# not authentication.  The immutable per-file identities deliberately reject
# both extra files and byte substitutions.  ``.gitattributes`` is not runtime
# metadata and is intentionally excluded.
PINNED_METADATA_FILES: dict[str, dict[str, int | str]] = {
    "README.md": {
        "bytes": 1923,
        "sha256": "d0d7a4d1a5af37c542594449d2ce893b9e3c33ccb031afb71cf13e4f23a5349d",
    },
    "added_tokens.json": {
        "bytes": 200,
        "sha256": "27373c2f39a52c87e674caf7e9604ec6756c68c5f8d5f140657299048b6ab8ba",
    },
    "config.json": {
        "bytes": 459349,
        "sha256": SOURCE_CONFIG_SHA256,
    },
    "configuration_kimi_k3.py": {
        "bytes": 11343,
        "sha256": "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
    },
    "encoding_k3.py": {
        "bytes": 22827,
        "sha256": "b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333",
    },
    "generation_config.json": {
        "bytes": 53,
        "sha256": "c6648c25e9705af7fba8847e243840d21b5cc63ddeb6297f750a7ddbb6a02836",
    },
    "kimi_k3_processor.py": {
        "bytes": 7660,
        "sha256": "ec9f7e86d2ab0eee07a8e7e7c037046e77ac3c25a710ad1298ec13be3b585b54",
    },
    "kimi_k3_vision_processing.py": {
        "bytes": 6686,
        "sha256": "d122b30bfd3a51a6f05d4bfcfda1e657827322b1353f7caefeebc2835d7736b5",
    },
    "media_utils.py": {
        "bytes": 13844,
        "sha256": "78403540328f9847d6b7ebc5c44eb2e6a752863de0afb7d0710728bb161dc60d",
    },
    "modeling_kimi_k3.py": {
        "bytes": 53444,
        "sha256": "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2",
    },
    "modeling_kimi_linear.py": {
        "bytes": 51506,
        "sha256": "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    },
    "preprocessor_config.json": {
        "bytes": 1011,
        "sha256": "4be333605990c53a816e586dee9d5dd545afb7a59947c17f8f7ef26b4782668e",
    },
    "tiktoken.model": {
        "bytes": 2795286,
        "sha256": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    },
    "tokenization_kimi.py": {
        "bytes": 16145,
        "sha256": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    },
    "tokenizer_config.json": {
        "bytes": 4790,
        "sha256": "d06a6e8a2ef0a09d62031591d0ea2b7c5128fd28a17ea693984bf85eafade1df",
    },
}
ALLOWED_METADATA_FILENAMES = frozenset(PINNED_METADATA_FILES)
REQUIRED_METADATA_FILENAMES = ALLOWED_METADATA_FILENAMES


DTYPES: dict[str, tuple[np.dtype, int]] = {
    "BOOL": (np.dtype("u1"), 1),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "I16": (np.dtype("<i2"), 2),
    "U16": (np.dtype("<u2"), 2),
    "I32": (np.dtype("<i4"), 4),
    "U32": (np.dtype("<u4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "U64": (np.dtype("<u8"), 8),
    "F16": (np.dtype("<f2"), 2),
    # NumPy on the coordinator may not expose bfloat16.  Treating BF16 as
    # little-endian uint16 preserves bits exactly while slicing.
    "BF16": (np.dtype("<u2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "F64": (np.dtype("<f8"), 8),
    "C64": (np.dtype("<c8"), 8),
    "C128": (np.dtype("<c16"), 16),
    "F8_E4M3": (np.dtype("u1"), 1),
    "F8_E5M2": (np.dtype("u1"), 1),
    "F8_E8M0": (np.dtype("u1"), 1),
}


class ConversionError(RuntimeError):
    """Raised when an input cannot be proven safe to convert."""


@dataclasses.dataclass(frozen=True)
class TensorDesc:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def itemsize(self) -> int:
        try:
            return DTYPES[self.dtype][1]
        except KeyError as exc:
            raise ConversionError(
                f"{self.name}: unsupported safetensors dtype {self.dtype!r}"
            ) from exc

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.itemsize


@dataclasses.dataclass(frozen=True)
class ShardRule:
    """A tensor slicing rule equivalent to MLX ``_shard``."""

    kind: str
    axis: int | None = None
    segments: int = 1
    reason: str = ""

    def canonical(self) -> dict:
        return {
            "kind": self.kind,
            "axis": self.axis,
            "segments": self.segments,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class TensorPlan:
    source: TensorDesc
    output_shape: tuple[int, ...]
    rule: ShardRule
    intervals: tuple[tuple[int, int], ...]
    axis: int | None

    @property
    def output_nbytes(self) -> int:
        return math.prod(self.output_shape) * self.source.itemsize


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SafeTensorFile:
    """Validated read-only safetensors file backed by mmap."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        self.metadata: dict[str, str] = {}
        self.tensors: OrderedDict[str, TensorDesc] = OrderedDict()
        self.data_offset = 0

    def __enter__(self) -> "SafeTensorFile":
        self._fh = os.fdopen(_open_regular_readonly(self.path), "rb")
        file_size = os.fstat(self._fh.fileno()).st_size
        raw_len = self._fh.read(8)
        if len(raw_len) != 8:
            raise ConversionError(f"{self.path}: truncated safetensors prefix")
        (header_len,) = struct.unpack("<Q", raw_len)
        if not (2 <= header_len <= 100_000_000):
            raise ConversionError(
                f"{self.path}: unreasonable safetensors header {header_len} bytes"
            )
        header_raw = self._fh.read(header_len)
        if len(header_raw) != header_len:
            raise ConversionError(f"{self.path}: truncated safetensors header")
        try:

            def reject_duplicate_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key {key!r}")
                    result[key] = value
                return result

            header = json.loads(
                header_raw.rstrip(b" ").decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
        except Exception as exc:
            raise ConversionError(
                f"{self.path}: invalid safetensors JSON header"
            ) from exc
        if not isinstance(header, dict):
            raise ConversionError(f"{self.path}: header is not an object")

        self.data_offset = 8 + header_len
        data_bytes = file_size - self.data_offset
        occupied: list[tuple[int, int, str]] = []
        for name, info in header.items():
            if name == "__metadata__":
                if not isinstance(info, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in info.items()
                ):
                    raise ConversionError(f"{self.path}: invalid __metadata__")
                self.metadata = dict(info)
                continue
            if not isinstance(name, str) or not name:
                raise ConversionError(f"{self.path}: invalid tensor name {name!r}")
            if not isinstance(info, dict) or set(info) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ConversionError(f"{self.path}: invalid descriptor for {name!r}")
            dtype = info["dtype"]
            raw_shape = info["shape"]
            raw_offsets = info["data_offsets"]
            if (
                not isinstance(dtype, str)
                or not isinstance(raw_shape, list)
                or any(type(dimension) is not int for dimension in raw_shape)
                or not isinstance(raw_offsets, list)
                or len(raw_offsets) != 2
                or any(type(offset) is not int for offset in raw_offsets)
            ):
                raise ConversionError(f"{self.path}: invalid descriptor for {name!r}")
            shape = tuple(raw_shape)
            start, end = raw_offsets
            if dtype not in DTYPES:
                raise ConversionError(
                    f"{self.path}:{name}: unsupported dtype {dtype!r}"
                )
            if any(x < 0 for x in shape):
                raise ConversionError(f"{self.path}:{name}: negative shape")
            if start < 0 or end < start or end > data_bytes:
                raise ConversionError(
                    f"{self.path}:{name}: offsets [{start}, {end}) outside data"
                )
            desc = TensorDesc(
                name=name,
                dtype=dtype,
                shape=shape,
                data_start=self.data_offset + start,
                data_end=self.data_offset + end,
            )
            if desc.nbytes != end - start:
                raise ConversionError(
                    f"{self.path}:{name}: shape implies {desc.nbytes} bytes, "
                    f"header has {end - start}"
                )
            self.tensors[name] = desc
            occupied.append((start, end, name))

        occupied.sort()
        previous_end = 0
        for start, end, name in occupied:
            if start < previous_end:
                raise ConversionError(f"{self.path}:{name}: overlapping tensor data")
            previous_end = end

        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *_exc) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._fh is not None:
            self._fh.close()
        self._mmap = None
        self._fh = None

    def array(self, desc: TensorDesc) -> np.ndarray:
        if self._mmap is None:
            raise RuntimeError("SafeTensorFile must be opened as a context manager")
        storage_dtype = DTYPES[desc.dtype][0]
        return np.ndarray(
            shape=desc.shape,
            dtype=storage_dtype,
            buffer=self._mmap,
            offset=desc.data_start,
            order="C",
        )


def _build_header(
    plans: Sequence[TensorPlan],
    metadata: Mapping[str, str],
) -> tuple[bytes, OrderedDict[str, dict]]:
    header: OrderedDict[str, dict] = OrderedDict()
    header["__metadata__"] = dict(metadata)
    cursor = 0
    for plan in plans:
        end = cursor + plan.output_nbytes
        header[plan.source.name] = {
            "dtype": plan.source.dtype,
            "shape": list(plan.output_shape),
            "data_offsets": [cursor, end],
        }
        cursor = end
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    # The safetensors header must be padded to an 8-byte boundary.
    raw += b" " * ((-len(raw)) % 8)
    return raw, header


def _copy_plan(
    source: SafeTensorFile,
    plan: TensorPlan,
    out: BinaryIO,
    digest: "hashlib._Hash",
    max_buffer_bytes: int,
) -> None:
    array = source.array(plan.source)
    if plan.axis is None:
        flat = array.reshape(-1)
        elems = max(1, max_buffer_bytes // plan.source.itemsize)
        for start in range(0, flat.size, elems):
            data = np.ascontiguousarray(flat[start : start + elems])
            raw = memoryview(data).cast("B")
            out.write(raw)
            digest.update(raw)
        return

    axis = plan.axis
    prefix = math.prod(plan.source.shape[:axis])
    axis_size = plan.source.shape[axis]
    suffix = math.prod(plan.source.shape[axis + 1 :])
    reshaped = array.reshape(prefix, axis_size, suffix)
    output_axis = sum(end - start for start, end in plan.intervals)
    bytes_per_prefix = max(1, output_axis * suffix * plan.source.itemsize)
    rows_per_chunk = max(1, max_buffer_bytes // bytes_per_prefix)

    for row in range(0, prefix, rows_per_chunk):
        block = reshaped[row : row + rows_per_chunk]
        pieces = [block[:, start:end, :] for start, end in plan.intervals]
        if len(pieces) == 1:
            data = np.ascontiguousarray(pieces[0])
        else:
            data = np.concatenate(pieces, axis=1)
        raw = memoryview(data).cast("B")
        out.write(raw)
        digest.update(raw)


def write_rank_shard(
    source: SafeTensorFile,
    destination: Path,
    plans: Sequence[TensorPlan],
    *,
    rank: int,
    world_size: int,
    max_buffer_bytes: int,
    temporary_path: Path | None = None,
) -> dict:
    """Write one rank output with fsync + header revalidation + atomic rename."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "mlx",
        "schema": SCHEMA,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file": source.path.name,
        "tp_rank": str(rank),
        "tp_world_size": str(world_size),
        "sharding_contract": CONTRACT_VERSION,
    }
    header_raw, _ = _build_header(plans, metadata)
    expected_size = 8 + len(header_raw) + sum(p.output_nbytes for p in plans)
    if temporary_path is None:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
        )
        tmp = Path(tmp_name)
    else:
        tmp = Path(temporary_path)
        if tmp.parent.resolve() != destination.parent.resolve():
            raise ConversionError(
                f"{tmp}: deterministic temporary must share destination directory"
            )
        try:
            fd = os.open(
                tmp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise ConversionError(
                f"{tmp}: deterministic conversion temporary already exists"
            ) from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out:
            prefix = struct.pack("<Q", len(header_raw))
            out.write(prefix)
            out.write(header_raw)
            digest.update(prefix)
            digest.update(header_raw)
            for plan in plans:
                _copy_plan(source, plan, out, digest, max_buffer_bytes)
            out.flush()
            os.fsync(out.fileno())

        if tmp.stat().st_size != expected_size:
            raise ConversionError(
                f"{tmp}: wrote {tmp.stat().st_size}, expected {expected_size}"
            )
        with SafeTensorFile(tmp) as check:
            expected = {p.source.name: (p.source.dtype, p.output_shape) for p in plans}
            actual = {
                name: (desc.dtype, desc.shape) for name, desc in check.tensors.items()
            }
            if actual != expected:
                raise ConversionError(
                    f"{tmp}: post-write header does not match conversion plan"
                )
        os.replace(tmp, destination)
        _fsync_dir(destination.parent)
        return {
            "name": destination.name,
            "bytes": expected_size,
            "sha256": digest.hexdigest(),
            "tensor_count": len(plans),
        }
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def write_synthetic_safetensors(
    path: Path,
    tensors: Mapping[str, tuple[str, np.ndarray]],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Small test helper that preserves explicit safetensors dtype codes."""

    plans: list[tuple[str, str, np.ndarray]] = []
    cursor = 0
    header: OrderedDict[str, dict] = OrderedDict()
    header["__metadata__"] = dict(metadata or {"format": "mlx"})
    for name, (dtype, array) in tensors.items():
        if dtype not in DTYPES:
            raise ValueError(dtype)
        expected = DTYPES[dtype][0]
        data = np.ascontiguousarray(array, dtype=expected)
        end = cursor + data.nbytes
        header[name] = {
            "dtype": dtype,
            "shape": list(data.shape),
            "data_offsets": [cursor, end],
        }
        plans.append((name, dtype, data))
        cursor = end
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for _name, _dtype, data in plans:
            out.write(memoryview(data).cast("B"))


class KimiK3ShardingContract:
    """Name-based form of Kimi K3 ``Model.shard`` at the pinned commit."""

    LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")

    def __init__(self, config: Mapping, world_size: int = DEFAULT_WORLD_SIZE):
        if config.get("model_type") != "kimi_k3":
            raise ConversionError(
                f"expected model_type='kimi_k3', got {config.get('model_type')!r}"
            )
        text = config.get("text_config", config)
        self.text = text
        self.world_size = world_size
        linear = text["linear_attn_config"]
        self.kda_layers = {int(x) - 1 for x in linear["kda_layers"]}
        self.use_full_rank_gate = bool(linear.get("use_full_rank_gate", False))
        self.q_lora_rank = text.get("q_lora_rank")
        self.mla_use_output_gate = bool(text.get("mla_use_output_gate", False))
        self.num_experts = int(text.get("num_experts") or 0)
        self.first_dense = int(text.get("first_k_dense_replace", 0))
        self.moe_freq = int(text.get("moe_layer_freq", 1))
        self.num_layers = int(text["num_hidden_layers"])

        if world_size < 1:
            raise ConversionError("world_size must be positive")
        if self.num_layers != 93 and config.get("_synthetic_test") is not True:
            raise ConversionError(
                f"pinned checkpoint should have 93 layers, got {self.num_layers}"
            )

    def is_moe(self, layer: int) -> bool:
        return (
            self.num_experts > 0
            and layer >= self.first_dense
            and layer % self.moe_freq == 0
        )

    @staticmethod
    def _a2s(reason: str, segments: int = 1) -> ShardRule:
        return ShardRule("all-to-sharded", None, segments, reason)

    @staticmethod
    def _s2a(reason: str) -> ShardRule:
        return ShardRule("sharded-to-all", None, 1, reason)

    @staticmethod
    def _axis(axis: int, reason: str, segments: int = 1) -> ShardRule:
        return ShardRule("axis", axis, segments, reason)

    @staticmethod
    def _rep(reason: str = "replicated by Model.shard") -> ShardRule:
        return ShardRule("replicated", None, 1, reason)

    def classify(self, name: str, ndim: int) -> ShardRule:
        if not name.startswith(TEXT_PREFIX):
            return ShardRule("excluded", None, 1, "text-only EXO deployment")
        match = self.LAYER_RE.match(name)
        if match is None:
            return self._rep("global text tensor is not sharded")
        layer = int(match.group(1))
        suffix = match.group(2)
        if not (0 <= layer < self.num_layers):
            raise ConversionError(f"{name}: layer index outside configured model")

        if layer in self.kda_layers:
            if suffix.startswith("self_attn.qkv_proj."):
                return self._a2s("KDA fused QKV output", segments=3)
            for module in (
                "self_attn.f_b_proj.",
                "self_attn.b_proj.",
                "self_attn.g_proj."
                if self.use_full_rank_gate
                else "self_attn.g_b_proj.",
            ):
                if suffix.startswith(module):
                    return self._a2s(f"KDA {module[:-1]} output")
            if suffix.startswith("self_attn.o_proj."):
                return self._s2a("KDA output projection input")
            if suffix == "self_attn.qkv_conv.conv.weight":
                return self._axis(0, "KDA fused QKV depthwise convolution", 3)
            if suffix == "self_attn.A_log":
                return self._axis(0, "KDA heads")
            if suffix == "self_attn.dt_bias":
                return self._axis(0, "KDA projected heads")
        else:
            q_module = (
                "self_attn.q_b_proj."
                if self.q_lora_rank is not None
                else "self_attn.q_proj."
            )
            if suffix.startswith(q_module):
                return self._a2s("MLA query heads")
            if self.mla_use_output_gate and suffix.startswith("self_attn.g_proj."):
                return self._a2s("MLA output gate heads")
            if suffix.startswith("self_attn.o_proj."):
                return self._s2a("MLA output projection input")
            if suffix.startswith(("self_attn.embed_q.", "self_attn.unembed_out.")):
                return self._axis(0, "MLA per-head MultiLinear")

        if self.is_moe(layer):
            if suffix.startswith(
                (
                    "mlp.switch_mlp.gate_proj.",
                    "mlp.switch_mlp.up_proj.",
                    "mlp.shared_experts.gate_proj.",
                    "mlp.shared_experts.up_proj.",
                )
            ):
                return self._a2s("MoE/shared expert intermediate output")
            if suffix.startswith(
                (
                    "mlp.switch_mlp.down_proj.",
                    "mlp.shared_experts.down_proj.",
                )
            ):
                return self._s2a("MoE/shared expert down-projection input")
        else:
            if suffix.startswith(("mlp.gate_proj.", "mlp.up_proj.")):
                return self._a2s("dense MLP intermediate output")
            if suffix.startswith("mlp.down_proj."):
                return self._s2a("dense MLP down-projection input")

        return self._rep()

    @staticmethod
    def resolve_axis(rule: ShardRule, name: str, ndim: int) -> int | None:
        if rule.kind in ("replicated", "excluded"):
            return None
        if ndim == 0:
            raise ConversionError(f"{name}: cannot shard a scalar")
        if rule.kind == "axis":
            axis = int(rule.axis)
        elif rule.kind == "all-to-sharded":
            # Exact mlx.nn.layers.distributed._all_to_sharded behavior.
            axis = -1 if name.endswith(".bias") else max(ndim - 2, 0)
        elif rule.kind == "sharded-to-all":
            # A true affine bias is replicated.  Quantization's ``biases``
            # tensor is intentionally split with weight/scales.
            if name.endswith(".bias"):
                return None
            axis = -1
        else:
            raise ConversionError(f"{name}: unknown rule {rule.kind!r}")
        if axis < 0:
            axis += ndim
        if not 0 <= axis < ndim:
            raise ConversionError(f"{name}: axis {axis} outside ndim={ndim}")
        return axis

    def plan(self, desc: TensorDesc, rank: int) -> TensorPlan | None:
        if not 0 <= rank < self.world_size:
            raise ConversionError(f"invalid rank {rank}/{self.world_size}")
        rule = self.classify(desc.name, len(desc.shape))
        if rule.kind == "excluded":
            return None
        axis = self.resolve_axis(rule, desc.name, len(desc.shape))
        if axis is None:
            return TensorPlan(desc, desc.shape, rule, (), None)

        dim = desc.shape[axis]
        if dim % rule.segments:
            raise ConversionError(
                f"{desc.name}: dimension {dim} on axis {axis} is not divisible "
                f"by segments={rule.segments}"
            )
        segment_size = dim // rule.segments
        if segment_size % self.world_size:
            raise ConversionError(
                f"{desc.name}: segment size {segment_size} is not divisible "
                f"by TP world_size={self.world_size}"
            )
        local = segment_size // self.world_size
        intervals = tuple(
            (
                segment * segment_size + rank * local,
                segment * segment_size + (rank + 1) * local,
            )
            for segment in range(rule.segments)
        )
        output_shape = list(desc.shape)
        output_shape[axis] = local * rule.segments
        return TensorPlan(desc, tuple(output_shape), rule, intervals, axis)

    def contract_digest(self) -> str:
        source = {
            "version": CONTRACT_VERSION,
            "world_size": self.world_size,
            "kda_layers": sorted(self.kda_layers),
            "use_full_rank_gate": self.use_full_rank_gate,
            "q_lora_rank": self.q_lora_rank,
            "mla_use_output_gate": self.mla_use_output_gate,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "first_dense": self.first_dense,
            "moe_freq": self.moe_freq,
        }
        return _sha256_bytes(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        )


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
    except Exception as exc:
        raise ConversionError(f"cannot load {path}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"{path}: expected JSON object")
    return value


def _safe_repo_relative_name(name: str) -> str:
    """Reject absolute/traversal names before any path join or unlink."""

    if not name or "\\" in name:
        raise ConversionError(f"unsafe repository filename {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ConversionError(f"unsafe repository filename {name!r}")
    if Path(name).is_absolute():
        raise ConversionError(f"unsafe repository filename {name!r}")
    return name


def _safe_checkpoint_basename(name: object, field: str) -> str:
    """Return a journal-controlled output basename or fail closed."""

    if not isinstance(name, str):
        raise ConversionError(f"existing journal {field} must be a string")
    _safe_repo_relative_name(name)
    if "/" in name or Path(name).name != name:
        raise ConversionError(
            f"existing journal {field} must be a safe checkpoint basename"
        )
    return name


def _journal_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConversionError(
            f"existing journal {field} must be an integer >= {minimum}"
        )
    return value


def _journal_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ConversionError(
            f"existing journal {field} must be a lowercase SHA-256 digest"
        )
    return value


def _validate_journal_source_record(value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise ConversionError(f"existing journal {field} must be an object")
    _journal_int(value.get("bytes"), f"{field}.bytes")
    _journal_sha256(value.get("sha256"), f"{field}.sha256")
    url = value.get("url")
    if url is not None and not isinstance(url, str):
        raise ConversionError(f"existing journal {field}.url must be a string or null")
    if type(value.get("resumed")) is not bool:
        raise ConversionError(f"existing journal {field}.resumed must be a boolean")


def _validate_journal_rank_record(
    value: object,
    *,
    filename: str,
    indexed_keys: set[str],
    contract: KimiK3ShardingContract,
    rank: int,
    field: str,
) -> tuple[TensorPlan, ...]:
    if not isinstance(value, dict):
        raise ConversionError(f"existing journal {field} must be an object")
    record_name = _safe_checkpoint_basename(value.get("name"), f"{field}.name")
    if record_name != filename:
        raise ConversionError(
            f"existing journal {field}.name does not match its file key"
        )
    _journal_int(value.get("bytes"), f"{field}.bytes", minimum=1)
    _journal_sha256(value.get("sha256"), f"{field}.sha256")
    tensor_count = _journal_int(
        value.get("tensor_count"), f"{field}.tensor_count", minimum=1
    )
    tensors = value.get("tensors")
    if not isinstance(tensors, dict):
        raise ConversionError(f"existing journal {field}.tensors must be an object")

    expected_keys = {
        name for name in indexed_keys if contract.classify(name, 1).kind != "excluded"
    }
    if set(tensors) != expected_keys:
        raise ConversionError(
            f"existing journal {field}.tensors does not match the pinned weight index"
        )
    if tensor_count != len(tensors):
        raise ConversionError(
            f"existing journal {field}.tensor_count does not match its tensor map"
        )

    plans: list[TensorPlan] = []
    for tensor_name, tensor_value in tensors.items():
        tensor_field = f"{field}.tensors[{tensor_name!r}]"
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ConversionError(
                f"existing journal {field}.tensors keys must be non-empty strings"
            )
        if not isinstance(tensor_value, dict):
            raise ConversionError(f"existing journal {tensor_field} must be an object")
        dtype = tensor_value.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise ConversionError(
                f"existing journal {tensor_field}.dtype is unsupported"
            )
        source_shape = tensor_value.get("source_shape")
        if (
            not isinstance(source_shape, list)
            or not source_shape
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in source_shape
            )
        ):
            raise ConversionError(
                f"existing journal {tensor_field}.source_shape is invalid"
            )
        desc = TensorDesc(
            tensor_name,
            dtype,
            tuple(source_shape),
            0,
            math.prod(source_shape) * DTYPES[dtype][1],
        )
        plan = contract.plan(desc, rank)
        if plan is None or tensor_value != _tensor_manifest(plan, filename):
            raise ConversionError(
                f"existing journal {tensor_field} does not match the TP contract"
            )
        plans.append(plan)
    return tuple(plans)


def _validate_resume_journal(
    journal: dict,
    *,
    config_sha256: str,
    index_sha256: str,
    world_size: int,
    contract: KimiK3ShardingContract,
    keys_by_file: Mapping[str, set[str]],
) -> dict:
    """Authenticate every field later trusted by source-free resume.

    Version 1 journals used the same committed-shard record layout.  They are
    deliberately accepted, fully revalidated, and marked v2 so the normal
    final journal write completes the metadata-only migration.
    """

    schema = journal.get("schema")
    if schema not in JOURNAL_SCHEMAS:
        raise ConversionError(f"existing journal has unsupported schema {schema!r}")
    expected_scalars: tuple[tuple[str, object], ...] = (
        ("source_repo", SOURCE_REPO),
        ("source_revision", SOURCE_REVISION),
        ("config_sha256", config_sha256),
        ("index_sha256", index_sha256),
        ("world_size", world_size),
        ("contract_digest", contract.contract_digest()),
    )
    for field, expected in expected_scalars:
        value = journal.get(field)
        if type(value) is not type(expected) or value != expected:
            raise ConversionError(
                f"existing journal {field} does not match the pinned conversion"
            )
    if type(journal.get("complete")) is not bool:
        raise ConversionError("existing journal complete must be a boolean")
    if journal["complete"] and not isinstance(journal.get("completed_at"), str):
        raise ConversionError(
            "existing completed journal must contain a completed_at string"
        )

    files = journal.get("files")
    if not isinstance(files, dict):
        raise ConversionError("existing journal files must be an object")
    known_files = set(keys_by_file)
    for filename_value, entry in files.items():
        filename = _safe_checkpoint_basename(filename_value, "files key")
        if filename not in known_files:
            raise ConversionError(
                f"existing journal contains unindexed checkpoint file {filename!r}"
            )
        field = f"files[{filename!r}]"
        if not isinstance(entry, dict):
            raise ConversionError(f"existing journal {field} must be an object")
        committed = entry.get("committed")
        excluded = entry.get("excluded")
        if type(committed) is not bool or type(excluded) is not bool:
            raise ConversionError(
                f"existing journal {field} committed/excluded must be booleans"
            )
        _validate_journal_source_record(entry.get("source"), f"{field}.source")
        ranks = entry.get("ranks")
        if not isinstance(ranks, dict):
            raise ConversionError(f"existing journal {field}.ranks must be an object")
        if not committed:
            if ranks:
                raise ConversionError(
                    f"existing uncommitted journal {field} cannot contain rank records"
                )
            continue

        indexed_keys = keys_by_file[filename]
        expected_tensor_keys = {
            name
            for name in indexed_keys
            if contract.classify(name, 1).kind != "excluded"
        }
        if excluded != (not expected_tensor_keys):
            raise ConversionError(
                f"existing journal {field}.excluded disagrees with the TP contract"
            )
        expected_rank_keys = (
            set() if excluded else {str(rank) for rank in range(world_size)}
        )
        if set(ranks) != expected_rank_keys:
            raise ConversionError(
                f"existing journal {field}.ranks does not match TP world size"
            )
        for rank_key, record in ranks.items():
            if (
                not isinstance(rank_key, str)
                or not rank_key.isascii()
                or not rank_key.isdecimal()
            ):
                raise ConversionError(
                    f"existing journal {field}.ranks keys must be canonical rank strings"
                )
            rank = int(rank_key)
            if str(rank) != rank_key:
                raise ConversionError(
                    f"existing journal {field}.ranks keys must be canonical rank strings"
                )
            _validate_journal_rank_record(
                record,
                filename=filename,
                indexed_keys=indexed_keys,
                contract=contract,
                rank=rank,
                field=f"{field}.ranks[{rank_key!r}]",
            )

    journal["schema"] = SCHEMA
    return journal


def _validate_pinned_metadata_objects(
    config: dict, index: dict, source: str = "metadata"
) -> tuple[dict, dict]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConversionError(f"{source}: missing weight_map")
    metadata = index.get("metadata", {})
    if int(metadata.get("total_size", -1)) != 816_773_159_296:
        raise ConversionError(
            "weight index total_size does not match pinned UVMAX checkpoint"
        )
    if int(metadata.get("total_parameters", -1)) != 2_779_483_539_072:
        raise ConversionError(
            "weight index total_parameters does not match pinned UVMAX checkpoint"
        )
    return config, index


def _validate_pinned_metadata(config_path: Path, index_path: Path) -> tuple[dict, dict]:
    config_sha = _sha256_file(config_path)
    index_sha = _sha256_file(index_path)
    if config_sha != SOURCE_CONFIG_SHA256:
        raise ConversionError(
            f"{config_path}: expected pinned SHA-256 {SOURCE_CONFIG_SHA256}, "
            f"got {config_sha}"
        )
    if index_sha != SOURCE_INDEX_SHA256:
        raise ConversionError(
            f"{index_path}: expected pinned SHA-256 {SOURCE_INDEX_SHA256}, "
            f"got {index_sha}"
        )
    return _validate_pinned_metadata_objects(
        _load_json(config_path),
        _load_json(index_path),
        f"{config_path} / {index_path}",
    )


def _download_url(repo: str, revision: str, filename: str) -> str:
    safe_repo = "/".join(urllib.parse.quote(x, safe="") for x in repo.split("/"))
    safe_file = "/".join(urllib.parse.quote(x, safe="") for x in filename.split("/"))
    return f"https://huggingface.co/{safe_repo}/resolve/{revision}/{safe_file}"


def download_resumable(
    url: str,
    ready_path: Path,
    *,
    chunk_size: int = 8 << 20,
) -> dict:
    """Download to a pinned temporary path, resuming with HTTP Range."""

    ready_path.parent.mkdir(parents=True, exist_ok=True)
    partial = ready_path.with_name(f".{ready_path.name}.download.partial")
    if ready_path.exists():
        return {
            "bytes": ready_path.stat().st_size,
            "sha256": _sha256_file(ready_path),
            "url": url,
            "resumed": True,
        }

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "k3-rank-local-tp/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if offset and exc.code == 416:
            os.replace(partial, ready_path)
            return {
                "bytes": ready_path.stat().st_size,
                "sha256": _sha256_file(ready_path),
                "url": url,
                "resumed": True,
            }
        raise

    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        partial.unlink()
        offset = 0
        response.close()
        response = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": headers["User-Agent"]}),
            timeout=120,
        )
    mode = "ab" if offset else "wb"
    digest = hashlib.sha256()
    if offset:
        with partial.open("rb") as existing:
            while chunk := existing.read(chunk_size):
                digest.update(chunk)
    with response, partial.open(mode) as out:
        while chunk := response.read(chunk_size):
            out.write(chunk)
            digest.update(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(partial, ready_path)
    _fsync_dir(ready_path.parent)
    return {
        "bytes": ready_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "url": url,
        "resumed": bool(offset),
    }


def _open_regular_readonly(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ConversionError(
            f"{path}: cannot safely open metadata file: {exc}"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ConversionError(f"{path}: metadata entry must be a regular file")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _sha256_regular_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    fd = _open_regular_readonly(path)
    with os.fdopen(fd, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_metadata_file(
    src: Path,
    dst: Path,
    expected_sha256: str,
    chunk_size: int = 8 << 20,
) -> Path:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dst.name}.",
        suffix=".metadata.partial",
        dir=dst.parent,
    )
    tmp = Path(tmp_name)
    src_fd: int | None = None
    digest = hashlib.sha256()
    try:
        src_fd = _open_regular_readonly(src)
        with os.fdopen(src_fd, "rb") as source, os.fdopen(fd, "wb") as output:
            src_fd = None
            while chunk := source.read(chunk_size):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), 0o644)
        if digest.hexdigest() != expected_sha256:
            raise ConversionError(f"{src}: metadata changed while it was being copied")
        return tmp
    except BaseException:
        if src_fd is not None:
            os.close(src_fd)
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _copy_metadata_files(
    metadata_dir: Path,
    rank_dirs: Sequence[Path],
) -> dict[str, dict[str, int | str]]:
    metadata_dir = Path(metadata_dir)
    if metadata_dir.is_symlink() or not metadata_dir.is_dir():
        raise ConversionError(
            f"{metadata_dir}: metadata directory must be a real directory"
        )

    sources: dict[str, Path] = {}
    for src in metadata_dir.iterdir():
        if src.is_symlink():
            raise ConversionError(f"{src}: metadata symlinks are not permitted")
        if src.name not in PINNED_METADATA_FILES:
            continue
        if not src.is_file():
            raise ConversionError(f"{src}: allowlisted metadata must be a regular file")
        sources[src.name] = src

    missing = sorted(set(PINNED_METADATA_FILES) - sources.keys())
    if missing:
        raise ConversionError(
            f"{metadata_dir}: missing required metadata: {', '.join(missing)}"
        )
    source_hashes: dict[str, str] = {}
    for name, src in sorted(sources.items()):
        identity = _regular_file_identity(src)
        expected = PINNED_METADATA_FILES[name]
        if identity[3] != expected["bytes"]:
            raise ConversionError(
                f"{src}: metadata size differs from the pinned source"
            )
        source_hashes[name] = _sha256_regular_file(src)
        if source_hashes[name] != expected["sha256"]:
            raise ConversionError(
                f"{src}: metadata checksum differs from the pinned source"
            )
    destinations: list[tuple[Path, Path, str]] = []
    roots = [Path(root) for root in rank_dirs]
    for root in roots:
        if root.is_symlink():
            raise ConversionError(f"{root}: rank directory cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ConversionError(f"{root}: rank destination must be a directory")
        for name, src in sorted(sources.items()):
            dst = root / name
            if dst.is_symlink():
                raise ConversionError(f"{dst}: refusing metadata symlink destination")
            if dst.exists():
                if not dst.is_file():
                    raise ConversionError(
                        f"{dst}: metadata destination must be a regular file"
                    )
                if _sha256_regular_file(dst) != source_hashes[name]:
                    raise ConversionError(
                        f"{dst}: refusing to overwrite different metadata"
                    )
                continue
            destinations.append((src, dst, source_hashes[name]))

    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for src, dst, expected_sha256 in destinations:
            staged.append((_stage_metadata_file(src, dst, expected_sha256), dst))
        for tmp, dst in staged:
            try:
                os.link(tmp, dst)
            except FileExistsError as exc:
                raise ConversionError(
                    f"{dst}: refusing to overwrite metadata created concurrently"
                ) from exc
            published.append((tmp, dst))
            _fsync_dir(dst.parent)
    except BaseException:
        for tmp, dst in reversed(published):
            with contextlib.suppress(OSError):
                if os.path.samefile(tmp, dst):
                    dst.unlink()
                    _fsync_dir(dst.parent)
        raise
    finally:
        for tmp, _ in staged:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        for root in roots:
            with contextlib.suppress(OSError):
                _fsync_dir(root)

    records: dict[str, dict[str, int | str]] = {}
    for name, expected_sha256 in sorted(source_hashes.items()):
        sizes = {(root / name).stat().st_size for root in roots}
        hashes = {_sha256_regular_file(root / name) for root in roots}
        if len(sizes) != 1 or hashes != {expected_sha256}:
            raise ConversionError(
                f"rank metadata copies diverged after publication: {name}"
            )
        records[name] = {"bytes": sizes.pop(), "sha256": expected_sha256}
    if records != PINNED_METADATA_FILES:
        raise ConversionError("rank metadata contract differs from the pinned source")
    return records


def _existing_output_valid(
    path: Path,
    expected: Sequence[TensorPlan],
    rank: int,
    world_size: int,
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with SafeTensorFile(path) as safe:
            if safe.metadata.get("source_revision") != SOURCE_REVISION:
                return False
            if safe.metadata.get("tp_rank") != str(rank):
                return False
            if safe.metadata.get("tp_world_size") != str(world_size):
                return False
            expected_headers = {
                p.source.name: (p.source.dtype, p.output_shape) for p in expected
            }
            actual_headers = {k: (v.dtype, v.shape) for k, v in safe.tensors.items()}
            return expected_headers == actual_headers
    except (OSError, ConversionError):
        return False


def _tensor_manifest(plan: TensorPlan, source_file: str) -> dict:
    return {
        "source_file": source_file,
        "dtype": plan.source.dtype,
        "source_shape": list(plan.source.shape),
        "rank_shape": list(plan.output_shape),
        "rule": plan.rule.canonical(),
        "resolved_axis": plan.axis,
        "intervals": [list(x) for x in plan.intervals],
        "bytes": plan.output_nbytes,
    }


def _upgrade_object(
    value: object,
    expected_fields: set[str],
    field: str,
) -> dict:
    if not isinstance(value, dict):
        raise ConversionError(f"legacy manifest {field} must be an object")
    if set(value) != expected_fields:
        raise ConversionError(
            f"legacy manifest {field} fields differ from the v1 contract: "
            f"extra={sorted(set(value) - expected_fields)}, "
            f"missing={sorted(expected_fields - set(value))}"
        )
    return value


def _same_json_contract(left: object, right: object) -> bool:
    """Compare JSON contracts without Python's bool/int equality coercion."""

    options = {
        "sort_keys": True,
        "separators": (",", ":"),
        "ensure_ascii": True,
        "allow_nan": False,
    }
    try:
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def _upgrade_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConversionError(
            f"legacy manifest {field} must be an integer >= {minimum}"
        )
    return value


def _upgrade_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ConversionError(
            f"legacy manifest {field} must be a lowercase SHA-256 digest"
        )
    return value


def _upgrade_basename(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ConversionError(f"legacy manifest {field} must be a string")
    _safe_repo_relative_name(value)
    if "/" in value or Path(value).name != value:
        raise ConversionError(
            f"legacy manifest {field} must be a safe checkpoint basename"
        )
    return value


def _regular_file_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ConversionError(f"{path}: missing checkpoint file") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ConversionError(f"{path}: checkpoint entry must be a regular file")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ConversionError(f"{path}: missing directory") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ConversionError(f"{path}: expected a real directory")
    return (info.st_dev, info.st_ino)


def _resolved_real_directory(path: Path, field: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ConversionError(f"{path}: {field} must be a real directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ConversionError(f"{path}: cannot resolve {field}") from exc
    _directory_identity(resolved)
    return resolved


def _read_regular_bytes(path: Path) -> bytes:
    fd = _open_regular_readonly(path)
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


def _load_regular_json(path: Path) -> dict:
    try:
        raw = _read_regular_bytes(path)
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"cannot load regular JSON file {path}") from exc
    return _load_json_bytes(raw, str(path))


def _load_json_bytes(raw: bytes, source: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ConversionError(f"cannot load JSON object from {source}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"{source}: expected JSON object")
    return value


def _validate_legacy_root_contract(manifest: dict, rank: int) -> None:
    _upgrade_object(
        manifest,
        {
            "schema",
            "complete",
            "source",
            "runtime",
            "tp",
            "rank_data_bytes",
            "files",
            "tensors",
        },
        f"rank {rank}",
    )
    if manifest["schema"] != LEGACY_SCHEMA:
        raise ConversionError(
            f"rank {rank} manifest must use {LEGACY_SCHEMA!r}, "
            f"got {manifest['schema']!r}"
        )
    if manifest["complete"] is not True:
        raise ConversionError(f"rank {rank} legacy manifest is incomplete")

    source = _upgrade_object(
        manifest["source"],
        {
            "repo",
            "revision",
            "config_sha256",
            "index_sha256",
            "total_size",
            "total_parameters",
        },
        f"rank {rank}.source",
    )
    expected_source = {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "config_sha256": SOURCE_CONFIG_SHA256,
        "index_sha256": SOURCE_INDEX_SHA256,
        "total_size": 816_773_159_296,
        "total_parameters": 2_779_483_539_072,
    }
    if not _same_json_contract(source, expected_source):
        raise ConversionError(
            f"rank {rank} legacy source contract does not match the pinned checkpoint"
        )

    runtime = _upgrade_object(
        manifest["runtime"],
        {"mlx_lm_pr", "mlx_lm_commit", "mlx_lm_kimi_k3_sha256"},
        f"rank {rank}.runtime",
    )
    expected_runtime = {
        "mlx_lm_pr": MLX_LM_PR,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "mlx_lm_kimi_k3_sha256": MLX_LM_KIMI_K3_SHA256,
    }
    if not _same_json_contract(runtime, expected_runtime):
        raise ConversionError(
            f"rank {rank} legacy runtime contract does not match the converter"
        )

    tp = _upgrade_object(
        manifest["tp"],
        {"rank", "world_size", "contract", "contract_digest"},
        f"rank {rank}.tp",
    )
    tp_rank = _upgrade_int(tp.get("rank"), f"rank {rank}.tp.rank")
    if tp_rank != rank:
        raise ConversionError(
            f"rank {rank} root contains manifest for rank {tp_rank!r}"
        )
    tp_world_size = _upgrade_int(
        tp.get("world_size"),
        f"rank {rank}.tp.world_size",
        minimum=1,
    )
    if tp_world_size != DEFAULT_WORLD_SIZE:
        raise ConversionError(f"rank {rank} manifest is not TP{DEFAULT_WORLD_SIZE}")
    if tp.get("contract") != CONTRACT_VERSION:
        raise ConversionError(f"rank {rank} manifest uses a different TP contract")
    _upgrade_sha256(tp.get("contract_digest"), f"rank {rank}.tp.contract_digest")
    _upgrade_int(manifest["rank_data_bytes"], f"rank {rank}.rank_data_bytes")
    if not isinstance(manifest["files"], dict) or not manifest["files"]:
        raise ConversionError(f"rank {rank} manifest files must be a non-empty object")
    if not isinstance(manifest["tensors"], dict) or not manifest["tensors"]:
        raise ConversionError(
            f"rank {rank} manifest tensors must be a non-empty object"
        )


def _validate_legacy_tensor_record(
    tensor_name: object,
    value: object,
    *,
    filename: str,
    contract: KimiK3ShardingContract,
    rank: int,
    field: str,
) -> TensorPlan:
    if not isinstance(tensor_name, str) or not tensor_name:
        raise ConversionError(f"legacy manifest {field} has an invalid tensor name")
    record = _upgrade_object(
        value,
        {
            "source_file",
            "dtype",
            "source_shape",
            "rank_shape",
            "rule",
            "resolved_axis",
            "intervals",
            "bytes",
        },
        field,
    )
    dtype = record.get("dtype")
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise ConversionError(f"legacy manifest {field}.dtype is unsupported")
    source_shape = record.get("source_shape")
    if (
        not isinstance(source_shape, list)
        or not source_shape
        or any(
            type(dimension) is not int or dimension <= 0 for dimension in source_shape
        )
    ):
        raise ConversionError(f"legacy manifest {field}.source_shape is invalid")
    desc = TensorDesc(
        tensor_name,
        dtype,
        tuple(source_shape),
        0,
        math.prod(source_shape) * DTYPES[dtype][1],
    )
    plan = contract.plan(desc, rank)
    if plan is None or not _same_json_contract(
        record,
        _tensor_manifest(plan, filename),
    ):
        raise ConversionError(
            f"legacy manifest {field} does not match the pinned TP contract"
        )
    return plan


def _validate_legacy_weight_file(
    path: Path,
    *,
    filename: str,
    record: dict,
    plans: Sequence[TensorPlan],
    rank: int,
) -> tuple[int, int, int, int, int, int]:
    identity_before = _regular_file_identity(path)
    expected_bytes = _upgrade_int(
        record.get("bytes"),
        f"rank {rank}.files[{filename!r}].bytes",
        minimum=1,
    )
    if identity_before[3] != expected_bytes:
        raise ConversionError(
            f"{path}: shard size {identity_before[3]} does not match legacy manifest "
            f"size {expected_bytes}"
        )
    expected_checksum = _upgrade_sha256(
        record.get("sha256"),
        f"rank {rank}.files[{filename!r}].sha256",
    )
    actual_checksum = _sha256_regular_file(path)
    if actual_checksum != expected_checksum:
        raise ConversionError(
            f"{path}: shard checksum does not match the legacy manifest"
        )

    expected_metadata = {
        "format": "mlx",
        "schema": LEGACY_SCHEMA,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file": filename,
        "tp_rank": str(rank),
        "tp_world_size": str(DEFAULT_WORLD_SIZE),
        "sharding_contract": CONTRACT_VERSION,
    }
    expected_header, _header = _build_header(plans, expected_metadata)
    canonical_size = (
        8 + len(expected_header) + sum(plan.output_nbytes for plan in plans)
    )
    if expected_bytes != canonical_size:
        raise ConversionError(
            f"{path}: shard size is not canonical for its safetensors header"
        )

    with SafeTensorFile(path) as shard:
        if shard.metadata != expected_metadata:
            raise ConversionError(
                f"{path}: safetensors metadata does not match the legacy TP contract"
            )
        expected_headers = {
            plan.source.name: (plan.source.dtype, plan.output_shape) for plan in plans
        }
        actual_headers = {
            name: (desc.dtype, desc.shape) for name, desc in shard.tensors.items()
        }
        if actual_headers != expected_headers:
            raise ConversionError(
                f"{path}: safetensors header does not match the legacy manifest"
            )
        cursor = shard.data_offset
        for desc in shard.tensors.values():
            if desc.data_start != cursor:
                raise ConversionError(f"{path}: safetensors payload contains a gap")
            cursor = desc.data_end
        if cursor != identity_before[3]:
            raise ConversionError(
                f"{path}: safetensors descriptors do not cover the physical payload"
            )

    identity_after = _regular_file_identity(path)
    if identity_after != identity_before:
        raise ConversionError(f"{path}: shard changed while it was being validated")
    return identity_after


def _validate_legacy_rank(
    root: Path,
    manifest: dict,
    contract: KimiK3ShardingContract,
    rank: int,
    expected_weight_map: Mapping[str, str],
) -> tuple[dict[str, tuple[int, int, int, int, int, int]], dict]:
    if manifest["tp"]["contract_digest"] != contract.contract_digest():
        raise ConversionError(
            f"rank {rank} manifest TP digest does not match its pinned config"
        )

    combined_tensors: dict[str, dict] = {}
    observed: dict[str, tuple[int, int, int, int, int, int]] = {}
    files = manifest["files"]
    expected_files = set(expected_weight_map.values())
    if set(files) != expected_files:
        raise ConversionError(
            f"rank {rank} shard inventory differs from the authenticated source index"
        )
    for raw_filename, raw_record in sorted(files.items()):
        filename = _upgrade_basename(
            raw_filename,
            f"rank {rank}.files key",
        )
        record = _upgrade_object(
            raw_record,
            {"name", "bytes", "sha256", "tensor_count", "tensors"},
            f"rank {rank}.files[{filename!r}]",
        )
        if (
            _upgrade_basename(
                record.get("name"),
                f"rank {rank}.files[{filename!r}].name",
            )
            != filename
        ):
            raise ConversionError(
                f"rank {rank} file record name differs from its manifest key"
            )
        tensors = record.get("tensors")
        if not isinstance(tensors, dict) or not tensors:
            raise ConversionError(
                f"rank {rank} file {filename!r} has no tensor records"
            )
        tensor_count = _upgrade_int(
            record.get("tensor_count"),
            f"rank {rank}.files[{filename!r}].tensor_count",
            minimum=1,
        )
        if tensor_count != len(tensors):
            raise ConversionError(
                f"rank {rank} file {filename!r} tensor_count is inconsistent"
            )
        plans: list[TensorPlan] = []
        for tensor_name, tensor_record in sorted(tensors.items()):
            if tensor_name in combined_tensors:
                raise ConversionError(
                    f"rank {rank} tensor {tensor_name!r} appears in multiple files"
                )
            plan = _validate_legacy_tensor_record(
                tensor_name,
                tensor_record,
                filename=filename,
                contract=contract,
                rank=rank,
                field=(f"rank {rank}.files[{filename!r}].tensors[{tensor_name!r}]"),
            )
            plans.append(plan)
            combined_tensors[tensor_name] = tensor_record
            if expected_weight_map.get(tensor_name) != filename:
                raise ConversionError(
                    f"rank {rank} tensor {tensor_name!r} source file differs from "
                    "the authenticated source index"
                )
        path = root / filename
        observed[str(path)] = _validate_legacy_weight_file(
            path,
            filename=filename,
            record=record,
            plans=plans,
            rank=rank,
        )

    if not _same_json_contract(manifest["tensors"], combined_tensors):
        raise ConversionError(
            f"rank {rank} top-level tensor map differs from its file records"
        )
    if set(combined_tensors) != set(expected_weight_map):
        raise ConversionError(
            f"rank {rank} tensor inventory differs from the authenticated source index"
        )
    expected_data_bytes = sum(
        int(tensor["bytes"]) for tensor in combined_tensors.values()
    )
    if manifest["rank_data_bytes"] != expected_data_bytes:
        raise ConversionError(
            f"rank {rank} rank_data_bytes differs from its tensor records"
        )

    index_path = root / "model.safetensors.index.json"
    observed[str(index_path)] = _regular_file_identity(index_path)
    index = _load_json_bytes(_read_regular_bytes(index_path), str(index_path))
    expected_index = {
        "metadata": {
            "total_size": expected_data_bytes,
            "source_total_parameters": manifest["source"]["total_parameters"],
            "tp_rank": rank,
            "tp_world_size": DEFAULT_WORLD_SIZE,
            "source_revision": SOURCE_REVISION,
        },
        "weight_map": {
            name: filename for name, filename in sorted(expected_weight_map.items())
        },
    }
    if not _same_json_contract(index, expected_index):
        raise ConversionError(
            f"{index_path}: rank-local weight index differs from the legacy manifest"
        )
    if _regular_file_identity(index_path) != observed[str(index_path)]:
        raise ConversionError(f"{index_path}: index changed while it was validated")
    return observed, combined_tensors


def _authenticated_rank_weight_map(
    source_index_path: Path,
    config: Mapping,
    contract: KimiK3ShardingContract,
) -> tuple[dict[str, str], tuple[int, int, int, int, int, int]]:
    """Return the complete non-excluded inventory from the pinned source index."""

    identity_before = _regular_file_identity(source_index_path)
    source_index_bytes = _read_regular_bytes(source_index_path)
    checksum = _sha256_bytes(source_index_bytes)
    if checksum != SOURCE_INDEX_SHA256:
        raise ConversionError(
            f"{source_index_path}: expected pinned SHA-256 {SOURCE_INDEX_SHA256}, "
            f"got {checksum}"
        )
    source_index = _load_json_bytes(source_index_bytes, str(source_index_path))
    _validate_pinned_metadata_objects(
        dict(config),
        source_index,
        f"authenticated source index {source_index_path}",
    )
    weight_map = source_index["weight_map"]
    expected: dict[str, str] = {}
    for raw_name, raw_filename in weight_map.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ConversionError(
                f"{source_index_path}: source index has an invalid tensor name"
            )
        filename = _upgrade_basename(
            raw_filename,
            f"authenticated source index weight_map[{raw_name!r}]",
        )
        if contract.classify(raw_name, 1).kind != "excluded":
            expected[raw_name] = filename
    if not expected:
        raise ConversionError(
            f"{source_index_path}: authenticated source index has no rank tensors"
        )
    identity_after = _regular_file_identity(source_index_path)
    if identity_after != identity_before:
        raise ConversionError(
            f"{source_index_path}: authenticated source index changed while read"
        )
    return expected, identity_after


def _validate_rank_symmetry(manifests: Sequence[dict]) -> None:
    rank0, rank1 = manifests
    if not _same_json_contract(rank0["source"], rank1["source"]):
        raise ConversionError("legacy rank source contracts are not symmetric")
    if not _same_json_contract(rank0["runtime"], rank1["runtime"]):
        raise ConversionError("legacy rank runtime contracts are not symmetric")
    if set(rank0["files"]) != set(rank1["files"]):
        raise ConversionError("legacy rank shard inventories are not symmetric")
    if set(rank0["tensors"]) != set(rank1["tensors"]):
        raise ConversionError("legacy rank tensor inventories are not symmetric")
    if rank0["rank_data_bytes"] != rank1["rank_data_bytes"]:
        raise ConversionError("legacy rank data sizes are not symmetric")

    common_tensor_fields = {
        "source_file",
        "dtype",
        "source_shape",
        "rule",
        "resolved_axis",
    }
    for name in rank0["tensors"]:
        left = rank0["tensors"][name]
        right = rank1["tensors"][name]
        if any(left[field] != right[field] for field in common_tensor_fields):
            raise ConversionError(
                f"legacy tensor {name!r} source contracts are not symmetric"
            )
    for filename in rank0["files"]:
        left = rank0["files"][filename]
        right = rank1["files"][filename]
        if left["tensor_count"] != right["tensor_count"]:
            raise ConversionError(
                f"legacy shard {filename!r} tensor counts are not symmetric"
            )
        if left["bytes"] != right["bytes"]:
            raise ConversionError(
                f"legacy shard {filename!r} file sizes are not symmetric"
            )
        if set(left["tensors"]) != set(right["tensors"]):
            raise ConversionError(
                f"legacy shard {filename!r} tensor inventories are not symmetric"
            )


def _validate_upgrade_metadata(
    rank_dirs: Sequence[Path],
    manifests: Sequence[dict],
) -> tuple[
    dict[str, dict[str, int | str]],
    dict[str, tuple[int, int, int, int, int, int]],
]:
    rank_records: list[dict[str, dict[str, int | str]]] = []
    observed: dict[str, tuple[int, int, int, int, int, int]] = {}
    for rank, (root, manifest) in enumerate(zip(rank_dirs, manifests, strict=True)):
        weight_filenames = set(manifest["files"])
        expected_non_metadata = weight_filenames | set(INTERNAL_CHECKPOINT_FILENAMES)
        actual_entries: set[str] = set()
        for child in root.iterdir():
            identity = _regular_file_identity(child)
            actual_entries.add(child.name)
            observed[str(child)] = identity
        metadata_names = actual_entries - expected_non_metadata
        missing_internal = expected_non_metadata - actual_entries
        if missing_internal:
            raise ConversionError(
                f"rank {rank} checkpoint is missing manifest-listed files: "
                f"{sorted(missing_internal)}"
            )
        expected_metadata_names = set(PINNED_METADATA_FILES)
        if metadata_names != expected_metadata_names:
            raise ConversionError(
                f"rank {rank} checkpoint metadata inventory differs from the "
                f"authenticated source: extra={sorted(metadata_names - expected_metadata_names)}, "
                f"missing={sorted(expected_metadata_names - metadata_names)}"
            )

        records: dict[str, dict[str, int | str]] = {}
        for name in sorted(metadata_names):
            path = root / name
            identity_before = observed[str(path)]
            expected_record = PINNED_METADATA_FILES[name]
            if identity_before[3] != expected_record["bytes"]:
                raise ConversionError(
                    f"{path}: metadata size differs from the authenticated source"
                )
            checksum = _sha256_regular_file(path)
            if checksum != expected_record["sha256"]:
                raise ConversionError(
                    f"{path}: metadata checksum differs from the authenticated source"
                )
            if _regular_file_identity(path) != identity_before:
                raise ConversionError(
                    f"{path}: metadata changed while it was being validated"
                )
            records[name] = {
                "bytes": identity_before[3],
                "sha256": checksum,
            }
        rank_records.append(records)

    if rank_records[0] != rank_records[1]:
        raise ConversionError("rank metadata files are not byte-identical")
    records = rank_records[0]
    if records != PINNED_METADATA_FILES:
        raise ConversionError("rank metadata contract differs from the pinned source")
    return records, observed


def _encode_json(payload: Mapping) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _pinned_metadata_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            PINNED_METADATA_FILES,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _deterministic_upgraded_manifest(original: Mapping) -> dict:
    upgraded = dict(original)
    upgraded["schema"] = SCHEMA
    upgraded["metadata_files"] = PINNED_METADATA_FILES
    upgraded["metadata_contract_sha256"] = _pinned_metadata_contract_sha256()
    return upgraded


def _stage_upgrade_payload(path: Path, payload: bytes, suffix: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=suffix,
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        return tmp
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _upgrade_transaction_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
        or value in {".", ".."}
    ):
        raise ConversionError(
            "manifest upgrade transaction ID must be a safe 1-128 character name"
        )
    return value


@contextlib.contextmanager
def _manifest_upgrade_lock(transaction_root: Path):
    lock_path = transaction_root / ".manifest-upgrade.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ConversionError(f"{lock_path}: cannot open upgrade lock") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConversionError(
                f"{transaction_root}: another manifest upgrade is active"
            ) from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_upgrade_artifact(path: Path, payload: bytes) -> dict[str, int | str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ConversionError(f"{path}: cannot create transaction artifact") from exc
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise
    return {
        "artifact": path.name,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _manifest_identity(path: Path) -> dict[str, int | str]:
    identity_before = _regular_file_identity(path)
    checksum = _sha256_regular_file(path)
    identity_after = _regular_file_identity(path)
    if identity_after != identity_before:
        raise ConversionError(f"{path}: manifest changed while it was hashed")
    return {"bytes": identity_after[3], "sha256": checksum}


def _create_manifest_upgrade_transaction(
    *,
    transaction_root: Path,
    transaction_id: str,
    roots: Sequence[Path],
    root_identities: Sequence[tuple[int, int]],
    source_index: Path,
    source_index_identity: tuple[int, int, int, int, int, int],
    originals: Sequence[bytes],
    upgraded: Sequence[bytes],
) -> tuple[Path, dict]:
    transaction_id = _upgrade_transaction_id(transaction_id)
    transaction = transaction_root / transaction_id
    try:
        transaction.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ConversionError(
            f"{transaction}: transaction already exists; recover it or choose a new ID"
        ) from exc
    created: list[Path] = []
    try:
        rank_records: list[dict] = []
        for rank, (root, root_identity, original, new) in enumerate(
            zip(roots, root_identities, originals, upgraded, strict=True)
        ):
            original_path = transaction / f"rank{rank}.original.json"
            upgraded_path = transaction / f"rank{rank}.upgraded.json"
            original_record = _write_upgrade_artifact(original_path, original)
            created.append(original_path)
            upgraded_record = _write_upgrade_artifact(upgraded_path, new)
            created.append(upgraded_path)
            rank_records.append(
                {
                    "rank": rank,
                    "root": str(root),
                    "root_identity": list(root_identity),
                    "manifest": str(root / "tp_manifest.json"),
                    "original": original_record,
                    "upgraded": upgraded_record,
                }
            )
        _fsync_dir(transaction)
        journal = {
            "schema": MANIFEST_UPGRADE_TRANSACTION_SCHEMA,
            "transaction_id": transaction_id,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "state": "prepared",
            "source_index": {
                "path": str(source_index),
                "sha256": SOURCE_INDEX_SHA256,
                "identity": list(source_index_identity),
            },
            "ranks": rank_records,
        }
        _atomic_json(transaction / "transaction.json", journal)
        _fsync_dir(transaction_root)
        return transaction, journal
    except BaseException:
        for path in reversed(created):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        with contextlib.suppress(FileNotFoundError):
            (transaction / "transaction.json").unlink()
        with contextlib.suppress(OSError):
            transaction.rmdir()
        raise


def _validate_transaction_artifact(
    transaction: Path,
    raw_record: object,
    field: str,
) -> tuple[Path, dict[str, int | str], bytes]:
    record = _upgrade_object(raw_record, {"artifact", "bytes", "sha256"}, field)
    artifact_name = _upgrade_basename(record.get("artifact"), f"{field}.artifact")
    byte_count = _upgrade_int(record.get("bytes"), f"{field}.bytes")
    checksum = _upgrade_sha256(record.get("sha256"), f"{field}.sha256")
    path = transaction / artifact_name
    identity_before = _regular_file_identity(path)
    payload = _read_regular_bytes(path)
    identity_after = _regular_file_identity(path)
    if (
        identity_after != identity_before
        or len(payload) != byte_count
        or _sha256_bytes(payload) != checksum
    ):
        raise ConversionError(f"{path}: transaction artifact identity mismatch")
    return (
        path,
        {"artifact": artifact_name, "bytes": byte_count, "sha256": checksum},
        payload,
    )


def _load_manifest_upgrade_transaction(transaction: Path) -> tuple[dict, list[dict]]:
    transaction = _resolved_real_directory(transaction, "transaction directory")
    journal = _load_regular_json(transaction / "transaction.json")
    _upgrade_object(
        journal,
        {"schema", "transaction_id", "created_at", "state", "source_index", "ranks"},
        "upgrade transaction",
    )
    if journal.get("schema") != MANIFEST_UPGRADE_TRANSACTION_SCHEMA:
        raise ConversionError(f"{transaction}: unsupported transaction schema")
    if _upgrade_transaction_id(journal.get("transaction_id")) != transaction.name:
        raise ConversionError(f"{transaction}: transaction ID differs from its path")
    if not isinstance(journal.get("created_at"), str) or not isinstance(
        journal.get("state"), str
    ):
        raise ConversionError(f"{transaction}: malformed transaction state")
    if journal["state"] not in MANIFEST_UPGRADE_TRANSACTION_STATES:
        raise ConversionError(
            f"{transaction}: unrecognized transaction state {journal['state']!r}"
        )
    source = _upgrade_object(
        journal.get("source_index"),
        {"path", "sha256", "identity"},
        "upgrade transaction.source_index",
    )
    if source.get("sha256") != SOURCE_INDEX_SHA256:
        raise ConversionError(f"{transaction}: source index pin differs from runtime")
    if (
        not isinstance(source.get("path"), str)
        or not Path(source["path"]).is_absolute()
    ):
        raise ConversionError(f"{transaction}: source index path is not absolute")
    source_identity = source.get("identity")
    if (
        not isinstance(source_identity, list)
        or len(source_identity) != 6
        or any(type(item) is not int for item in source_identity)
    ):
        raise ConversionError(f"{transaction}: malformed source index identity")

    raw_ranks = journal.get("ranks")
    if not isinstance(raw_ranks, list) or len(raw_ranks) != DEFAULT_WORLD_SIZE:
        raise ConversionError(f"{transaction}: malformed rank transaction records")
    ranks: list[dict] = []
    for expected_rank, raw_rank in enumerate(raw_ranks):
        field = f"upgrade transaction.ranks[{expected_rank}]"
        record = _upgrade_object(
            raw_rank,
            {"rank", "root", "root_identity", "manifest", "original", "upgraded"},
            field,
        )
        if record.get("rank") != expected_rank:
            raise ConversionError(f"{transaction}: transaction rank order is invalid")
        if (
            not isinstance(record.get("root"), str)
            or not Path(record["root"]).is_absolute()
        ):
            raise ConversionError(f"{transaction}: malformed rank root")
        root = _resolved_real_directory(Path(record["root"]), "rank root")
        root_identity = record.get("root_identity")
        if (
            not isinstance(root_identity, list)
            or len(root_identity) != 2
            or any(type(item) is not int for item in root_identity)
            or tuple(root_identity) != _directory_identity(root)
        ):
            raise ConversionError(f"{root}: rank root identity changed")
        manifest = root / "tp_manifest.json"
        if record.get("manifest") != str(manifest):
            raise ConversionError(f"{transaction}: malformed manifest path")
        original_path, original, original_payload = _validate_transaction_artifact(
            transaction, record.get("original"), f"{field}.original"
        )
        if original["sha256"] != LEGACY_MANIFEST_SHA256[expected_rank]:
            raise ConversionError(
                f"{original_path}: original rank-{expected_rank} manifest does not "
                "match the compiled legacy identity"
            )
        original_manifest = _load_json_bytes(original_payload, str(original_path))
        _validate_legacy_root_contract(original_manifest, expected_rank)
        expected_upgraded_manifest = _deterministic_upgraded_manifest(original_manifest)
        expected_upgraded_payload = _encode_json(expected_upgraded_manifest)
        upgraded_path, upgraded, upgraded_payload = _validate_transaction_artifact(
            transaction, record.get("upgraded"), f"{field}.upgraded"
        )
        if (
            upgraded["bytes"] != len(expected_upgraded_payload)
            or upgraded["sha256"] != _sha256_bytes(expected_upgraded_payload)
            or upgraded_payload != expected_upgraded_payload
        ):
            raise ConversionError(
                f"{upgraded_path}: upgraded artifact differs from the deterministic "
                "compiled v2 transformation"
            )
        ranks.append(
            {
                "rank": expected_rank,
                "root": root,
                "root_identity": tuple(root_identity),
                "manifest": manifest,
                "original_path": original_path,
                "original": original,
                "original_manifest": original_manifest,
                "upgraded_path": upgraded_path,
                "upgraded": upgraded,
                "upgraded_manifest": expected_upgraded_manifest,
            }
        )
    return journal, ranks


def _set_upgrade_transaction_state(
    transaction: Path,
    journal: dict,
    state: str,
) -> None:
    current = journal.get("state")
    if current not in MANIFEST_UPGRADE_TRANSACTION_STATES:
        raise ConversionError(f"{transaction}: current transaction state is invalid")
    if state not in MANIFEST_UPGRADE_STATE_TRANSITIONS[current]:
        raise ConversionError(
            f"{transaction}: invalid transaction state transition "
            f"{current!r} -> {state!r}"
        )
    journal["state"] = state
    _atomic_json(transaction / "transaction.json", journal)


def _publish_transaction_artifact(
    rank_record: Mapping,
    target: str,
) -> bool:
    root = rank_record["root"]
    if _directory_identity(root) != rank_record["root_identity"]:
        raise ConversionError(f"{root}: rank root identity changed before publication")
    manifest = rank_record["manifest"]
    desired = rank_record[target]
    other = rank_record["original" if target == "upgraded" else "upgraded"]
    current = _manifest_identity(manifest)
    if current == {"bytes": desired["bytes"], "sha256": desired["sha256"]}:
        return False
    if current != {"bytes": other["bytes"], "sha256": other["sha256"]}:
        raise ConversionError(
            f"{manifest}: current manifest is neither authenticated transaction state"
        )
    payload = _read_regular_bytes(rank_record[f"{target}_path"])
    if len(payload) != desired["bytes"] or _sha256_bytes(payload) != desired["sha256"]:
        raise ConversionError(
            f"{manifest}: transaction artifact changed before publish"
        )
    staged = _stage_upgrade_payload(manifest, payload, ".upgrade-transaction.partial")
    try:
        if _directory_identity(root) != rank_record["root_identity"]:
            raise ConversionError(f"{root}: rank root identity changed before replace")
        if _manifest_identity(manifest) != current:
            raise ConversionError(f"{manifest}: manifest changed before replace")
        os.replace(staged, manifest)
        _fsync_dir(root)
    finally:
        with contextlib.suppress(FileNotFoundError):
            staged.unlink()
    if _manifest_identity(manifest) != {
        "bytes": desired["bytes"],
        "sha256": desired["sha256"],
    }:
        raise ConversionError(f"{manifest}: published manifest identity mismatch")
    return True


def _preflight_transaction_states(ranks: Sequence[Mapping]) -> None:
    for record in ranks:
        current = _manifest_identity(record["manifest"])
        allowed = {
            (record["original"]["bytes"], record["original"]["sha256"]),
            (record["upgraded"]["bytes"], record["upgraded"]["sha256"]),
        }
        if (current["bytes"], current["sha256"]) not in allowed:
            raise ConversionError(
                f"{record['manifest']}: refusing to clobber an unrecognized manifest"
            )


def _validate_manifest_upgrade_checkpoint(
    journal: Mapping,
    ranks: Sequence[Mapping],
) -> None:
    """Full-hash every immutable input needed to complete a recovery."""

    roots = [record["root"] for record in ranks]
    manifests = [record["original_manifest"] for record in ranks]
    for record in ranks:
        if _directory_identity(record["root"]) != record["root_identity"]:
            raise ConversionError(
                f"{record['root']}: rank root identity changed during recovery"
            )

    source_path = Path(journal["source_index"]["path"])
    source_identity = tuple(journal["source_index"]["identity"])
    if (
        _regular_file_identity(source_path) != source_identity
        or _sha256_regular_file(source_path) != SOURCE_INDEX_SHA256
    ):
        raise ConversionError(
            f"{source_path}: authenticated source index is unavailable or changed"
        )

    config_paths = [root / "config.json" for root in roots]
    if [_sha256_regular_file(path) for path in config_paths] != [
        SOURCE_CONFIG_SHA256
    ] * DEFAULT_WORLD_SIZE:
        raise ConversionError("rank config.json files do not match the pinned source")
    configs = [_load_regular_json(path) for path in config_paths]
    if configs[0] != configs[1]:
        raise ConversionError("rank config.json files are not byte-equivalent JSON")
    contract = KimiK3ShardingContract(configs[0], DEFAULT_WORLD_SIZE)
    expected_weight_map, refreshed_source_identity = _authenticated_rank_weight_map(
        source_path,
        configs[0],
        contract,
    )
    if refreshed_source_identity != source_identity:
        raise ConversionError(
            f"{source_path}: authenticated source index identity changed"
        )

    for rank, (root, manifest) in enumerate(zip(roots, manifests, strict=True)):
        _validate_legacy_rank(
            root,
            manifest,
            contract,
            rank,
            expected_weight_map,
        )
    _validate_rank_symmetry(manifests)
    metadata_files, _observed = _validate_upgrade_metadata(roots, manifests)
    if metadata_files != PINNED_METADATA_FILES:
        raise ConversionError("rank metadata differs from the compiled source contract")


def _recover_manifest_upgrade_locked(
    transaction: Path,
    *,
    action: str,
) -> dict:
    if action not in {"rollback", "complete"}:
        raise ConversionError("manifest recovery action must be rollback or complete")
    journal, ranks = _load_manifest_upgrade_transaction(transaction)
    _preflight_transaction_states(ranks)
    if action == "complete":
        _validate_manifest_upgrade_checkpoint(journal, ranks)
    _set_upgrade_transaction_state(
        transaction,
        journal,
        f"recovering-{action}",
    )
    target = "upgraded" if action == "complete" else "original"
    ordered = ranks if action == "complete" else list(reversed(ranks))
    for record in ordered:
        _publish_transaction_artifact(record, target)
        _set_upgrade_transaction_state(
            transaction,
            journal,
            f"{action}-rank-{record['rank']}-durable",
        )
    if action == "complete":
        _validate_manifest_upgrade_checkpoint(journal, ranks)
    final_state = "committed" if action == "complete" else "rolled-back"
    _set_upgrade_transaction_state(transaction, journal, final_state)
    _preflight_transaction_states(ranks)
    for record in ranks:
        expected = record[target]
        if _manifest_identity(record["manifest"]) != {
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            raise ConversionError(
                f"{record['manifest']}: recovery did not reach the requested state"
            )
    return {
        "schema": MANIFEST_UPGRADE_SCHEMA,
        "transaction": str(transaction),
        "transaction_id": journal["transaction_id"],
        "action": action,
        "state": final_state,
    }


def recover_manifest_upgrade(*, transaction: Path, action: str) -> dict:
    transaction = _resolved_real_directory(transaction, "transaction directory")
    transaction_root = _resolved_real_directory(transaction.parent, "transaction root")
    with _manifest_upgrade_lock(transaction_root):
        return _recover_manifest_upgrade_locked(transaction, action=action)


def _upgrade_rank_local_manifests_locked(
    *,
    roots: Sequence[Path],
    source_index: Path,
    transaction_root: Path,
    transaction_id: str,
) -> dict:
    """Upgrade an already-complete TP2 v1 pair without source weights.

    The two legacy manifests are treated only as claims.  This routine proves
    the pinned source/runtime/TP contract, every rank shard checksum and header,
    the rank-local indexes, cross-rank symmetry, and the exact metadata
    allowlist before publishing either v2 manifest.
    """

    root_identities = [_directory_identity(root) for root in roots]

    manifest_paths = [root / "tp_manifest.json" for root in roots]
    manifest_identities = {
        str(path): _regular_file_identity(path) for path in manifest_paths
    }
    manifest_originals = [_read_regular_bytes(path) for path in manifest_paths]
    for rank, original in enumerate(manifest_originals):
        actual_sha256 = _sha256_bytes(original)
        if actual_sha256 != LEGACY_MANIFEST_SHA256[rank]:
            raise ConversionError(
                f"{manifest_paths[rank]}: legacy manifest SHA-256 differs from the "
                f"compiled rank-{rank} identity"
            )
    manifests = [
        _load_json_bytes(original, str(path))
        for path, original in zip(manifest_paths, manifest_originals, strict=True)
    ]
    for rank, manifest in enumerate(manifests):
        _validate_legacy_root_contract(manifest, rank)

    config_paths = [root / "config.json" for root in roots]
    config_hashes = [_sha256_regular_file(path) for path in config_paths]
    if config_hashes != [SOURCE_CONFIG_SHA256] * DEFAULT_WORLD_SIZE:
        raise ConversionError("rank config.json files do not match the pinned source")
    configs = [_load_regular_json(path) for path in config_paths]
    if configs[0] != configs[1]:
        raise ConversionError("rank config.json files are not byte-equivalent JSON")
    contract = KimiK3ShardingContract(configs[0], DEFAULT_WORLD_SIZE)
    expected_weight_map, source_index_identity = _authenticated_rank_weight_map(
        source_index,
        configs[0],
        contract,
    )

    observed: dict[str, tuple[int, int, int, int, int, int]] = {}
    for rank, (root, manifest) in enumerate(zip(roots, manifests, strict=True)):
        rank_observed, _tensors = _validate_legacy_rank(
            root,
            manifest,
            contract,
            rank,
            expected_weight_map,
        )
        observed.update(rank_observed)
    _validate_rank_symmetry(manifests)
    metadata_files, metadata_observed = _validate_upgrade_metadata(roots, manifests)
    for path, identity in metadata_observed.items():
        if path in observed and observed[path] != identity:
            raise ConversionError(f"{path}: checkpoint changed during validation")
        observed[path] = identity

    metadata_contract_sha256 = _pinned_metadata_contract_sha256()
    if metadata_files != PINNED_METADATA_FILES:
        raise ConversionError("validated metadata differs from the compiled contract")
    upgraded_manifests = [
        _deterministic_upgraded_manifest(manifest) for manifest in manifests
    ]

    refreshed_metadata, refreshed_observed = _validate_upgrade_metadata(
        roots,
        manifests,
    )
    if refreshed_metadata != metadata_files or refreshed_observed != observed:
        raise ConversionError("checkpoint inventory changed before publication")
    for path, expected in observed.items():
        if _regular_file_identity(Path(path)) != expected:
            raise ConversionError(f"{path}: checkpoint changed before publication")
    for path, expected in manifest_identities.items():
        if _regular_file_identity(Path(path)) != expected:
            raise ConversionError(f"{path}: manifest changed before publication")
    if (
        _regular_file_identity(source_index) != source_index_identity
        or _sha256_regular_file(source_index) != SOURCE_INDEX_SHA256
    ):
        raise ConversionError(
            f"{source_index}: authenticated source index changed before publication"
        )
    for root, expected in zip(roots, root_identities, strict=True):
        if _directory_identity(root) != expected:
            raise ConversionError(f"{root}: rank root changed before publication")

    payloads = [_encode_json(manifest) for manifest in upgraded_manifests]
    transaction, journal = _create_manifest_upgrade_transaction(
        transaction_root=transaction_root,
        transaction_id=transaction_id,
        roots=roots,
        root_identities=root_identities,
        source_index=source_index,
        source_index_identity=source_index_identity,
        originals=manifest_originals,
        upgraded=payloads,
    )
    try:
        journal, transaction_ranks = _load_manifest_upgrade_transaction(transaction)
        for record in transaction_ranks:
            if _manifest_identity(record["manifest"]) != {
                "bytes": record["original"]["bytes"],
                "sha256": record["original"]["sha256"],
            }:
                raise ConversionError(
                    f"{record['manifest']}: legacy manifest changed before publication"
                )
        for record in transaction_ranks:
            _publish_transaction_artifact(record, "upgraded")
            _set_upgrade_transaction_state(
                transaction,
                journal,
                f"rank-{record['rank']}-published",
            )
        _set_upgrade_transaction_state(transaction, journal, "committed")
    except Exception as exc:
        try:
            _recover_manifest_upgrade_locked(transaction, action="rollback")
        except Exception as rollback_exc:
            raise ConversionError(
                f"manifest upgrade failed and durable rollback also failed; "
                f"recover transaction {transaction}: {rollback_exc}"
            ) from exc
        raise

    for path, expected in zip(manifest_paths, upgraded_manifests, strict=True):
        if _load_regular_json(path) != expected:
            raise ConversionError(f"{path}: published v2 manifest failed validation")
    refreshed_metadata, post_publish_observed = _validate_upgrade_metadata(
        roots,
        manifests,
    )
    if refreshed_metadata != metadata_files:
        raise ConversionError("checkpoint metadata changed during publication")
    manifest_path_strings = {str(path) for path in manifest_paths}
    for path, expected in observed.items():
        if (
            path not in manifest_path_strings
            and post_publish_observed.get(path) != expected
        ):
            raise ConversionError(f"{path}: checkpoint changed during publication")
    if (
        _regular_file_identity(source_index) != source_index_identity
        or _sha256_regular_file(source_index) != SOURCE_INDEX_SHA256
    ):
        raise ConversionError(
            f"{source_index}: authenticated source index changed during publication"
        )
    for root, expected in zip(roots, root_identities, strict=True):
        if _directory_identity(root) != expected:
            raise ConversionError(f"{root}: rank root changed during publication")

    return {
        "schema": MANIFEST_UPGRADE_SCHEMA,
        "from_schema": LEGACY_SCHEMA,
        "to_schema": SCHEMA,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "tp_world_size": DEFAULT_WORLD_SIZE,
        "contract_digest": contract.contract_digest(),
        "metadata_files": metadata_files,
        "metadata_contract_sha256": metadata_contract_sha256,
        "authenticated_source_index": {
            "path": str(source_index),
            "sha256": SOURCE_INDEX_SHA256,
            "tensor_count": len(expected_weight_map),
            "weight_file_count": len(set(expected_weight_map.values())),
        },
        "ranks": [
            {
                "rank": rank,
                "root": str(root.resolve()),
                "manifest_sha256": _sha256_file(root / "tp_manifest.json"),
                "weight_file_count": len(manifests[rank]["files"]),
                "tensor_count": len(manifests[rank]["tensors"]),
                "rank_data_bytes": manifests[rank]["rank_data_bytes"],
            }
            for rank, root in enumerate(roots)
        ],
        "transaction": {
            "id": transaction_id,
            "path": str(transaction),
            "state": "committed",
            "source_weights_modified": False,
            "metadata_files_modified": False,
            "both_manifests_validated_before_publish": True,
            "per_manifest_atomic_replace": True,
            "durable_recovery": True,
            "recovery_actions": ["rollback", "complete"],
        },
    }


def upgrade_rank_local_manifests(
    *,
    rank_dirs: Sequence[Path],
    source_index: Path,
    transaction_dir: Path,
    transaction_id: str,
) -> dict:
    """Upgrade a complete TP2 v1 pair using authenticated, recoverable inputs."""

    if len(rank_dirs) != DEFAULT_WORLD_SIZE:
        raise ConversionError(
            f"--rank-dir must be supplied exactly {DEFAULT_WORLD_SIZE} times"
        )
    roots = [
        _resolved_real_directory(Path(root), f"rank {rank} root")
        for rank, root in enumerate(rank_dirs)
    ]
    if len(set(roots)) != DEFAULT_WORLD_SIZE:
        raise ConversionError("rank roots must be distinct")
    source_index = Path(source_index)
    if source_index.is_symlink():
        raise ConversionError(
            f"{source_index}: authenticated source index cannot be a symlink"
        )
    try:
        source_index = source_index.resolve(strict=True)
    except OSError as exc:
        raise ConversionError(
            f"{source_index}: authenticated source index is unavailable"
        ) from exc
    _regular_file_identity(source_index)
    transaction_root = _resolved_real_directory(
        Path(transaction_dir), "transaction root"
    )
    if any(
        transaction_root == root or transaction_root.is_relative_to(root)
        for root in roots
    ):
        raise ConversionError("transaction root cannot be inside a rank checkpoint")
    _upgrade_transaction_id(transaction_id)
    with _manifest_upgrade_lock(transaction_root):
        return _upgrade_rank_local_manifests_locked(
            roots=roots,
            source_index=source_index,
            transaction_root=transaction_root,
            transaction_id=transaction_id,
        )


def convert_checkpoint(
    *,
    metadata_dir: Path,
    rank_dirs: Sequence[Path],
    source_dir: Path | None,
    cache_dir: Path,
    max_buffer_bytes: int = 64 << 20,
    keep_source: bool = False,
) -> dict:
    """Convert all indexed source files into rank-local TP checkpoints."""

    metadata_dir = Path(metadata_dir)
    config_path = metadata_dir / "config.json"
    index_path = metadata_dir / "model.safetensors.index.json"
    config, source_index = _validate_pinned_metadata(config_path, index_path)
    world_size = len(rank_dirs)
    if world_size != DEFAULT_WORLD_SIZE:
        raise ConversionError(
            f"this pinned deployment requires TP2, got {world_size} rank roots"
        )
    rank_dirs = [Path(x) for x in rank_dirs]
    if len({x.resolve() for x in rank_dirs}) != world_size:
        raise ConversionError("rank roots must be distinct")
    for root in rank_dirs:
        root.mkdir(parents=True, exist_ok=True)

    contract = KimiK3ShardingContract(config, world_size)
    config_sha = _sha256_file(config_path)
    index_sha = _sha256_file(index_path)
    metadata_files = _copy_metadata_files(metadata_dir, rank_dirs)
    metadata_contract_sha256 = hashlib.sha256(
        json.dumps(
            metadata_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    weight_map: dict[str, str] = source_index["weight_map"]
    keys_by_file: dict[str, set[str]] = defaultdict(set)
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise ConversionError("weight_map keys and values must be strings")
        _safe_repo_relative_name(filename)
        keys_by_file[filename].add(name)

    journal_path = metadata_dir / "tp-conversion-journal.json"
    if journal_path.exists():
        journal = _validate_resume_journal(
            _load_json(journal_path),
            config_sha256=config_sha,
            index_sha256=index_sha,
            world_size=world_size,
            contract=contract,
            keys_by_file=keys_by_file,
        )
    else:
        journal = {
            "schema": SCHEMA,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "mlx_lm_pr": MLX_LM_PR,
            "mlx_lm_commit": MLX_LM_COMMIT,
            "mlx_lm_kimi_k3_sha256": MLX_LM_KIMI_K3_SHA256,
            "config_sha256": config_sha,
            "index_sha256": index_sha,
            "contract_digest": contract.contract_digest(),
            "world_size": world_size,
            "files": {},
            "complete": False,
        }
        _atomic_json(journal_path, journal)

    rank_weight_maps: list[dict[str, str]] = [dict() for _ in rank_dirs]
    rank_tensors: list[dict[str, dict]] = [dict() for _ in rank_dirs]
    rank_files: list[dict[str, dict]] = [dict() for _ in rank_dirs]
    rank_total_data = [0] * world_size

    cache_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(keys_by_file)
    for file_number, filename in enumerate(source_files, 1):
        indexed_keys = keys_by_file[filename]
        local_source = (
            Path(source_dir) / filename
            if source_dir is not None
            else cache_dir / filename
        )

        # If every output already proves its expected headers, source I/O can
        # be skipped only after reading the source header.  A completed journal
        # holds those expected plans, allowing truly source-free resume.
        old_entry = journal["files"].get(filename, {})
        if old_entry.get("committed") is True:
            all_present = True
            for rank, root in enumerate(rank_dirs):
                record = old_entry.get("ranks", {}).get(str(rank))
                if record is None:
                    if old_entry.get("excluded"):
                        continue
                    all_present = False
                    break
                path = root / record["name"]
                expected_plans = _validate_journal_rank_record(
                    record,
                    filename=filename,
                    indexed_keys=indexed_keys,
                    contract=contract,
                    rank=rank,
                    field=f"files[{filename!r}].ranks[{rank!r}]",
                )
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != record["bytes"]
                    or _sha256_file(path) != record["sha256"]
                    or not _existing_output_valid(
                        path,
                        expected_plans,
                        rank,
                        world_size,
                    )
                ):
                    all_present = False
                    break
                rank_files[rank][filename] = record
                for name, tensor in record["tensors"].items():
                    rank_weight_maps[rank][name] = filename
                    rank_tensors[rank][name] = tensor
                    rank_total_data[rank] += int(tensor["bytes"])
            if all_present:
                print(
                    f"[{file_number}/{len(source_files)}] verified {filename}",
                    flush=True,
                )
                continue

        if source_dir is not None:
            if not local_source.is_file():
                raise ConversionError(f"missing source shard {local_source}")
            source_download = {
                "bytes": local_source.stat().st_size,
                "sha256": _sha256_file(local_source),
                "url": None,
                "resumed": False,
            }
        else:
            url = _download_url(SOURCE_REPO, SOURCE_REVISION, filename)
            print(
                f"[{file_number}/{len(source_files)}] download {filename}",
                flush=True,
            )
            source_download = download_resumable(url, local_source)

        with SafeTensorFile(local_source) as safe:
            actual_keys = set(safe.tensors)
            if actual_keys != indexed_keys:
                missing = sorted(indexed_keys - actual_keys)[:5]
                extra = sorted(actual_keys - indexed_keys)[:5]
                raise ConversionError(
                    f"{filename}: index/header mismatch; missing={missing}, "
                    f"extra={extra}"
                )

            plans_by_rank: list[list[TensorPlan]] = [[] for _ in rank_dirs]
            for desc in safe.tensors.values():
                for rank in range(world_size):
                    plan = contract.plan(desc, rank)
                    if plan is not None:
                        plans_by_rank[rank].append(plan)

            entry = {
                "source": source_download,
                "ranks": {},
                "excluded": all(not plans for plans in plans_by_rank),
                "committed": False,
            }
            for rank, (root, plans) in enumerate(
                zip(rank_dirs, plans_by_rank, strict=True)
            ):
                if not plans:
                    continue
                destination = root / filename
                if _existing_output_valid(destination, plans, rank, world_size):
                    output = {
                        "name": filename,
                        "bytes": destination.stat().st_size,
                        "sha256": _sha256_file(destination),
                        "tensor_count": len(plans),
                    }
                else:
                    print(
                        f"[{file_number}/{len(source_files)}] rank {rank} "
                        f"write {filename}",
                        flush=True,
                    )
                    output = write_rank_shard(
                        safe,
                        destination,
                        plans,
                        rank=rank,
                        world_size=world_size,
                        max_buffer_bytes=max_buffer_bytes,
                    )
                output["tensors"] = {
                    p.source.name: _tensor_manifest(p, filename) for p in plans
                }
                entry["ranks"][str(rank)] = output

            # Source is retained if any output write raises.  Reaching this
            # point proves all rank outputs were committed.
            entry["committed"] = True
            journal["files"][filename] = entry
            _atomic_json(journal_path, journal)

        if source_dir is None and not keep_source:
            local_source.unlink()
            _fsync_dir(local_source.parent)

        for rank in range(world_size):
            record = entry["ranks"].get(str(rank))
            if record is None:
                continue
            rank_files[rank][filename] = record
            for name, tensor in record["tensors"].items():
                rank_weight_maps[rank][name] = filename
                rank_tensors[rank][name] = tensor
                rank_total_data[rank] += int(tensor["bytes"])

    for rank, root in enumerate(rank_dirs):
        weight_map_sorted = dict(sorted(rank_weight_maps[rank].items()))
        index_payload = {
            "metadata": {
                "total_size": rank_total_data[rank],
                "source_total_parameters": source_index["metadata"]["total_parameters"],
                "tp_rank": rank,
                "tp_world_size": world_size,
                "source_revision": SOURCE_REVISION,
            },
            "weight_map": weight_map_sorted,
        }
        _atomic_json(root / "model.safetensors.index.json", index_payload)
        manifest = {
            "schema": SCHEMA,
            "complete": True,
            "source": {
                "repo": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "config_sha256": config_sha,
                "index_sha256": index_sha,
                "total_size": source_index["metadata"]["total_size"],
                "total_parameters": source_index["metadata"]["total_parameters"],
            },
            "runtime": {
                "mlx_lm_pr": MLX_LM_PR,
                "mlx_lm_commit": MLX_LM_COMMIT,
                "mlx_lm_kimi_k3_sha256": MLX_LM_KIMI_K3_SHA256,
            },
            "tp": {
                "rank": rank,
                "world_size": world_size,
                "contract": CONTRACT_VERSION,
                "contract_digest": contract.contract_digest(),
            },
            "rank_data_bytes": rank_total_data[rank],
            "metadata_files": metadata_files,
            "metadata_contract_sha256": metadata_contract_sha256,
            "files": dict(sorted(rank_files[rank].items())),
            "tensors": dict(sorted(rank_tensors[rank].items())),
        }
        _atomic_json(root / "tp_manifest.json", manifest)

    journal["complete"] = True
    journal["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(journal_path, journal)
    return journal


def audit_contract(config_path: Path, index_path: Path, world_size: int = 2) -> dict:
    """Audit every indexed key that can be classified without weight headers."""

    config, index = _validate_pinned_metadata(config_path, index_path)
    contract = KimiK3ShardingContract(config, world_size)
    counts: dict[str, int] = defaultdict(int)
    reason_counts: dict[str, int] = defaultdict(int)
    excluded_files: set[str] = set()
    for name, filename in index["weight_map"].items():
        # ndim is only needed to resolve dynamic axes, not to classify.
        rule = contract.classify(name, 3)
        counts[rule.kind] += 1
        reason_counts[rule.reason] += 1
        if rule.kind == "excluded":
            excluded_files.add(filename)
    return {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "tensor_count": len(index["weight_map"]),
        "source_file_count": len(set(index["weight_map"].values())),
        "counts_by_rule": dict(sorted(counts.items())),
        "counts_by_reason": dict(sorted(reason_counts.items())),
        "files_containing_excluded_tensors": len(excluded_files),
        "contract_digest": contract.contract_digest(),
    }


def audit_shard(metadata_dir: Path, shard_path: Path) -> dict:
    """Validate and plan one real source shard without reading tensor payloads.

    The pinned config and index are authenticated before the safetensors header
    is opened. The command does not create files, hash the multi-gigabyte
    payload, or call ``SafeTensorFile.array``; it validates only the header,
    physical file bounds, exact index membership, and both TP2 plans.
    """

    metadata_dir = Path(metadata_dir)
    shard_path = Path(shard_path)
    config_path = metadata_dir / "config.json"
    index_path = metadata_dir / "model.safetensors.index.json"
    config, source_index = _validate_pinned_metadata(config_path, index_path)

    source_name = _safe_repo_relative_name(shard_path.name)
    if shard_path.suffix != ".safetensors":
        raise ConversionError(f"{shard_path}: expected a .safetensors source shard")

    weight_map = source_index["weight_map"]
    indexed_keys: set[str] = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(filename, str):
            raise ConversionError("weight_map keys and values must be strings")
        _safe_repo_relative_name(filename)
        if filename == source_name:
            indexed_keys.add(tensor_name)
    if not indexed_keys:
        raise ConversionError(
            f"{source_name}: filename is not present in the pinned weight index"
        )

    contract = KimiK3ShardingContract(config, DEFAULT_WORLD_SIZE)
    ranks: list[dict] = []
    excluded: list[dict] = []
    source_bytes = shard_path.stat().st_size
    with SafeTensorFile(shard_path) as source:
        actual_keys = set(source.tensors)
        if actual_keys != indexed_keys:
            missing = sorted(indexed_keys - actual_keys)
            extra = sorted(actual_keys - indexed_keys)
            raise ConversionError(
                f"{source_name}: index/header mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        source_payload_bytes = sum(desc.nbytes for desc in source.tensors.values())
        physical_payload_bytes = source_bytes - source.data_offset
        if source_payload_bytes != physical_payload_bytes:
            raise ConversionError(
                f"{source_name}: tensor descriptors cover {source_payload_bytes} "
                f"bytes, physical payload has {physical_payload_bytes}"
            )

        plans_by_rank: list[list[TensorPlan]] = [[] for _ in range(DEFAULT_WORLD_SIZE)]
        for desc in source.tensors.values():
            rule = contract.classify(desc.name, len(desc.shape))
            if rule.kind == "excluded":
                excluded.append(
                    {
                        "name": desc.name,
                        "dtype": desc.dtype,
                        "source_shape": list(desc.shape),
                        "source_bytes": desc.nbytes,
                        "rule": rule.canonical(),
                    }
                )
                continue
            for rank in range(DEFAULT_WORLD_SIZE):
                plan = contract.plan(desc, rank)
                if plan is None:
                    raise ConversionError(
                        f"{desc.name}: non-excluded tensor produced no TP{DEFAULT_WORLD_SIZE} "
                        f"plan for rank {rank}"
                    )
                plans_by_rank[rank].append(plan)

        for rank, plans in enumerate(plans_by_rank):
            output_metadata = {
                "format": "mlx",
                "schema": SCHEMA,
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "source_file": source_name,
                "tp_rank": str(rank),
                "tp_world_size": str(DEFAULT_WORLD_SIZE),
                "sharding_contract": CONTRACT_VERSION,
            }
            header_raw, _ = _build_header(plans, output_metadata)
            data_bytes = sum(plan.output_nbytes for plan in plans)
            ranks.append(
                {
                    "rank": rank,
                    "tensor_count": len(plans),
                    "data_bytes": data_bytes,
                    "predicted_file_bytes": 8 + len(header_raw) + data_bytes,
                    "tensors": [
                        {
                            "name": plan.source.name,
                            "dtype": plan.source.dtype,
                            "source_shape": list(plan.source.shape),
                            "rank_shape": list(plan.output_shape),
                            "source_bytes": plan.source.nbytes,
                            "rank_bytes": plan.output_nbytes,
                            "rule": plan.rule.canonical(),
                            "resolved_axis": plan.axis,
                            "intervals": [
                                list(interval) for interval in plan.intervals
                            ],
                        }
                        for plan in plans
                    ],
                }
            )

        source_tensor_count = len(source.tensors)
        source_header_bytes = source.data_offset

    return {
        "schema": SHARD_AUDIT_SCHEMA,
        "read_only": True,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "config_sha256": _sha256_file(config_path),
        "index_sha256": _sha256_file(index_path),
        "contract": CONTRACT_VERSION,
        "contract_digest": contract.contract_digest(),
        "tp_world_size": DEFAULT_WORLD_SIZE,
        "shard": {
            "name": source_name,
            "path": str(shard_path.resolve()),
            "file_bytes": source_bytes,
            "header_bytes": source_header_bytes,
            "payload_bytes": source_payload_bytes,
            "tensor_count": source_tensor_count,
        },
        "excluded_tensor_count": len(excluded),
        "excluded_tensors": excluded,
        "ranks": ranks,
    }


def _validated_rank_output(
    path: Path,
    source_name: str,
    plans: Sequence[TensorPlan],
    record: Mapping,
    rank: int,
) -> str:
    """Re-open a derived rank shard and return its verified SHA-256."""

    expected_size = int(record["bytes"])
    if path.stat().st_size != expected_size:
        raise ConversionError(
            f"{path}: output size {path.stat().st_size}, expected {expected_size}"
        )
    with SafeTensorFile(path) as output:
        expected_metadata = {
            "format": "mlx",
            "schema": SCHEMA,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_file": source_name,
            "tp_rank": str(rank),
            "tp_world_size": str(DEFAULT_WORLD_SIZE),
            "sharding_contract": CONTRACT_VERSION,
        }
        if output.metadata != expected_metadata:
            raise ConversionError(
                f"{path}: output metadata does not match the conversion contract"
            )
        expected_headers = {
            plan.source.name: (plan.source.dtype, plan.output_shape) for plan in plans
        }
        actual_headers = {
            name: (desc.dtype, desc.shape) for name, desc in output.tensors.items()
        }
        if actual_headers != expected_headers:
            raise ConversionError(f"{path}: output header does not match TP2 plan")

    checksum = _sha256_file(path)
    if checksum != record["sha256"]:
        raise ConversionError(
            f"{path}: post-write checksum {checksum} does not match {record['sha256']}"
        )
    return checksum


def convert_shard(
    *,
    metadata_dir: Path,
    shard_path: Path,
    rank_dirs: Sequence[Path],
    max_buffer_bytes: int = 64 << 20,
    transaction_id: str = "manual",
) -> dict:
    """Convert exactly one pinned source shard into a validated TP2 pair.

    Both rank outputs are written to private staged names and fully validated
    before final names are published. Existing final outputs are never
    overwritten. The source is opened read-only and is never renamed, unlinked,
    or otherwise modified.
    """

    if len(rank_dirs) != DEFAULT_WORLD_SIZE:
        raise ConversionError(
            f"--rank-dir must be supplied exactly {DEFAULT_WORLD_SIZE} times"
        )
    if max_buffer_bytes < 1:
        raise ConversionError("max_buffer_bytes must be positive")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", transaction_id) is None:
        raise ConversionError(
            "transaction_id must be 1-128 allowlisted filename characters"
        )

    metadata_dir = Path(metadata_dir)
    shard_path = Path(shard_path)
    rank_dirs = [Path(path) for path in rank_dirs]
    resolved_rank_dirs = [path.resolve() for path in rank_dirs]
    if len(set(resolved_rank_dirs)) != DEFAULT_WORLD_SIZE:
        raise ConversionError("rank output directories must be distinct")

    # This authenticates the metadata and exact index/header key set and proves
    # both rank plans before any output directory or temporary file is created.
    audit = audit_shard(metadata_dir, shard_path)
    source_name = audit["shard"]["name"]
    if any(rank["tensor_count"] == 0 for rank in audit["ranks"]):
        raise ConversionError(
            f"{source_name}: source has no text tensors to convert for TP2"
        )
    final_paths = [directory / source_name for directory in rank_dirs]
    source_resolved = shard_path.resolve()
    for final_path in final_paths:
        if os.path.lexists(final_path):
            raise ConversionError(f"refusing to overwrite existing output {final_path}")
        if final_path.resolve() == source_resolved:
            raise ConversionError(f"output path aliases source shard {shard_path}")

    config, source_index = _validate_pinned_metadata(
        metadata_dir / "config.json",
        metadata_dir / "model.safetensors.index.json",
    )
    indexed_keys = {
        name
        for name, filename in source_index["weight_map"].items()
        if filename == source_name
    }
    contract = KimiK3ShardingContract(config, DEFAULT_WORLD_SIZE)

    source_stat_before = shard_path.stat()
    source_identity_before = (
        source_stat_before.st_dev,
        source_stat_before.st_ino,
        source_stat_before.st_mode,
        source_stat_before.st_size,
        source_stat_before.st_mtime_ns,
        source_stat_before.st_ctime_ns,
    )
    source_checksum = _sha256_file(shard_path)

    stage_paths: list[Path] = []
    published_paths: list[Path] = []
    records: list[dict] = []
    checksums: list[str] = []
    plans_by_rank: list[list[TensorPlan]] = [[] for _ in range(DEFAULT_WORLD_SIZE)]
    try:
        write_partial_paths: list[Path] = []
        for rank, directory in enumerate(rank_dirs):
            directory.mkdir(parents=True, exist_ok=True)
            stage_path = directory / (
                f".{source_name}.rank{rank}.{transaction_id}.staged"
            )
            write_partial = directory / (
                f".{source_name}.rank{rank}.{transaction_id}.write.partial"
            )
            for path in (stage_path, write_partial):
                if os.path.lexists(path):
                    raise ConversionError(
                        f"{path}: deterministic transaction path already exists"
                    )
            stage_paths.append(stage_path)
            write_partial_paths.append(write_partial)

        with SafeTensorFile(shard_path) as source:
            if set(source.tensors) != indexed_keys:
                raise ConversionError(
                    f"{source_name}: index/header changed after read-only audit"
                )
            for desc in source.tensors.values():
                for rank in range(DEFAULT_WORLD_SIZE):
                    plan = contract.plan(desc, rank)
                    if plan is not None:
                        plans_by_rank[rank].append(plan)

            for rank in range(DEFAULT_WORLD_SIZE):
                record = write_rank_shard(
                    source,
                    stage_paths[rank],
                    plans_by_rank[rank],
                    rank=rank,
                    world_size=DEFAULT_WORLD_SIZE,
                    max_buffer_bytes=max_buffer_bytes,
                    temporary_path=write_partial_paths[rank],
                )
                records.append(record)
                checksums.append(
                    _validated_rank_output(
                        stage_paths[rank],
                        source_name,
                        plans_by_rank[rank],
                        record,
                        rank,
                    )
                )

        source_stat_after = shard_path.stat()
        source_identity_after = (
            source_stat_after.st_dev,
            source_stat_after.st_ino,
            source_stat_after.st_mode,
            source_stat_after.st_size,
            source_stat_after.st_mtime_ns,
            source_stat_after.st_ctime_ns,
        )
        if source_identity_after != source_identity_before:
            raise ConversionError(f"{shard_path}: source changed during conversion")

        # Hard-link publication is atomic and no-clobber for each destination.
        # Staging and final names share a directory, hence a filesystem.
        for stage_path, final_path in zip(
            stage_paths,
            final_paths,
            strict=True,
        ):
            os.link(stage_path, final_path, follow_symlinks=False)
            published_paths.append(final_path)
        for directory in set(path.parent for path in final_paths):
            _fsync_dir(directory)
        for stage_path in stage_paths:
            stage_path.unlink()
        for directory in set(path.parent for path in stage_paths):
            _fsync_dir(directory)

        return {
            "schema": SHARD_CONVERSION_SCHEMA,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source": {
                "name": source_name,
                "path": str(source_resolved),
                "bytes": source_stat_before.st_size,
                "sha256": source_checksum,
                "modified": False,
            },
            "tp_world_size": DEFAULT_WORLD_SIZE,
            "max_buffer_bytes": max_buffer_bytes,
            "transaction_id": transaction_id,
            "outputs": [
                {
                    "rank": rank,
                    "path": str(final_path.resolve()),
                    "bytes": int(records[rank]["bytes"]),
                    "data_bytes": sum(
                        plan.output_nbytes for plan in plans_by_rank[rank]
                    ),
                    "tensor_count": len(plans_by_rank[rank]),
                    "sha256": checksums[rank],
                }
                for rank, final_path in enumerate(final_paths)
            ],
            "transaction": {
                "existing_outputs_overwritten": False,
                "both_outputs_validated_before_publish": True,
                "source_deleted": False,
            },
        }
    except BaseException:
        for final_path in reversed(published_paths):
            with contextlib.suppress(FileNotFoundError):
                final_path.unlink()
        raise
    finally:
        for stage_path in stage_paths:
            with contextlib.suppress(FileNotFoundError):
                stage_path.unlink()
        for partial_path in locals().get("write_partial_paths", []):
            with contextlib.suppress(FileNotFoundError):
                partial_path.unlink()


def audit_remote_contract() -> dict:
    """Fetch only the two small pinned metadata files and audit the key map."""

    def fetch(name: str, expected_sha256: str) -> dict:
        request = urllib.request.Request(
            _download_url(SOURCE_REPO, SOURCE_REVISION, name),
            headers={"User-Agent": "k3-rank-local-tp/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        actual_sha256 = _sha256_bytes(raw)
        if actual_sha256 != expected_sha256:
            raise ConversionError(
                f"remote {name}: expected SHA-256 {expected_sha256}, "
                f"got {actual_sha256}"
            )
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ConversionError(f"remote {name}: expected JSON object")
        return value

    config, index = _validate_pinned_metadata_objects(
        fetch("config.json", SOURCE_CONFIG_SHA256),
        fetch("model.safetensors.index.json", SOURCE_INDEX_SHA256),
        f"{SOURCE_REPO}@{SOURCE_REVISION}",
    )
    contract = KimiK3ShardingContract(config, DEFAULT_WORLD_SIZE)
    counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    files_by_kind: dict[str, set[str]] = defaultdict(set)
    for name, filename in index["weight_map"].items():
        rule = contract.classify(name, 3)
        counts[rule.kind] += 1
        reasons[rule.reason] += 1
        files_by_kind[rule.kind].add(filename)
    return {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "tensor_count": len(index["weight_map"]),
        "source_file_count": len(set(index["weight_map"].values())),
        "counts_by_rule": dict(sorted(counts.items())),
        "counts_by_reason": dict(sorted(reasons.items())),
        "file_counts_by_rule": {
            key: len(value) for key, value in sorted(files_by_kind.items())
        },
        "contract_digest": contract.contract_digest(),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="audit pinned index key classification")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--index", type=Path, required=True)
    audit_one = sub.add_parser(
        "audit-shard",
        help="read-only TP2 plan for one indexed real safetensors shard",
    )
    audit_one.add_argument("--metadata-dir", type=Path, required=True)
    audit_one.add_argument("--shard", type=Path, required=True)
    convert_one = sub.add_parser(
        "convert-shard",
        help="convert one pinned source shard into a validated TP2 output pair",
    )
    convert_one.add_argument("--metadata-dir", type=Path, required=True)
    convert_one.add_argument("--shard", type=Path, required=True)
    convert_one.add_argument(
        "--rank-dir",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly twice, rank 0 then rank 1",
    )
    convert_one.add_argument("--max-buffer-mib", type=int, default=64)
    convert_one.add_argument(
        "--transaction-id",
        default="manual",
        help="deterministic allowlisted owner for staged/write-temporary names",
    )
    sub.add_parser(
        "audit-remote",
        help="fetch only pinned config/index metadata and audit classification",
    )

    convert = sub.add_parser("convert", help="stream-convert the pinned checkpoint")
    convert.add_argument("--metadata-dir", type=Path, required=True)
    convert.add_argument(
        "--rank-dir",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly twice, rank 0 then rank 1",
    )
    convert.add_argument(
        "--source-dir",
        type=Path,
        help="local source shards; omit to download each pinned shard once",
    )
    convert.add_argument("--cache-dir", type=Path, required=True)
    convert.add_argument("--max-buffer-mib", type=int, default=64)
    convert.add_argument("--keep-source", action="store_true")
    upgrade = sub.add_parser(
        "upgrade-manifests",
        help="source-free, fail-closed upgrade of a complete TP2 v1 rank pair",
    )
    upgrade.add_argument(
        "--rank-dir",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly twice, rank 0 then rank 1",
    )
    upgrade.add_argument(
        "--source-index",
        type=Path,
        required=True,
        help="immutable raw source model.safetensors.index.json",
    )
    upgrade.add_argument(
        "--transaction-dir",
        type=Path,
        required=True,
        help="existing local directory for durable recovery records",
    )
    upgrade.add_argument(
        "--transaction-id",
        required=True,
        help="unique deterministic identifier for this publication attempt",
    )
    recover = sub.add_parser(
        "recover-manifests",
        help="recover an interrupted manifest upgrade without clobbering unknown state",
    )
    recover.add_argument("--transaction", type=Path, required=True)
    recover.add_argument(
        "--action",
        choices=("rollback", "complete"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "audit":
            report = audit_contract(args.config, args.index)
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "audit-shard":
            report = audit_shard(args.metadata_dir, args.shard)
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "convert-shard":
            if len(args.rank_dir) != DEFAULT_WORLD_SIZE:
                raise ConversionError(
                    f"--rank-dir must be supplied exactly {DEFAULT_WORLD_SIZE} times"
                )
            if args.max_buffer_mib < 1:
                raise ConversionError("--max-buffer-mib must be positive")
            report = convert_shard(
                metadata_dir=args.metadata_dir,
                shard_path=args.shard,
                rank_dirs=args.rank_dir,
                max_buffer_bytes=args.max_buffer_mib << 20,
                transaction_id=args.transaction_id,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "audit-remote":
            print(json.dumps(audit_remote_contract(), indent=2, sort_keys=True))
        elif args.command == "convert":
            if len(args.rank_dir) != 2:
                raise ConversionError("--rank-dir must be supplied exactly twice")
            if args.max_buffer_mib < 1:
                raise ConversionError("--max-buffer-mib must be positive")
            convert_checkpoint(
                metadata_dir=args.metadata_dir,
                rank_dirs=args.rank_dir,
                source_dir=args.source_dir,
                cache_dir=args.cache_dir,
                max_buffer_bytes=args.max_buffer_mib << 20,
                keep_source=args.keep_source,
            )
        elif args.command == "upgrade-manifests":
            report = upgrade_rank_local_manifests(
                rank_dirs=args.rank_dir,
                source_index=args.source_index,
                transaction_dir=args.transaction_dir,
                transaction_id=args.transaction_id,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "recover-manifests":
            report = recover_manifest_upgrade(
                transaction=args.transaction,
                action=args.action,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ConversionError, OSError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
