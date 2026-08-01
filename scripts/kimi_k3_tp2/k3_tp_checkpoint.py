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
SOURCE_INDEX_SHA256 = (
    "ac65bcb3cd9e07cab3e7942ff455dde33879a9e02211bae40938e22fc204ae09"
)
SCHEMA = "k3-rank-local-tp/v2"
SHARD_AUDIT_SCHEMA = "k3-rank-local-tp-shard-audit/v1"
SHARD_CONVERSION_SCHEMA = "k3-rank-local-tp-shard-conversion/v1"
CONTRACT_VERSION = "mlx-lm-kimi-k3-shard@7d505c2"
DEFAULT_WORLD_SIZE = 2
TEXT_PREFIX = "language_model."

# Only these non-weight files may cross from an untrusted model snapshot into a
# rank-local checkpoint.  Keep this list explicit: copying an arbitrary sibling
# file can accidentally publish credentials, host configuration, or executable
# hooks that were never reviewed.
ALLOWED_METADATA_FILENAMES = frozenset(
    {
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "README.md",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "configuration_kimi_k3.py",
        "encoding_k3.py",
        "generation_config.json",
        "kimi_k3_processor.py",
        "kimi_k3_vision_processing.py",
        "media_utils.py",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenization_kimi.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)
REQUIRED_METADATA_FILENAMES = frozenset({"config.json"})
LICENSE_FILENAMES = frozenset({"LICENSE", "LICENSE.md", "LICENSE.txt"})


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
        self._fh = self.path.open("rb")
        file_size = self.path.stat().st_size
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
            header = json.loads(header_raw.rstrip(b" ").decode("utf-8"))
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
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in info.items()
                ):
                    raise ConversionError(f"{self.path}: invalid __metadata__")
                self.metadata = dict(info)
                continue
            try:
                dtype = info["dtype"]
                shape = tuple(int(x) for x in info["shape"])
                start, end = (int(x) for x in info["data_offsets"])
            except Exception as exc:
                raise ConversionError(
                    f"{self.path}: invalid descriptor for {name!r}"
                ) from exc
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
                raise ConversionError(
                    f"{self.path}:{name}: overlapping tensor data"
                )
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
    bytes_per_prefix = max(
        1, output_axis * suffix * plan.source.itemsize
    )
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
            expected = {
                p.source.name: (p.source.dtype, p.output_shape) for p in plans
            }
            actual = {
                name: (desc.dtype, desc.shape)
                for name, desc in check.tensors.items()
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
            if suffix.startswith(
                ("self_attn.embed_q.", "self_attn.unembed_out.")
            ):
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
    safe_file = "/".join(
        urllib.parse.quote(x, safe="") for x in filename.split("/")
    )
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
        raise ConversionError(f"{path}: cannot safely open metadata file: {exc}") from exc
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
        if src.name not in ALLOWED_METADATA_FILENAMES:
            continue
        if not src.is_file():
            raise ConversionError(f"{src}: allowlisted metadata must be a regular file")
        sources[src.name] = src

    missing = sorted(REQUIRED_METADATA_FILENAMES - sources.keys())
    if missing:
        raise ConversionError(
            f"{metadata_dir}: missing required metadata: {', '.join(missing)}"
        )
    if not (LICENSE_FILENAMES & sources.keys()):
        raise ConversionError(
            f"{metadata_dir}: missing Kimi K3 license; expected one of "
            f"{', '.join(sorted(LICENSE_FILENAMES))}"
        )

    source_hashes = {
        name: _sha256_regular_file(src) for name, src in sorted(sources.items())
    }
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
            staged.append(
                (_stage_metadata_file(src, dst, expected_sha256), dst)
            )
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
        records[name] = {
            "bytes": sizes.pop(),
            "sha256": expected_sha256,
        }
    return records


def _existing_output_valid(
    path: Path,
    expected: Sequence[TensorPlan],
    rank: int,
    world_size: int,
) -> bool:
    if not path.exists():
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
            actual_headers = {
                k: (v.dtype, v.shape) for k, v in safe.tensors.items()
            }
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
        journal = _load_json(journal_path)
        if journal.get("source_revision") != SOURCE_REVISION:
            raise ConversionError("existing journal is for a different revision")
        if journal.get("config_sha256") != config_sha:
            raise ConversionError("config changed since conversion began")
        if journal.get("index_sha256") != index_sha:
            raise ConversionError("weight index changed since conversion began")
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
                if (
                    not path.exists()
                    or path.stat().st_size != record["bytes"]
                    or _sha256_file(path) != record["sha256"]
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
                "source_total_parameters": source_index["metadata"][
                    "total_parameters"
                ],
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

        plans_by_rank: list[list[TensorPlan]] = [
            [] for _ in range(DEFAULT_WORLD_SIZE)
        ]
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
                            "intervals": [list(interval) for interval in plan.intervals],
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
            f"{path}: post-write checksum {checksum} does not match "
            f"{record['sha256']}"
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
    plans_by_rank: list[list[TensorPlan]] = [
        [] for _ in range(DEFAULT_WORLD_SIZE)
    ]
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
        return 0
    except (ConversionError, OSError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
