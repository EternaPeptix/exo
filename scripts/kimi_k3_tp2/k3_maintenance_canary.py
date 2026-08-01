#!/usr/bin/env python3
"""Fail-closed maintenance planner for the two-rank Kimi K3 canaries.

The default operation performs read-only preflight probes and emits a stable
JSON plan.  A mutating action requires ``--apply``, an action-specific
confirmation string, and the exact digest of the inspected plan.  Commands are
always executed as argv arrays; this program never invokes a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

SCHEMA: Final = "k3-maintenance-canary/v1"
INVENTORY_SCHEMA: Final = "k3-maintenance-canary-inventory/v1"
FACTS_SCHEMA: Final = "k3-maintenance-canary-facts/v1"

# Candidate sources plus distinct rollback/baseline evidence.
EXO_DSPARK_CONTRACT_COMMIT: Final = "6eb59a4770d7a2efaa632c0b3f3df26fde545361"
MLX_LM_DSPARK_COMMIT: Final = "bf378e33831e745715a88418a44ce20ab1075b9b"
MLX_DSPARK_COMMIT: Final = "2cfb83040011c273377a25df8ed16def80c6646c"
MLX_ACCEPTED_COMMIT: Final = "57b87fe47cfce34d6dc59d0e274d8ee36bfb9308"
MLX_FACTORIZED_COMMIT: Final = "152f01807c8327ac154b8ed56dd9279a6f9506e6"
MLX_LM_FACTORIZED_WIRE_COMMIT: Final = "53dbe04a0499ffb3e98ede90ff5a82f118f77f04"
MLX_LM_KIMI_K3_SHA256: Final = (
    "3e283240117d298d95e33f7238cb49abc5606aafdd26f70062e841518059088b"
)
MLX_LM_KIMI_K3_DSPARK_SHA256: Final = (
    "5ba010755e703f39b86aed1ad999576a18f2f93c041b502bbe3f57b197af2f01"
)
MLX_LM_GATED_DELTA_SHA256: Final = (
    "44aef2791ed0cd5cfb84e31ef00cb4df3d40ae184e6c6852b7f0dba7406d2f78"
)
MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256: Final = (
    "d51bf88fa603846f4ac9876faf724d273a99ca8b6643715f68df5993eb352d68"
)
MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256: Final = (
    "0aa226e32b992e5bb18a14b4a3ead225b6a6f531431249e1ae6b5bbb35ca29d1"
)
MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256: Final = (
    "2b9841394f8334e02044f2e6b418f0bc7ff41cc719a877464a311c5e8c899ee9"
)
MLX_LM_KIMI_K3_PACKED_MOE_FRONT_SHA256: Final = (
    "82076bf9c0098f2fc72a0434a5482e72a6e022fc982574f05867dcce32645435"
)

DSPARK_REVISION: Final = "eb03982e58d4fb79bcfc099e902158f562e2e27b"
DSPARK_CONFIG_SHA256: Final = (
    "6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f"
)
DSPARK_MODEL_SHA256: Final = (
    "29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495"
)
DSPARK_MODEL_BYTES: Final = 4_498_585_858
DSPARK_PATH: Final = (
    "/Users/jeweled/.exo/models/"
    "RadixArk--Kimi-K3-DSpark-eb03982e58d4fb79bcfc099e902158f562e2e27b"
)
ACCEPTED_COMPLETION_SHA256: Final = (
    "c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71"
)

EXPECTED_TRANSPORT_MATRIX: Final = [
    [None, ["rdma_en5"]],
    [["rdma_en5"], None],
]
MIN_RAM_BYTES: Final = 500 * 1024**3
MIN_DISK_FREE_BYTES: Final = 16 * 1024**3

ACTIONS: Final[tuple[str, ...]] = (
    "stop",
    "start-baseline",
    "start-dspark-width3",
    "start-dspark-width8",
    "rollback",
    "start-factorized-off",
    "start-factorized-on",
)

ACCEPTED_FLAGS: Final[dict[str, str]] = {
    "EXO_NO_BATCH": "1",
    "EXO_MLX_JACCL_FORCE_MESH": "1",
    "EXO_MLX_MAX_ATTENTION_CELLS_PER_CHUNK": "268435456",
    "EXO_MLX_K3_VOCAB_PARALLEL_HEAD": "1",
    "EXO_MLX_K3_VOCAB_PARALLEL_GREEDY": "1",
    "EXO_MLX_K3_REQUANT_ROUTED_LATENT_MXFP4": "0",
    "EXO_MLX_K3_REQUANT_ATTENTION_QKVG_MXFP4": "0",
    "MLX_METAL_FAST_SYNCH": "1",
    "MLX_METAL_K3_AFFINE8_ROWPAIR": "0",
    "MLX_METAL_K3_PACKED_FRONT_ROWPAIR": "0",
    "MLX_LM_KIMI_K3_FUSED_EXPERTS": "1",
    "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH2": "0",
    "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE": "1",
    "MLX_LM_KIMI_K3_FUSED_ROUTER": "1",
    "MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS": "1",
    "MLX_LM_KIMI_K3_PACKED_KDA_SKINNY": "1",
    "MLX_LM_KIMI_K3_PACKED_KDA_WIDE": "1",
    "MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD": "1",
    "MLX_LM_KIMI_K3_FUSED_POST_KDA_RMS_SIGMOID_GATE": "0",
    "MLX_LM_KIMI_K3_COMPILED_DECODE": "0",
    "MLX_LM_KIMI_K3_PACKED_MOE_FRONT": "0",
    "MLX_LM_KIMI_K3_PACKED_MOE_FRONT_WIDTH8": "0",
    "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT": "1",
    "MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL": "1",
    "MLX_LM_EXPERIMENTAL_KDA_ROW_DECODE": "0",
    "MLX_LM_KIMI_K3_DSPARK_SEGMENTED_SDPA": "0",
    "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES": "laguna8",
    "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE": "hidden",
}


class CanaryError(RuntimeError):
    """A canary inventory, preflight, or apply contract failed."""


@dataclass(frozen=True)
class Node:
    name: str
    rank: int
    ssh: tuple[str, ...]
    paths: Mapping[str, str]
    rank_manifest_sha256: str


@dataclass(frozen=True)
class Inventory:
    path: Path
    nodes: tuple[Node, Node]
    actions: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ActionStep:
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    required_file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class Arguments:
    inventory: Path
    facts_dir: Path | None
    facts_output_dir: Path | None
    output: Path | None
    action: str
    apply: bool
    maintenance_confirm: str | None
    plan_digest: str | None


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CanaryError(f"{name} must be a JSON object")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise CanaryError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], raw)


def _json_loads(text: str) -> object:
    return cast(object, json.loads(text))


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CanaryError(f"{name} must be a non-empty string")
    return value


def load_inventory(path: Path) -> Inventory:
    raw = _object(_json_loads(path.read_text(encoding="utf-8")), "inventory")
    if raw.get("schema") != INVENTORY_SCHEMA:
        raise CanaryError(f"inventory schema must be {INVENTORY_SCHEMA}")
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list):
        raise CanaryError("inventory must contain exactly two nodes")
    node_values = cast(list[object], raw_nodes)
    if len(node_values) != 2:
        raise CanaryError("inventory must contain exactly two nodes")
    nodes: list[Node] = []
    for index, value in enumerate(node_values):
        item = _object(value, f"nodes[{index}]")
        rank = item.get("rank")
        if type(rank) is not int or rank not in (0, 1):
            raise CanaryError("node ranks must be exactly 0 and 1")
        ssh_raw = item.get("ssh")
        if not isinstance(ssh_raw, list) or not ssh_raw:
            raise CanaryError(f"nodes[{index}].ssh must be a non-empty argv list")
        ssh_values = cast(list[object], ssh_raw)
        if not all(isinstance(part, str) and part for part in ssh_values):
            raise CanaryError(f"nodes[{index}].ssh must be a non-empty argv list")
        raw_paths = _object(item.get("paths"), f"nodes[{index}].paths")
        paths = {
            name: _string(raw_paths.get(name), f"nodes[{index}].paths.{name}")
            for name in (
                "exo_root",
                "mlx_lm_root",
                "mlx_root",
                "factorized_mlx_root",
                "rank_checkpoint",
                "dspark_checkpoint",
                "transport_contract",
                "artifact_root",
            )
        }
        if paths["dspark_checkpoint"] != DSPARK_PATH:
            raise CanaryError(f"DSpark path must be pinned to {DSPARK_PATH}")
        nodes.append(
            Node(
                name=_string(item.get("name"), f"nodes[{index}].name"),
                rank=rank,
                ssh=tuple(cast(list[str], ssh_values)),
                paths=paths,
                rank_manifest_sha256=_string(
                    item.get("rank_manifest_sha256"),
                    f"nodes[{index}].rank_manifest_sha256",
                ),
            )
        )
    nodes.sort(key=lambda node: node.rank)
    if [node.rank for node in nodes] != [0, 1]:
        raise CanaryError("node ranks must be unique")
    raw_actions = _object(raw.get("actions"), "actions")
    expected_actions: set[str] = set(ACTIONS)
    if set(raw_actions) != expected_actions:
        missing = sorted(expected_actions - set(raw_actions))
        unknown = sorted(set(raw_actions) - expected_actions)
        raise CanaryError(
            f"inventory actions must exactly match the contract; "
            f"missing={missing}, unknown={unknown}"
        )
    actions: dict[str, Mapping[str, object]] = {}
    for action in ACTIONS:
        raw_action = _object(raw_actions[action], f"actions.{action}")
        actions[action] = raw_action
        enabled = raw_action.get("enabled")
        if type(enabled) is not bool:
            raise CanaryError(f"actions.{action}.enabled must be a boolean")
        _string(raw_action.get("reason"), f"actions.{action}.reason")
        raw_steps = raw_action.get("steps")
        if not isinstance(raw_steps, list) or (enabled and not raw_steps):
            raise CanaryError(
                f"actions.{action}.steps must be a list and cannot be empty "
                "when enabled"
            )
        step_values = cast(list[object], raw_steps)
        for index, step_value in enumerate(step_values):
            _parse_action_step(step_value, f"actions.{action}.steps[{index}]")
    return Inventory(path=path, nodes=(nodes[0], nodes[1]), actions=actions)


def _parse_action_step(value: object, name: str) -> ActionStep:
    step = _object(value, name)
    raw_argv = step.get("argv")
    if not isinstance(raw_argv, list) or not raw_argv:
        raise CanaryError(f"{name}.argv must be a non-empty argv list")
    argv_values = cast(list[object], raw_argv)
    if not all(isinstance(part, str) and part for part in argv_values):
        raise CanaryError(f"{name}.argv must be a non-empty argv list")
    environment_object = _object(step.get("environment", {}), f"{name}.environment")
    environment = {
        key: _string(item, f"{name}.environment.{key}")
        for key, item in environment_object.items()
    }
    required_object = _object(
        step.get("required_file_sha256", {}),
        f"{name}.required_file_sha256",
    )
    required_file_sha256: dict[str, str] = {}
    for raw_path, raw_digest in required_object.items():
        path = Path(raw_path)
        digest = _string(raw_digest, f"{name}.required_file_sha256.{raw_path}")
        if not path.is_absolute():
            raise CanaryError(f"{name} required file paths must be absolute")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise CanaryError(f"{name} required file hashes must be SHA-256 hex")
        required_file_sha256[raw_path] = digest
    return ActionStep(
        tuple(cast(list[str], argv_values)),
        environment,
        required_file_sha256,
    )


_REMOTE_PROBE = r"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

request = json.loads(sys.argv[1])

def run(argv):
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

paths = request["paths"]
git_roots = {
    "exo": paths["exo_root"],
    "mlx_lm": paths["mlx_lm_root"],
    "mlx": paths["mlx_root"],
    "factorized_mlx": paths["factorized_mlx_root"],
}
commits = {name: run(["git", "-C", root, "rev-parse", "HEAD"])["stdout"] for name, root in git_roots.items()}
git_clean = {
    name: run(["git", "-C", root, "status", "--porcelain", "--untracked-files=all"])
    for name, root in git_roots.items()
}
factorized_wire = run([
    "git", "-C", paths["mlx_lm_root"], "merge-base", "--is-ancestor",
    request["mlx_lm_factorized_wire_commit"], "HEAD",
])
required = {
    "exo_dspark_contract": pathlib.Path(paths["exo_root"], "src/exo/worker/engines/mlx/generator/kimi_k3_dspark.py"),
    "exo_generate": pathlib.Path(paths["exo_root"], "src/exo/worker/engines/mlx/generator/generate.py"),
    "mlx_lm_dspark": pathlib.Path(paths["mlx_lm_root"], "mlx_lm/models/kimi_k3_dspark.py"),
    "mlx_lm_kimi_k3": pathlib.Path(paths["mlx_lm_root"], "mlx_lm/models/kimi_k3.py"),
    "mlx_lm_gated_delta": pathlib.Path(paths["mlx_lm_root"], "mlx_lm/models/gated_delta.py"),
    "mlx_lm_kimi_k3_fused_expert": pathlib.Path(paths["mlx_lm_root"], "mlx_lm/models/kimi_k3_fused_expert.py"),
    "mlx_lm_kimi_k3_fused_switch_glu": pathlib.Path(paths["mlx_lm_root"], "mlx_lm/models/kimi_k3_fused_switch_glu.py"),
    "mlx_lm_kimi_k3_fused_down_reduce": pathlib.Path(paths["mlx_lm_root"], "mlx_lm/models/kimi_k3_fused_down_reduce.py"),
    "mlx_lm_kimi_k3_packed_moe_front": pathlib.Path(paths["mlx_lm_root"], "mlx_lm/models/kimi_k3_packed_moe_front.py"),
    "factorized_kernel": pathlib.Path(paths["factorized_mlx_root"], "mlx/backend/metal/kernels/steel/attn/kernels/steel_factorized_attention.h"),
    "rank_manifest": pathlib.Path(paths["rank_checkpoint"], "tp_manifest.json"),
    "transport": pathlib.Path(paths["transport_contract"]),
}
dspark = pathlib.Path(paths["dspark_checkpoint"])
config = dspark / "config.json"
model = dspark / "model.safetensors"
transport = json.loads(required["transport"].read_text()) if required["transport"].is_file() else None
interfaces = {}
for interface in ("en3", "en4", "en5", "en6"):
    result = run(["/sbin/ifconfig", interface])
    interfaces[interface] = {"exists": result["returncode"] == 0, "active": "status: active" in result["stdout"], "raw": result["stdout"]}
thunderbolt = run(["/usr/sbin/system_profiler", "SPThunderboltDataType", "-json", "-detailLevel", "mini"])
global_tb5_80g_present = bool(re.search(r"(?:80\s*Gb/s|80\s*Gbps|80Gbase)", thunderbolt["stdout"], re.IGNORECASE))
ram = run(["/usr/sbin/sysctl", "-n", "hw.memsize"])
disk = shutil.disk_usage(paths["artifact_root"] if os.path.exists(paths["artifact_root"]) else pathlib.Path(paths["artifact_root"]).parent)
generate_text = required["exo_generate"].read_text() if required["exo_generate"].is_file() else ""
facts = {
    "schema": "k3-maintenance-canary-facts/v1",
    "node": request["name"],
    "rank": request["rank"],
    "commits": commits,
    "git_clean": {name: result["returncode"] == 0 and not result["stdout"] for name, result in git_clean.items()},
    "mlx_lm_factorized_wire_is_ancestor": factorized_wire["returncode"] == 0,
    "required_files": {name: path.is_file() for name, path in required.items()},
    "mlx_lm_source_sha256": {
        name: digest(required[name]) if required[name].is_file() else None
        for name in (
            "mlx_lm_kimi_k3",
            "mlx_lm_dspark",
            "mlx_lm_gated_delta",
            "mlx_lm_kimi_k3_fused_expert",
            "mlx_lm_kimi_k3_fused_switch_glu",
            "mlx_lm_kimi_k3_fused_down_reduce",
            "mlx_lm_kimi_k3_packed_moe_front",
        )
    },
    "dspark": {
        "path": str(dspark),
        "config_sha256": digest(config) if config.is_file() else None,
        "model_sha256": digest(model) if model.is_file() else None,
        "model_bytes": model.stat().st_size if model.is_file() else None,
    },
    "rank_manifest_sha256": digest(required["rank_manifest"]) if required["rank_manifest"].is_file() else None,
    "transport": transport,
    "interfaces": interfaces,
    # system_profiler does not authoritatively correlate a Thunderbolt port to
    # BSD interface en5. Keep global TB5 evidence separate and fail DSpark
    # readiness until a correlated/manual-attested fact is available.
    "global_tb5_80g_present": global_tb5_80g_present,
    "en5_tb5_correlated": False,
    "ram_bytes": int(ram["stdout"]) if ram["returncode"] == 0 and ram["stdout"].isdigit() else None,
    "disk_free_bytes": disk.free,
    "adapter_wired": "kimi_k3_dspark" in generate_text,
}
print(json.dumps(facts, sort_keys=True))
""".strip()


Probe = Callable[[Node], Mapping[str, object]]


def ssh_probe(node: Node) -> Mapping[str, object]:
    request = json.dumps(
        {
            "name": node.name,
            "rank": node.rank,
            "paths": node.paths,
            "mlx_lm_factorized_wire_commit": MLX_LM_FACTORIZED_WIRE_COMMIT,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    result = subprocess.run(
        [*node.ssh, "/usr/bin/python3", "-c", _REMOTE_PROBE, request],
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    if result.returncode != 0:
        raise CanaryError(
            f"{node.name} read-only preflight failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return _object(_json_loads(result.stdout), f"{node.name} facts")


def facts_directory_probe(directory: Path) -> Probe:
    def read(node: Node) -> Mapping[str, object]:
        return _object(
            _json_loads((directory / f"rank{node.rank}.json").read_text()),
            f"rank{node.rank} facts",
        )

    return read


def _nested(mapping: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _object(mapping.get(name), name)


def validate_facts(node: Node, facts: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if facts.get("schema") != FACTS_SCHEMA:
        errors.append(f"facts schema must be {FACTS_SCHEMA}")
    if facts.get("rank") != node.rank or facts.get("node") != node.name:
        errors.append("facts node/rank identity mismatch")
    commits = _nested(facts, "commits")
    expected_commits = {
        "exo": EXO_DSPARK_CONTRACT_COMMIT,
        "mlx_lm": MLX_LM_DSPARK_COMMIT,
        "mlx": MLX_DSPARK_COMMIT,
        "factorized_mlx": MLX_FACTORIZED_COMMIT,
    }
    for name, expected in expected_commits.items():
        if commits.get(name) != expected:
            errors.append(f"{name} commit must be {expected}")
    git_clean = _nested(facts, "git_clean")
    for name in expected_commits:
        if git_clean.get(name) is not True:
            errors.append(f"{name} source worktree must be clean")
    if facts.get("mlx_lm_factorized_wire_is_ancestor") is not True:
        errors.append(
            "consolidated MLX-LM head must contain factorized wire commit "
            f"{MLX_LM_FACTORIZED_WIRE_COMMIT}"
        )
    required = _nested(facts, "required_files")
    for name in (
        "exo_dspark_contract",
        "exo_generate",
        "mlx_lm_dspark",
        "mlx_lm_kimi_k3",
        "mlx_lm_gated_delta",
        "mlx_lm_kimi_k3_fused_expert",
        "mlx_lm_kimi_k3_fused_switch_glu",
        "mlx_lm_kimi_k3_fused_down_reduce",
        "mlx_lm_kimi_k3_packed_moe_front",
        "factorized_kernel",
        "rank_manifest",
        "transport",
    ):
        if required.get(name) is not True:
            errors.append(f"required file missing: {name}")
    source_hashes = _nested(facts, "mlx_lm_source_sha256")
    expected_source_hashes = {
        "mlx_lm_kimi_k3": MLX_LM_KIMI_K3_SHA256,
        "mlx_lm_dspark": MLX_LM_KIMI_K3_DSPARK_SHA256,
        "mlx_lm_gated_delta": MLX_LM_GATED_DELTA_SHA256,
        "mlx_lm_kimi_k3_fused_expert": MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256,
        "mlx_lm_kimi_k3_fused_switch_glu": (MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256),
        "mlx_lm_kimi_k3_fused_down_reduce": (MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256),
        "mlx_lm_kimi_k3_packed_moe_front": (MLX_LM_KIMI_K3_PACKED_MOE_FRONT_SHA256),
    }
    for name, expected in expected_source_hashes.items():
        if source_hashes.get(name) != expected:
            errors.append(f"{name} SHA-256 must be {expected}")
    dspark = _nested(facts, "dspark")
    if dspark.get("path") != DSPARK_PATH:
        errors.append("DSpark checkpoint path mismatch")
    if dspark.get("config_sha256") != DSPARK_CONFIG_SHA256:
        errors.append("DSpark config hash mismatch")
    if dspark.get("model_sha256") != DSPARK_MODEL_SHA256:
        errors.append("DSpark model hash mismatch")
    if dspark.get("model_bytes") != DSPARK_MODEL_BYTES:
        errors.append("DSpark model byte size mismatch")
    if facts.get("rank_manifest_sha256") != node.rank_manifest_sha256:
        errors.append("rank-local manifest hash mismatch")
    transport = facts.get("transport")
    if not isinstance(transport, dict) or _object(
        cast(object, transport), "transport"
    ).get("device_matrix") != (EXPECTED_TRANSPORT_MATRIX):
        errors.append("transport must be the one-rail rdma_en5 matrix")
    interfaces = _nested(facts, "interfaces")
    for interface in ("en3", "en4", "en5", "en6"):
        status = _nested(interfaces, interface)
        expected_active = interface == "en5"
        if status.get("exists") is not True:
            errors.append(f"{interface} is missing")
        if status.get("active") is not expected_active:
            errors.append(
                f"{interface} must be {'active' if expected_active else 'inactive'}"
            )
    if facts.get("global_tb5_80g_present") is not True:
        errors.append("system_profiler must report at least one 80 Gb/s TB5 link")
    ram = facts.get("ram_bytes")
    if type(ram) is not int or ram < MIN_RAM_BYTES:
        errors.append(f"physical RAM must be at least {MIN_RAM_BYTES} bytes")
    disk = facts.get("disk_free_bytes")
    if type(disk) is not int or disk < MIN_DISK_FREE_BYTES:
        errors.append(f"artifact disk free must be at least {MIN_DISK_FREE_BYTES}")
    return errors


def _speculation_environment(width: int) -> dict[str, str]:
    return {
        "EXO_MLX_KIMI_K3_DSPARK_SPECULATIVE": "1",
        "EXO_MLX_KIMI_K3_DSPARK_CHECKPOINT": DSPARK_PATH,
        "EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH": str(width),
        "EXO_MLX_KIMI_K3_DSPARK_ROUND_TELEMETRY": "1",
        "MLX_LM_KIMI_K3_DSPARK_PROPOSER": "1",
        "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE": "1",
    }


def _abba_plan() -> list[dict[str, object]]:
    baseline = {
        "mode": "baseline",
        "environment": ACCEPTED_FLAGS,
        "unset": [
            "EXO_MLX_KIMI_K3_DSPARK_SPECULATIVE",
            "EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH",
            "MLX_LM_KIMI_K3_FACTORIZED_SDPA_PREFILL",
        ],
    }
    phases: list[dict[str, object]] = [
        {
            "id": "target-verifier",
            "kind": "exact-target-verifier",
            "widths": [2, 3, 8],
            "gates": {
                "bitwise_logits": True,
                "cache_digest_equal": True,
                "completion_sha256": ACCEPTED_COMPLETION_SHA256,
                "minimum_speedup_by_width": {"2": 1.20, "3": 1.45, "8": 2.00},
            },
        }
    ]
    for label, width, min_acceptance, min_emitted in (
        ("gamma2-width3", 3, 0.70, 2.391),
        ("gamma7-width8", 8, 0.50, 4.50),
    ):
        candidate = {
            "mode": label,
            "environment": {**ACCEPTED_FLAGS, **_speculation_environment(width)},
            "gates": {
                "completion_sha256": ACCEPTED_COMPLETION_SHA256,
                "fallback_rounds": 0,
                "error_rounds": 0,
                "rank_disagreements": 0,
                "minimum_acceptance_rate": min_acceptance,
                "minimum_mean_emitted_per_round": min_emitted,
                "minimum_decode_tps_for_promotion": 17.0,
                "maximum_abba_baseline_drift_fraction": 0.03,
            },
        }
        phases.append(
            {
                "id": f"abba-{label}",
                "kind": "short-prompt-abba",
                "prompt_tokens": 575,
                "decode_tokens": 128,
                "seed": 20260729,
                "sequence": [
                    {"leg": "A1", **baseline},
                    {"leg": "B1", **candidate},
                    {"leg": "B2", **candidate},
                    {"leg": "A2", **baseline},
                ],
            }
        )
    return phases


def _factorized_plan() -> dict[str, object]:
    return {
        "separate_from_dspark": True,
        "mlx_commit": MLX_FACTORIZED_COMMIT,
        "mlx_lm_wire_commit": MLX_LM_FACTORIZED_WIRE_COMMIT,
        "sequence": [
            {"leg": "A1", "factorized_prefill": False},
            {"leg": "B1", "factorized_prefill": True},
            {"leg": "B2", "factorized_prefill": True},
            {"leg": "A2", "factorized_prefill": False},
        ],
        "cases": [
            {"query_tokens": 32, "context_tokens": 512},
            {"query_tokens": 32, "context_tokens": 4096},
            {"query_tokens": 32, "context_tokens": 8192},
        ],
        "gates": {
            "query_tokens_must_exceed": 8,
            "selected_fast_path_calls_when_on_minimum": 1,
            "selected_fast_path_calls_when_off": 0,
            "fallback_calls_when_supported": 0,
            "completion_sha256": ACCEPTED_COMPLETION_SHA256,
            "minimum_prefill_tps_fraction_of_abba_baseline": 0.95,
            "maximum_decode_tps_regression_fraction": 0.02,
            "maximum_peak_memory_increase_bytes": 268_435_456,
            "score_memory_slope_bytes_per_key": 0,
        },
    }


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _action_contract(inventory: Inventory) -> dict[str, object]:
    contract: dict[str, object] = {}
    for action in ACTIONS:
        raw_action = _object(inventory.actions[action], f"actions.{action}")
        raw_steps = cast(list[object], raw_action["steps"])
        steps: list[dict[str, object]] = []
        for index, raw_step in enumerate(raw_steps):
            step = _parse_action_step(raw_step, f"actions.{action}.steps[{index}]")
            steps.append(
                {
                    "argv": list(step.argv),
                    "environment": dict(step.environment),
                    "required_file_sha256": dict(step.required_file_sha256),
                }
            )
        contract[action] = {
            "enabled": raw_action["enabled"],
            "reason": raw_action["reason"],
            "steps": steps,
        }
    return contract


def _action_prerequisite_errors(
    action_contract: Mapping[str, object], action: str
) -> list[str]:
    """Validate only local immutable files needed to execute an action."""

    errors: list[str] = []
    action_value = _nested(action_contract, action)
    if action_value.get("enabled") is not True:
        return ["action is disabled"]
    raw_steps = action_value.get("steps")
    if not isinstance(raw_steps, list):
        return ["action steps are malformed"]
    for index, raw_step in enumerate(cast(list[object], raw_steps)):
        step = _object(raw_step, f"actions.{action}.steps[{index}]")
        required = _nested(step, "required_file_sha256")
        for raw_path, expected in required.items():
            path = Path(raw_path)
            if not path.is_file():
                errors.append(f"required local file is missing: {path}")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(8 << 20):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                errors.append(f"required local file hash mismatch: {path}")
    return errors


def build_plan(
    inventory: Inventory,
    facts_by_rank: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    all_errors: list[str] = []
    adapter_wired = True
    en5_tb5_correlated = True
    for node in inventory.nodes:
        facts = facts_by_rank[node.rank]
        errors = validate_facts(node, facts)
        all_errors.extend(f"{node.name}: {error}" for error in errors)
        adapter_wired = adapter_wired and facts.get("adapter_wired") is True
        en5_tb5_correlated = (
            en5_tb5_correlated and facts.get("en5_tb5_correlated") is True
        )
        reports.append(
            {"node": node.name, "rank": node.rank, "pass": not errors, "errors": errors}
        )
    action_contract = _action_contract(inventory)

    def configured(action: str) -> bool:
        return _nested(action_contract, action).get("enabled") is True

    action_prerequisites = {
        action: _action_prerequisite_errors(action_contract, action)
        for action in ACTIONS
    }

    def locally_ready(action: str) -> bool:
        return configured(action) and not action_prerequisites[action]

    action_readiness = {
        # Stop and the two legacy actions are recovery controls, not candidate
        # promotion. Their immutable local harness prerequisites remain
        # fail-closed, while the harness re-resolves its pinned remote legacy
        # source before mutation. Candidate-only fact failures must not disable
        # the rollback promised by trigger_on_any_gate_failure.
        "stop": locally_ready("stop"),
        "start-baseline": locally_ready("start-baseline"),
        "rollback": locally_ready("rollback"),
        "start-dspark-width3": (
            not all_errors
            and adapter_wired
            and en5_tb5_correlated
            and locally_ready("start-dspark-width3")
        ),
        "start-dspark-width8": (
            not all_errors
            and adapter_wired
            and en5_tb5_correlated
            and locally_ready("start-dspark-width8")
        ),
        "start-factorized-off": (
            not all_errors and locally_ready("start-factorized-off")
        ),
        "start-factorized-on": (
            not all_errors and locally_ready("start-factorized-on")
        ),
    }
    core: dict[str, object] = {
        "schema": SCHEMA,
        "default_mode": "preflight-only",
        "pins": {
            "exo_dspark_contract": EXO_DSPARK_CONTRACT_COMMIT,
            "mlx_lm_dspark": MLX_LM_DSPARK_COMMIT,
            "mlx_dspark": MLX_DSPARK_COMMIT,
            "mlx_lm_kimi_k3_sha256": MLX_LM_KIMI_K3_SHA256,
            "mlx_lm_kimi_k3_dspark_sha256": MLX_LM_KIMI_K3_DSPARK_SHA256,
            "mlx_lm_gated_delta_sha256": MLX_LM_GATED_DELTA_SHA256,
            "mlx_lm_kimi_k3_fused_expert_sha256": (MLX_LM_KIMI_K3_FUSED_EXPERT_SHA256),
            "mlx_lm_kimi_k3_fused_switch_glu_sha256": (
                MLX_LM_KIMI_K3_FUSED_SWITCH_GLU_SHA256
            ),
            "mlx_lm_kimi_k3_fused_down_reduce_sha256": (
                MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE_SHA256
            ),
            "mlx_lm_kimi_k3_packed_moe_front_sha256": (
                MLX_LM_KIMI_K3_PACKED_MOE_FRONT_SHA256
            ),
            "mlx_rollback_only_decode": MLX_ACCEPTED_COMMIT,
            "mlx_factorized_rollback_only": MLX_FACTORIZED_COMMIT,
            "mlx_lm_factorized_wire": MLX_LM_FACTORIZED_WIRE_COMMIT,
            "dspark_revision": DSPARK_REVISION,
            "dspark_path": DSPARK_PATH,
            "dspark_config_sha256": DSPARK_CONFIG_SHA256,
            "dspark_model_sha256": DSPARK_MODEL_SHA256,
            "dspark_model_bytes": DSPARK_MODEL_BYTES,
        },
        "topology": {
            "world_size": 2,
            "active_interface": "en5",
            "active_rail": "rdma_en5",
            "expected_link_bps": 80_000_000_000,
            "global_tb5_80g_evidence_only": True,
            "en5_tb5_correlation_required_for_dspark": True,
            "device_matrix": EXPECTED_TRANSPORT_MATRIX,
        },
        "preflight": {
            "pass": not all_errors,
            "reports": reports,
            "errors": all_errors,
            "adapter_wired_on_both_ranks": adapter_wired,
            "en5_tb5_correlated_on_both_ranks": en5_tb5_correlated,
        },
        "accepted_flags": ACCEPTED_FLAGS,
        "actions": action_contract,
        "action_prerequisite_errors": action_prerequisites,
        "dspark_canary": _abba_plan(),
        "factorized_prefill_canary": _factorized_plan(),
        "rollback": {
            "trigger_on_any_gate_failure": True,
            "target": (
                "rollback-only baseline on MLX 57b87fe; DSpark, segmented SDPA, "
                "KDA decode tile, and factorized flags unset"
            ),
            "completion_sha256": ACCEPTED_COMPLETION_SHA256,
        },
        "action_readiness": action_readiness,
    }
    digest = _canonical_digest(core)
    core["plan_digest"] = digest
    core["confirmations"] = {
        action: f"CONFIRM-K3-MAINTENANCE-{action}-{digest[:12]}" for action in ACTIONS
    }
    return core


Executor = Callable[[Sequence[str], Mapping[str, str]], int]


def _execute(argv: Sequence[str], environment: Mapping[str, str]) -> int:
    child_environment = os.environ.copy()
    child_environment.update(environment)
    result = subprocess.run([*argv], env=child_environment, check=False)
    return result.returncode


def apply_action(
    inventory: Inventory,
    plan: Mapping[str, object],
    *,
    action: str,
    confirmation: str | None,
    inspected_digest: str | None,
    executor: Executor = _execute,
) -> None:
    digest = _string(plan.get("plan_digest"), "plan_digest")
    confirmations = _nested(plan, "confirmations")
    expected_confirmation = _string(confirmations.get(action), "confirmation")
    if inspected_digest != digest:
        raise CanaryError("--plan-digest must match the inspected preflight plan")
    if confirmation != expected_confirmation:
        raise CanaryError(
            f"--maintenance-confirm must exactly equal {expected_confirmation!r}"
        )
    readiness = _nested(plan, "action_readiness")
    if readiness.get(action) is not True:
        raise CanaryError(f"action {action} is not ready in this preflight plan")
    raw_action = _object(inventory.actions.get(action), f"actions.{action}")
    if raw_action.get("enabled") is not True:
        raise CanaryError(f"inventory action {action} is disabled")
    raw_steps = cast(list[object], raw_action["steps"])
    for index, raw_step in enumerate(raw_steps):
        step = _parse_action_step(raw_step, f"actions.{action}.steps[{index}]")
        for raw_path, expected in step.required_file_sha256.items():
            path = Path(raw_path)
            if not path.is_file():
                raise CanaryError(
                    f"action {action} required local file is missing: {path}"
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise CanaryError(
                    f"action {action} required local file hash mismatch: {path}"
                )
        returncode = executor(step.argv, step.environment)
        if returncode != 0:
            raise CanaryError(
                f"action {action} step {index + 1} failed with exit code {returncode}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).with_name("k3_maintenance_canary.inventory.json"),
    )
    parser.add_argument("--facts-dir", type=Path)
    parser.add_argument(
        "--facts-output-dir",
        type=Path,
        help="write a new rank0.json/rank1.json read-only preflight snapshot",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action", choices=("plan", *ACTIONS), default="plan")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--maintenance-confirm")
    parser.add_argument("--plan-digest")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    probe: Probe | None = None,
    executor: Executor = _execute,
) -> int:
    args = cast(Arguments, cast(object, _parser().parse_args(argv)))
    try:
        if args.apply and args.facts_dir is not None:
            raise CanaryError("--apply cannot use replayed --facts-dir snapshots")
        inventory = load_inventory(args.inventory)
        selected_probe = (
            probe
            if probe is not None
            else (
                facts_directory_probe(args.facts_dir)
                if args.facts_dir is not None
                else ssh_probe
            )
        )
        facts: dict[int, Mapping[str, object]] = {
            node.rank: selected_probe(node) for node in inventory.nodes
        }
        if args.facts_output_dir is not None:
            fact_paths = [
                args.facts_output_dir / f"rank{node.rank}.json"
                for node in inventory.nodes
            ]
            existing = [path for path in fact_paths if path.exists()]
            if existing:
                raise CanaryError(f"refusing to overwrite facts snapshots: {existing}")
            args.facts_output_dir.mkdir(parents=True, exist_ok=True)
            for node, path in zip(inventory.nodes, fact_paths, strict=True):
                path.write_text(
                    json.dumps(facts[node.rank], indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        plan = build_plan(inventory, facts)
        rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.write_text(rendered, encoding="utf-8")
        if args.apply:
            if args.action == "plan":
                raise CanaryError("--apply requires an explicit non-plan --action")
            apply_action(
                inventory,
                plan,
                action=args.action,
                confirmation=args.maintenance_confirm,
                inspected_digest=args.plan_digest,
                executor=executor,
            )
        elif args.maintenance_confirm is not None or args.plan_digest is not None:
            raise CanaryError(
                "confirmation and plan digest are accepted only together with --apply"
            )
        return 0 if _nested(plan, "preflight").get("pass") is True else 2
    except (CanaryError, OSError, json.JSONDecodeError) as error:
        print(f"k3 maintenance canary: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
