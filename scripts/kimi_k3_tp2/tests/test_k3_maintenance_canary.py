from __future__ import annotations

import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import k3_maintenance_canary as canary  # noqa: E402


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, dict)
    raw = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in raw)
    return cast(Mapping[str, object], raw)


def _mutable_mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], _mapping(value))


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def _json_mapping(text: str) -> Mapping[str, object]:
    return _mapping(cast(object, json.loads(text)))


def _actions() -> dict[str, object]:
    actions: dict[str, object] = {}
    for action in canary.ACTIONS:
        enabled = action in {"stop", "start-baseline", "rollback"}
        step_count = 2 if action == "start-baseline" else 1
        actions[action] = {
            "enabled": enabled,
            "reason": f"test contract for {action}",
            "steps": (
                [
                    {
                        "argv": ["/usr/bin/true", action, str(index)],
                        "environment": {"K3_TEST_ACTION": action},
                    }
                    for index in range(step_count)
                ]
                if enabled
                else []
            ),
        }
    return actions


def _inventory_data() -> dict[str, object]:
    common_paths = {
        "exo_root": "/staged/exo",
        "mlx_lm_root": "/staged/mlx-lm",
        "mlx_root": "/staged/mlx",
        "factorized_mlx_root": "/staged/factorized-mlx",
        "dspark_checkpoint": canary.DSPARK_PATH,
        "transport_contract": "/staged/transport.json",
        "artifact_root": "/staged/artifacts",
    }
    return {
        "schema": canary.INVENTORY_SCHEMA,
        "nodes": [
            {
                "name": "512S1",
                "rank": 0,
                "ssh": ["/usr/bin/ssh", "rank0"],
                "paths": {
                    **common_paths,
                    "rank_checkpoint": "/models/rank0",
                },
                "rank_manifest_sha256": "0" * 64,
            },
            {
                "name": "512S2",
                "rank": 1,
                "ssh": ["/usr/bin/ssh", "rank1"],
                "paths": {
                    **common_paths,
                    "rank_checkpoint": "/models/rank1",
                },
                "rank_manifest_sha256": "1" * 64,
            },
        ],
        "actions": _actions(),
    }


def _write_inventory(tmp_path: Path, data: object | None = None) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(_inventory_data() if data is None else data),
        encoding="utf-8",
    )
    return path


def _facts(
    node: canary.Node,
    *,
    adapter_wired: bool = False,
    factorized_wire_is_ancestor: bool = True,
    en5_tb5_correlated: bool = False,
) -> dict[str, object]:
    return {
        "schema": canary.FACTS_SCHEMA,
        "node": node.name,
        "rank": node.rank,
        "commits": {
            "exo": canary.EXO_DSPARK_CONTRACT_COMMIT,
            "mlx_lm": canary.MLX_LM_DSPARK_COMMIT,
            "mlx": canary.MLX_ACCEPTED_COMMIT,
            "factorized_mlx": canary.MLX_FACTORIZED_COMMIT,
        },
        "git_clean": {
            "exo": True,
            "mlx_lm": True,
            "mlx": True,
            "factorized_mlx": True,
        },
        "mlx_lm_factorized_wire_is_ancestor": factorized_wire_is_ancestor,
        "required_files": {
            "exo_dspark_contract": True,
            "exo_generate": True,
            "mlx_lm_dspark": True,
            "mlx_lm_kimi_k3": True,
            "mlx_lm_gated_delta": True,
            "factorized_kernel": True,
            "rank_manifest": True,
            "transport": True,
        },
        "mlx_lm_source_sha256": {
            "mlx_lm_kimi_k3": canary.MLX_LM_KIMI_K3_SHA256,
            "mlx_lm_dspark": canary.MLX_LM_KIMI_K3_DSPARK_SHA256,
            "mlx_lm_gated_delta": canary.MLX_LM_GATED_DELTA_SHA256,
        },
        "dspark": {
            "path": canary.DSPARK_PATH,
            "config_sha256": canary.DSPARK_CONFIG_SHA256,
            "model_sha256": canary.DSPARK_MODEL_SHA256,
            "model_bytes": canary.DSPARK_MODEL_BYTES,
        },
        "rank_manifest_sha256": node.rank_manifest_sha256,
        "transport": {"device_matrix": canary.EXPECTED_TRANSPORT_MATRIX},
        "interfaces": {
            interface: {
                "exists": True,
                "active": interface == "en5",
            }
            for interface in ("en3", "en4", "en5", "en6")
        },
        "global_tb5_80g_present": True,
        "en5_tb5_correlated": en5_tb5_correlated,
        "ram_bytes": canary.MIN_RAM_BYTES,
        "disk_free_bytes": canary.MIN_DISK_FREE_BYTES,
        "adapter_wired": adapter_wired,
    }


def _valid_plan(
    tmp_path: Path,
    *,
    adapter_wired: bool = False,
) -> tuple[canary.Inventory, dict[str, object]]:
    inventory = canary.load_inventory(_write_inventory(tmp_path))
    facts = {
        node.rank: _facts(node, adapter_wired=adapter_wired) for node in inventory.nodes
    }
    return inventory, canary.build_plan(inventory, facts)


def test_plan_pins_abba_gates_and_disabled_unwired_actions(
    tmp_path: Path,
) -> None:
    _, plan = _valid_plan(tmp_path)

    assert plan["pins"] == {
        "exo_dspark_contract": canary.EXO_DSPARK_CONTRACT_COMMIT,
        "mlx_lm_dspark": canary.MLX_LM_DSPARK_COMMIT,
        "mlx_lm_kimi_k3_sha256": canary.MLX_LM_KIMI_K3_SHA256,
        "mlx_lm_kimi_k3_dspark_sha256": canary.MLX_LM_KIMI_K3_DSPARK_SHA256,
        "mlx_lm_gated_delta_sha256": canary.MLX_LM_GATED_DELTA_SHA256,
        "mlx_rollback_only_decode": canary.MLX_ACCEPTED_COMMIT,
        "mlx_factorized_rollback_only": canary.MLX_FACTORIZED_COMMIT,
        "mlx_lm_factorized_wire": canary.MLX_LM_FACTORIZED_WIRE_COMMIT,
        "dspark_revision": canary.DSPARK_REVISION,
        "dspark_path": canary.DSPARK_PATH,
        "dspark_config_sha256": canary.DSPARK_CONFIG_SHA256,
        "dspark_model_sha256": canary.DSPARK_MODEL_SHA256,
        "dspark_model_bytes": canary.DSPARK_MODEL_BYTES,
    }
    preflight = _mapping(plan["preflight"])
    assert preflight["pass"] is True
    assert preflight["adapter_wired_on_both_ranks"] is False
    assert preflight["en5_tb5_correlated_on_both_ranks"] is False
    readiness = _mapping(plan["action_readiness"])
    assert readiness["start-baseline"] is True
    assert readiness["start-dspark-width3"] is False
    assert readiness["start-dspark-width8"] is False
    accepted_flags = _mapping(plan["accepted_flags"])
    assert accepted_flags["EXO_MLX_K3_VOCAB_PARALLEL_GREEDY"] == "1"
    assert accepted_flags["MLX_LM_KIMI_K3_DSPARK_SEGMENTED_SDPA"] == "0"
    assert accepted_flags["MLX_LM_EXPERIMENTAL_KDA_ROW_DECODE"] == "0"

    phases = _list(plan["dspark_canary"])
    verifier = _mapping(phases[0])
    assert verifier["widths"] == [2, 3, 8]
    assert _mapping(verifier["gates"])["minimum_speedup_by_width"] == {
        "2": 1.20,
        "3": 1.45,
        "8": 2.00,
    }
    for raw_phase in phases[1:]:
        phase = _mapping(raw_phase)
        assert [_mapping(leg)["leg"] for leg in _list(phase["sequence"])] == [
            "A1",
            "B1",
            "B2",
            "A2",
        ]
    width3 = _mapping(_list(_mapping(phases[1])["sequence"])[1])
    width8 = _mapping(_list(_mapping(phases[2])["sequence"])[1])
    assert _mapping(width3["environment"])["EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH"] == "3"
    assert _mapping(width3["gates"])["minimum_mean_emitted_per_round"] == 2.391
    assert _mapping(width8["environment"])["EXO_MLX_KIMI_K3_DSPARK_VERIFY_WIDTH"] == "8"

    factorized = _mapping(plan["factorized_prefill_canary"])
    assert factorized["separate_from_dspark"] is True
    assert factorized["cases"] == [
        {"query_tokens": 32, "context_tokens": 512},
        {"query_tokens": 32, "context_tokens": 4096},
        {"query_tokens": 32, "context_tokens": 8192},
    ]


def test_checked_in_inventory_is_candidate_pinned_and_legacy_actions_are_rollback_only() -> (
    None
):
    inventory = canary.load_inventory(
        SCRIPT_DIR / "k3_maintenance_canary.inventory.json"
    )

    assert inventory.nodes[0].paths["exo_root"].endswith("6eb59a4")
    assert inventory.nodes[0].paths["mlx_lm_root"].endswith("1bcf430")
    for action in ("start-baseline", "rollback"):
        contract = _mapping(inventory.actions[action])
        assert "rollback-only" in str(contract["reason"]).lower()
        for raw_step in _list(contract["steps"]):
            environment = _mapping(_mapping(raw_step)["environment"])
            if "K3_REMOTE_EXO_SOURCE" in environment:
                assert environment["MLX_LM_KIMI_K3_DSPARK_SEGMENTED_SDPA"] == "0"
                assert environment["MLX_LM_EXPERIMENTAL_KDA_ROW_DECODE"] == "0"
    for action in ("start-dspark-width3", "start-dspark-width8"):
        contract = _mapping(inventory.actions[action])
        assert contract["enabled"] is False
        assert "hardware canary" in str(contract["reason"]).lower()


def test_dspark_readiness_requires_en5_to_tb5_correlation(tmp_path: Path) -> None:
    data = _inventory_data()
    actions = _mutable_mapping(data["actions"])
    dspark = _mutable_mapping(actions["start-dspark-width3"])
    dspark["enabled"] = True
    dspark["steps"] = [{"argv": ["/usr/bin/true"], "environment": {}}]
    inventory = canary.load_inventory(_write_inventory(tmp_path, data))
    uncorrelated = {
        node.rank: _facts(node, adapter_wired=True) for node in inventory.nodes
    }
    correlated = {
        node.rank: _facts(
            node,
            adapter_wired=True,
            en5_tb5_correlated=True,
        )
        for node in inventory.nodes
    }

    uncorrelated_readiness = _mapping(
        canary.build_plan(inventory, uncorrelated)["action_readiness"]
    )
    correlated_readiness = _mapping(
        canary.build_plan(inventory, correlated)["action_readiness"]
    )

    assert uncorrelated_readiness["start-dspark-width3"] is False
    assert correlated_readiness["start-dspark-width3"] is True


def test_preflight_rejects_missing_factorized_wire_ancestor(tmp_path: Path) -> None:
    inventory = canary.load_inventory(_write_inventory(tmp_path))
    facts = {
        node.rank: _facts(node, factorized_wire_is_ancestor=False)
        for node in inventory.nodes
    }

    preflight = _mapping(canary.build_plan(inventory, facts)["preflight"])

    assert preflight["pass"] is False
    ancestor_errors = [
        str(error)
        for error in _list(preflight["errors"])
        if "factorized wire commit" in str(error)
    ]
    assert len(ancestor_errors) == 2
    assert all(
        canary.MLX_LM_FACTORIZED_WIRE_COMMIT in str(error) for error in ancestor_errors
    )


def test_preflight_rejects_wrong_candidate_runtime_source_hash(tmp_path: Path) -> None:
    inventory = canary.load_inventory(_write_inventory(tmp_path))
    facts = {node.rank: _facts(node) for node in inventory.nodes}
    source_hashes = _mutable_mapping(facts[0]["mlx_lm_source_sha256"])
    source_hashes["mlx_lm_gated_delta"] = "0" * 64

    preflight = _mapping(canary.build_plan(inventory, facts)["preflight"])

    assert preflight["pass"] is False
    assert any(
        "mlx_lm_gated_delta SHA-256" in str(error)
        for error in _list(preflight["errors"])
    )


def test_action_digest_binds_exact_command_contract(tmp_path: Path) -> None:
    _, first = _valid_plan(tmp_path)
    data = _inventory_data()
    actions = _mapping(data["actions"])
    baseline = _mapping(actions["start-baseline"])
    steps = _list(baseline["steps"])
    step = _mapping(steps[0])
    argv = _list(step["argv"])
    argv.append("changed-after-inspection")
    second_inventory = canary.load_inventory(_write_inventory(tmp_path, data))
    second = canary.build_plan(
        second_inventory,
        {node.rank: _facts(node) for node in second_inventory.nodes},
    )

    assert first["plan_digest"] != second["plan_digest"]
    assert first["confirmations"] != second["confirmations"]


def test_default_main_is_preflight_only_and_never_executes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = _write_inventory(tmp_path)
    inventory = canary.load_inventory(inventory_path)
    facts = {node.rank: _facts(node) for node in inventory.nodes}
    executions: list[tuple[Sequence[str], Mapping[str, str]]] = []

    result = canary.main(
        ["--inventory", str(inventory_path)],
        probe=lambda node: facts[node.rank],
        executor=lambda argv, environment: executions.append((argv, environment)) or 0,
    )

    assert result == 0
    assert executions == []
    rendered = _json_mapping(capsys.readouterr().out)
    assert rendered["default_mode"] == "preflight-only"


def test_apply_requires_exact_digest_and_action_confirmation(
    tmp_path: Path,
) -> None:
    inventory, plan = _valid_plan(tmp_path)
    executions: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def execute(argv: Sequence[str], environment: Mapping[str, str]) -> int:
        executions.append((tuple(argv), dict(environment)))
        return 0

    with pytest.raises(canary.CanaryError, match="plan-digest"):
        canary.apply_action(
            inventory,
            plan,
            action="start-baseline",
            confirmation=None,
            inspected_digest="wrong",
            executor=execute,
        )
    with pytest.raises(canary.CanaryError, match="maintenance-confirm"):
        canary.apply_action(
            inventory,
            plan,
            action="start-baseline",
            confirmation="wrong",
            inspected_digest=str(plan["plan_digest"]),
            executor=execute,
        )
    assert executions == []

    confirmations = _mapping(plan["confirmations"])
    canary.apply_action(
        inventory,
        plan,
        action="start-baseline",
        confirmation=str(confirmations["start-baseline"]),
        inspected_digest=str(plan["plan_digest"]),
        executor=execute,
    )
    assert [execution[0][-1] for execution in executions] == ["0", "1"]


def test_apply_rejects_disabled_dspark_even_when_adapter_is_wired(
    tmp_path: Path,
) -> None:
    inventory, plan = _valid_plan(tmp_path, adapter_wired=True)
    assert _mapping(plan["preflight"])["adapter_wired_on_both_ranks"] is True
    assert _mapping(plan["action_readiness"])["start-dspark-width8"] is False
    confirmations = _mapping(plan["confirmations"])
    with pytest.raises(canary.CanaryError, match="not ready"):
        canary.apply_action(
            inventory,
            plan,
            action="start-dspark-width8",
            confirmation=str(confirmations["start-dspark-width8"]),
            inspected_digest=str(plan["plan_digest"]),
        )


def test_failed_topology_blocks_start_but_retains_confirmed_stop(
    tmp_path: Path,
) -> None:
    inventory = canary.load_inventory(_write_inventory(tmp_path))
    facts_by_rank = {node.rank: _facts(node) for node in inventory.nodes}
    interfaces = _mutable_mapping(facts_by_rank[1]["interfaces"])
    en5 = _mutable_mapping(interfaces["en5"])
    en5["active"] = False
    plan = canary.build_plan(inventory, facts_by_rank)

    preflight = _mapping(plan["preflight"])
    readiness = _mapping(plan["action_readiness"])
    assert preflight["pass"] is False
    assert any(
        "en5 must be active" in str(error) for error in _list(preflight["errors"])
    )
    assert readiness["start-baseline"] is False
    assert readiness["rollback"] is False
    assert readiness["stop"] is True


def test_main_apply_missing_confirmation_does_not_execute(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = _write_inventory(tmp_path)
    inventory = canary.load_inventory(inventory_path)
    facts = {node.rank: _facts(node) for node in inventory.nodes}
    executions: list[Sequence[str]] = []

    result = canary.main(
        [
            "--inventory",
            str(inventory_path),
            "--action",
            "start-baseline",
            "--apply",
        ],
        probe=lambda node: facts[node.rank],
        executor=lambda argv, _environment: executions.append(argv) or 0,
    )

    assert result == 2
    assert executions == []
    assert "--plan-digest" in capsys.readouterr().err


def test_facts_snapshot_replays_the_same_plan_digest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = _write_inventory(tmp_path)
    inventory = canary.load_inventory(inventory_path)
    facts = {node.rank: _facts(node) for node in inventory.nodes}
    snapshot = tmp_path / "facts"

    assert (
        canary.main(
            [
                "--inventory",
                str(inventory_path),
                "--facts-output-dir",
                str(snapshot),
            ],
            probe=lambda node: facts[node.rank],
        )
        == 0
    )
    live_plan = _json_mapping(capsys.readouterr().out)
    assert (
        canary.main(
            [
                "--inventory",
                str(inventory_path),
                "--facts-dir",
                str(snapshot),
            ]
        )
        == 0
    )
    replayed_plan = _json_mapping(capsys.readouterr().out)
    assert live_plan["plan_digest"] == replayed_plan["plan_digest"]

    assert (
        canary.main(
            [
                "--inventory",
                str(inventory_path),
                "--facts-output-dir",
                str(snapshot),
            ],
            probe=lambda node: facts[node.rank],
        )
        == 2
    )
    assert "refusing to overwrite" in capsys.readouterr().err


def test_inventory_rejects_unpinned_dspark_path(tmp_path: Path) -> None:
    data = copy.deepcopy(_inventory_data())
    nodes = _list(data["nodes"])
    node = _mapping(nodes[0])
    paths = _mutable_mapping(node["paths"])
    paths["dspark_checkpoint"] = "/tmp/follows-main"

    with pytest.raises(canary.CanaryError, match="DSpark path must be pinned"):
        canary.load_inventory(_write_inventory(tmp_path, data))


def test_apply_rejects_replayed_facts_before_probe_or_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = _write_inventory(tmp_path)
    probes: list[canary.Node] = []
    executions: list[Sequence[str]] = []

    result = canary.main(
        [
            "--inventory",
            str(inventory_path),
            "--facts-dir",
            str(tmp_path / "stale"),
            "--action",
            "stop",
            "--apply",
            "--plan-digest",
            "0" * 64,
            "--maintenance-confirm",
            "stale",
        ],
        probe=lambda node: probes.append(node) or _facts(node),
        executor=lambda argv, _environment: executions.append(argv) or 0,
    )

    assert result == 2
    assert probes == []
    assert executions == []
    assert "cannot use replayed" in capsys.readouterr().err
