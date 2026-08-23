"""Hermetic tests for the fail-closed Kimi K3 frontier telemetry v4 path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "engines"
    / "mlx"
    / "generator"
    / "k3_frontier_telemetry.py"
)


def _load_module():
    name = f"k3_frontier_telemetry_test_{os.urandom(8).hex()}"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    module.RSS_SAMPLE_INTERVAL_SECONDS = 0.001
    return module


class _FakeProcess:
    def __init__(self) -> None:
        self.rss = 1_000

    def memory_info(self):
        return types.SimpleNamespace(rss=self.rss)


class _FakeMx:
    def __init__(self) -> None:
        self.active = 2_000
        self.cache = 300
        self.peak = 2_000

    def get_active_memory(self) -> int:
        return self.active

    def get_cache_memory(self) -> int:
        return self.cache

    def get_peak_memory(self) -> int:
        return self.peak


def _core(sequence: int, token: int) -> dict[str, object]:
    counters = {
        "replayssm_telemetry_revision_before": 100 + sequence * 4,
        "replayssm_telemetry_revision_after": 104 + sequence * 4,
        "replayssm_telemetry_revision_delta": 4,
        "replayssm_attempted_prepares_delta": 2,
        "replayssm_batched_prepares_delta": 1,
        "replayssm_batched_commits_delta": 1,
        "replayssm_identity_prepares_delta": 1,
        "replayssm_identity_commits_delta": 1,
        "replayssm_fallback_prepares_delta": 0,
        "replayssm_fallback_commits_delta": 0,
        "replayssm_batched_errors_delta": 0,
        "replayssm_identity_errors_delta": 0,
        "replayssm_layers_batched_delta": 69,
        "replayssm_layers_identity_committed_delta": 69,
    }
    return {
        "receipt_schema_version": 2,
        "request_sequence": sequence,
        "request_token": token,
        **counters,
    }


class TelemetryCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_module()
        self.mx = _FakeMx()
        self.process = _FakeProcess()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self.temp.name)
        self.directory.chmod(0o700)
        self.saved = {
            name: os.environ.pop(name, None)
            for name in (
                self.mod.ENABLE_ENV,
                self.mod.DIRECTORY_ENV,
                self.mod.SESSION_SHA256_ENV,
            )
        }
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        self.mod._reset_state_for_tests()
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.temp.cleanup()

    def _enable(self) -> None:
        os.environ[self.mod.ENABLE_ENV] = "1"
        os.environ[self.mod.DIRECTORY_ENV] = str(self.directory)
        os.environ[self.mod.SESSION_SHA256_ENV] = hashlib.sha256(b"session").hexdigest()
        self.mod._rss_lifetime_highwater_bytes = lambda: 5_000

    def _claim(self, index: int):
        nonce = hashlib.sha256(f"nonce-{index}".encode()).hexdigest()
        fake_psutil = types.SimpleNamespace(Process=lambda _pid: self.process)
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            return self.mod.claim(
                request_index=index,
                request_nonce_sha256=nonce,
                rank=0,
                world_size=2,
                mx_module=self.mx,
            )

    def _finish(self, index: int):
        context = self._claim(index)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        self.process.rss += 100
        self.mx.active += 50
        self.mx.cache += 10
        self.mx.peak += 100
        self.mod.observe_metal(context)
        evidence = self.mod.finalize(context, _core(index + 1, 10_000 + index))
        assert evidence is not None
        self.assertEqual(context.publication_state, "finalized-unpublished")
        self.mod.mark_published(context)
        self.assertEqual(context.publication_state, "published")
        return evidence


class TestDisabledDefault(TelemetryCase):
    def test_disabled_is_inert_and_rejects_orphan_controls(self) -> None:
        self.assertIsNone(
            self.mod.claim(
                request_index=None,
                request_nonce_sha256=None,
                rank=0,
                world_size=2,
                mx_module=self.mx,
            )
        )
        with self.assertRaisesRegex(ValueError, "headers require"):
            self.mod.claim(
                request_index=1,
                request_nonce_sha256="0" * 64,
                rank=0,
                world_size=2,
                mx_module=self.mx,
            )
        self.assertEqual(list(self.directory.iterdir()), [])


class TestSequenceAndEvidence(TelemetryCase):
    def test_post_reset_zero_peak_with_live_model_memory_is_valid(self) -> None:
        self._enable()
        context = self._claim(1)
        assert context is not None

        # MLX reset_peak_memory() sets the request peak to zero without
        # releasing the already-loaded model represented by active memory.
        self.mx.peak = 0
        self.mod.capture_reset_baseline(context)

        assert context.baseline is not None
        self.assertEqual(context.baseline.metal_active_bytes, 2_000)
        self.assertEqual(context.baseline.metal_cache_bytes, 300)
        self.assertEqual(context.baseline.metal_residency_bytes, 2_300)
        self.assertEqual(context.baseline.metal_active_peak_bytes, 0)
        self.mod.abort(context)

    def test_one_request_is_sanitized_and_hash_bound(self) -> None:
        self._enable()
        evidence = self._finish(1)
        path = self.directory / "request-01-rank0.json"
        raw = path.read_bytes()
        row = json.loads(raw)
        self.assertEqual(set(row), self.mod.REQUEST_EVIDENCE_KEYS)
        self.assertEqual(row["schema"], self.mod.SCHEMA)
        self.assertEqual(row["request_index"], 1)
        self.assertEqual(row["reset_generation"], 1)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), evidence.request_file_sha256)
        unsigned = dict(row)
        evidence_sha = unsigned.pop("evidence_sha256")
        self.assertEqual(evidence_sha, self.mod._sha256(unsigned))
        self.assertEqual(
            row["metal_active_peak_semantics"],
            "mlx-reset-scoped-authoritative",
        )
        self.assertEqual(
            row["metal_residency_peak_semantics"],
            "bounded-lifecycle-observed",
        )
        memory_values = {
            key: value
            for key, value in row.items()
            if key.startswith(("process_rss_", "rss_sample_", "metal_"))
            and not key.endswith("_semantics")
        }
        self.assertTrue(memory_values)
        self.assertTrue(all(type(value) is int for value in memory_values.values()))
        forbidden = ("prompt", "completion", "environment", "hostname", "path")
        self.assertFalse(
            any(term in key for key in row for term in forbidden),
            sorted(row),
        )
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_overlap_and_sequence_gap_poison(self) -> None:
        self._enable()
        first = self._claim(1)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            self._claim(2)
        self.mod.abort(first)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            self._claim(1)

    def test_first_claim_requires_exactly_empty_directory(self) -> None:
        self._enable()
        (self.directory / "stale").write_text("stale")
        with self.assertRaisesRegex(RuntimeError, "exact live sequence"):
            self._claim(1)

    def test_second_claim_rejects_extra_prior_inventory(self) -> None:
        self._enable()
        self._finish(1)
        (self.directory / "extra").write_text("extra")
        with self.assertRaisesRegex(RuntimeError, "exact live sequence"):
            self._claim(2)

    def test_second_claim_rejects_tampered_prior_inventory(self) -> None:
        self._enable()
        self._finish(1)
        target = self.directory / "request-01-rank0.json"
        target.write_bytes(b"tampered")
        target.chmod(0o600)
        with self.assertRaisesRegex(RuntimeError, "digest drifted"):
            self._claim(2)

    def test_finalize_rejects_extra_added_after_claim(self) -> None:
        self._enable()
        context = self._claim(1)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        (self.directory / "late-extra").write_text("late")
        with self.assertRaisesRegex(RuntimeError, "exact live sequence"):
            self.mod.finalize(context, _core(2, 10_001))
        self.assertFalse((self.directory / "request-01-rank0.json").exists())

    def test_finalized_unpublished_abort_poison_cannot_be_ignored(self) -> None:
        self._enable()
        context = self._claim(1)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        evidence = self.mod.finalize(context, _core(2, 10_001))
        self.assertIsNotNone(evidence)
        self.assertEqual(context.publication_state, "finalized-unpublished")
        self.mod.abort(context)
        self.assertEqual(context.publication_state, "poisoned")
        with self.assertRaisesRegex(RuntimeError, "overlap, replay, or sequence gap"):
            self._claim(2)

    def test_published_context_allows_exact_next_claim(self) -> None:
        self._enable()
        context = self._claim(1)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        self.assertIsNotNone(self.mod.finalize(context, _core(2, 10_001)))
        self.mod.mark_published(context)
        self.mod.abort(context)
        second = self._claim(2)
        self.assertIsNotNone(second)
        self.mod.abort(second)

    def test_o_excl_collision_fails_closed_without_overwrite(self) -> None:
        self._enable()
        context = self._claim(1)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        target = self.directory / "request-01-rank0.json"
        target.write_bytes(b"owned")
        target.chmod(0o600)
        with self.assertRaises(FileExistsError):
            self.mod.finalize(context, _core(2, 10_001))
        self.assertEqual(target.read_bytes(), b"owned")

    def test_post_create_failure_unlinks_authenticated_inode(self) -> None:
        self._enable()
        context = self._claim(1)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        original_inventory = self.mod._require_exact_inventory_fd
        calls = 0

        def fail_after_create(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected post-create inventory failure")
            return original_inventory(*args, **kwargs)

        directory_fsyncs: list[int] = []
        original_fsync = self.mod.os.fsync

        def observe_fsync(fd: int) -> None:
            info = os.fstat(fd)
            if pathlib.Path(self.directory).stat().st_ino == info.st_ino:
                directory_fsyncs.append(fd)
            original_fsync(fd)

        with (
            mock.patch.object(
                self.mod,
                "_require_exact_inventory_fd",
                side_effect=fail_after_create,
            ),
            mock.patch.object(self.mod.os, "fsync", side_effect=observe_fsync),
            self.assertRaisesRegex(
                RuntimeError,
                "injected post-create inventory failure",
            ),
        ):
            self.mod.finalize(context, _core(2, 10_001))
        stale = self.directory / "request-01-rank0.json"
        self.assertFalse(stale.exists())
        self.assertTrue(directory_fsyncs)
        with self.assertRaisesRegex(RuntimeError, "overlap, replay, or sequence gap"):
            self._claim(2)

    def test_replaced_name_is_never_unlinked_by_failed_writer(self) -> None:
        self._enable()
        context = self._claim(1)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        target = self.directory / "request-01-rank0.json"
        displaced = self.directory / "created-but-displaced.json"
        original_open = self.mod.os.open
        name_opens = 0

        def replace_before_reopen(path, flags, *args, **kwargs):
            nonlocal name_opens
            if path == target.name:
                name_opens += 1
                if name_opens == 2:
                    target.rename(displaced)
                    target.write_bytes(b"replacement")
                    target.chmod(0o600)
            return original_open(path, flags, *args, **kwargs)

        with (
            mock.patch.object(self.mod.os, "open", side_effect=replace_before_reopen),
            self.assertRaisesRegex(
                RuntimeError,
                "cleanup failed|inode is invalid|sequence|verification failed",
            ),
        ):
            self.mod.finalize(context, _core(2, 10_001))
        self.assertEqual(target.read_bytes(), b"replacement")
        self.assertTrue(displaced.exists())

    def test_sixteen_requests_publish_one_same_process_aggregate(self) -> None:
        self._enable()
        evidence = None
        for index in range(1, 17):
            evidence = self._finish(index)
            if index < 16:
                self.assertIsNone(evidence.process_complete_file_sha256)
        assert evidence is not None
        self.assertIsNotNone(evidence.process_complete_file_sha256)
        aggregate_path = self.directory / "process-complete-rank0.json"
        aggregate_raw = aggregate_path.read_bytes()
        aggregate = json.loads(aggregate_raw)
        self.assertEqual(set(aggregate), self.mod.PROCESS_EVIDENCE_KEYS)
        self.assertTrue(
            all(
                set(row) == self.mod.PROCESS_REQUEST_ROW_KEYS
                for row in aggregate["requests"]
            )
        )
        self.assertEqual(aggregate["schema"], self.mod.PROCESS_SCHEMA)
        self.assertEqual(aggregate["request_count"], 16)
        self.assertEqual(
            [row["request_index"] for row in aggregate["requests"]],
            list(range(1, 17)),
        )
        self.assertEqual(
            len({row["request_nonce_sha256"] for row in aggregate["requests"]}),
            16,
        )
        self.assertEqual(
            hashlib.sha256(aggregate_raw).hexdigest(),
            evidence.process_complete_file_sha256,
        )
        self.assertEqual(len(list(self.directory.iterdir())), 17)

    def test_request_sixteen_marker_failure_poison_after_process_file(self) -> None:
        self._enable()
        for index in range(1, 16):
            self._finish(index)
        context = self._claim(16)
        assert context is not None
        self.mod.capture_reset_baseline(context)
        evidence = self.mod.finalize(context, _core(17, 10_016))
        assert evidence is not None
        self.assertIsNotNone(evidence.process_complete_file_sha256)
        self.assertTrue((self.directory / "process-complete-rank0.json").exists())
        self.mod.abort(context)
        self.assertEqual(context.publication_state, "poisoned")
        with self.assertRaisesRegex(RuntimeError, "publication state is invalid"):
            self.mod.mark_published(context)
        with self.assertRaisesRegex(RuntimeError, "overlap, replay, or sequence gap"):
            self._claim(16)


if __name__ == "__main__":
    unittest.main()
