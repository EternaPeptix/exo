from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from exo.worker.engines.mlx.generator import kimi_k3_width4_receipt as subject


class _FakeMlxModule:
    __file__: str | None

    def __init__(self, path: str | None) -> None:
        self.__file__ = path


def _python_receipt(**totals: int) -> dict[str, object]:
    values = {
        "attempted": 3,
        "supported": 3,
        "dispatched": 3,
        "fallback": 0,
        "error": 0,
    }
    values.update(totals)
    return {
        "schema_version": 3,
        "generation": 1,
        "enabled": True,
        "totals": values,
        "stale_completions": {"total": 0},
    }


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        subject.WIDTH4_RECEIPT_LOG_ENV,
        subject.MLX_LM_WIDTH4_RECEIPT_ENV,
        subject.MLX_Q4_RECEIPT_ENV,
    ):
        monkeypatch.setenv(name, "1")
    subject.width4_receipt_log_enabled.cache_clear()


def test_default_off_does_not_touch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(subject.WIDTH4_RECEIPT_LOG_ENV, raising=False)
    subject.width4_receipt_log_enabled.cache_clear()
    assert (
        subject.capture_width4_dispatch_receipt(
            rank=0,
            mx_module=_FakeMlxModule(None),
            process_id=1,
            host="512S1.local",
        )
        is None
    )


def test_selector_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(subject.WIDTH4_RECEIPT_LOG_ENV, "true")
    subject.width4_receipt_log_enabled.cache_clear()
    with pytest.raises(ValueError, match="must be exactly"):
        subject.width4_receipt_log_enabled()


def test_child_selectors_are_mandatory(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv(subject.MLX_Q4_RECEIPT_ENV, "0")
    with pytest.raises(subject.Width4ReceiptError, match="child selectors"):
        subject.capture_width4_dispatch_receipt(
            rank=0,
            mx_module=_FakeMlxModule(None),
            process_id=1,
            host="512S1.local",
        )


@pytest.mark.parametrize(
    "totals",
    (
        {"attempted": 2},
        {"supported": 4},
        {"supported": 0, "dispatched": 1, "fallback": 2},
        {"error": -1, "dispatched": 4},
    ),
)
def test_impossible_python_receipt_is_rejected(totals: dict[str, int]) -> None:
    with pytest.raises(subject.Width4ReceiptError):
        subject.validate_width4_python_receipt(_python_receipt(**totals))


def test_stale_generation_completion_is_rejected() -> None:
    receipt = _python_receipt()
    receipt["stale_completions"] = {"total": 1}
    with pytest.raises(subject.Width4ReceiptError, match="stale"):
        subject.validate_width4_python_receipt(receipt)


def test_valid_but_nonpassing_python_receipt_fails_service_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    core = tmp_path / "mlx" / "core.cpython-313-darwin.so"
    libmlx = tmp_path / "mlx" / "lib" / "libmlx.dylib"
    libmlx.parent.mkdir(parents=True)
    core.write_bytes(b"core")
    libmlx.write_bytes(b"libmlx")

    class _Getter:
        argtypes: list[object] | None = None
        restype: object | None = None

        def __call__(self) -> int:
            return 1

    def fake_cdll(_path: str) -> SimpleNamespace:
        return SimpleNamespace(mlx_k3_affine6_q4_quad_dispatch_count=_Getter())

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    fused_expert = SimpleNamespace(
        snapshot_k3_width4_dispatch_receipt=lambda: _python_receipt(
            supported=0,
            dispatched=0,
            fallback=3,
        )
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_lm.models", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.kimi_k3_fused_expert",
        fused_expert,
    )
    with pytest.raises(subject.Width4ReceiptError, match="no dispatch"):
        subject.capture_width4_dispatch_receipt(
            rank=0,
            mx_module=_FakeMlxModule(str(core)),
            process_id=1,
            host="512S1.local",
        )


def test_capture_binds_python_and_native_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    core = tmp_path / "mlx" / "core.cpython-313-darwin.so"
    libmlx = tmp_path / "mlx" / "lib" / "libmlx.dylib"
    libmlx.parent.mkdir(parents=True)
    core.write_bytes(b"core")
    libmlx.write_bytes(b"libmlx")

    class _Getter:
        argtypes: list[object] | None = None
        restype: object | None = None

        def __call__(self) -> int:
            return 17

    getter = _Getter()

    def fake_cdll(_path: str) -> SimpleNamespace:
        return SimpleNamespace(mlx_k3_affine6_q4_quad_dispatch_count=getter)

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    fused_expert = SimpleNamespace(
        snapshot_k3_width4_dispatch_receipt=lambda: _python_receipt()
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_lm.models", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.kimi_k3_fused_expert",
        fused_expert,
    )
    receipt = subject.capture_width4_dispatch_receipt(
        rank=1,
        mx_module=_FakeMlxModule(str(core)),
        process_id=8123,
        host="512S2.local",
    )
    assert receipt is not None
    assert receipt["rank"] == 1
    assert receipt["process_id"] == 8123
    assert receipt["host"] == "512S2.local"
    assert receipt["native_q4_dispatch_count"] == 17
    assert receipt["libmlx_path"] == str(libmlx)
    assert receipt["terminal_output_consumed"] is True
    assert getter.argtypes == []
    assert getter.restype is ctypes.c_uint64
    rendered = subject.format_width4_dispatch_receipt(receipt)
    assert rendered.startswith(subject.RECEIPT_MARKER)
    assert json.loads(rendered.removeprefix(subject.RECEIPT_MARKER)) == receipt


def test_missing_native_symbol_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    core = tmp_path / "mlx" / "core.cpython-313-darwin.so"
    libmlx = tmp_path / "mlx" / "lib" / "libmlx.dylib"
    libmlx.parent.mkdir(parents=True)
    core.write_bytes(b"core")
    libmlx.write_bytes(b"libmlx")

    def fake_cdll(_path: str) -> SimpleNamespace:
        return SimpleNamespace()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    fused_expert = SimpleNamespace(
        snapshot_k3_width4_dispatch_receipt=lambda: _python_receipt()
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_lm.models", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.kimi_k3_fused_expert",
        fused_expert,
    )
    with pytest.raises(subject.Width4ReceiptError, match="lacks"):
        subject.capture_width4_dispatch_receipt(
            rank=0,
            mx_module=_FakeMlxModule(str(core)),
            process_id=1,
            host="512S1.local",
        )
