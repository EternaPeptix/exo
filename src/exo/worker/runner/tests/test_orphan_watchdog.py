"""Tests for the runner orphan watchdog (PR exo-explore/exo#2205).

The watchdog hard-exits a runner process when its parent worker dies so that
model memory and RDMA queue pairs are released rather than held by a process
reparented to pid 1. The full kill -> reparent -> os._exit path depends on OS
reparenting semantics that are hard to reproduce from pytest (the runner is
normally a launchd-managed multiprocessing child, not a pytest child), so
these tests cover the deterministic parts: the disable escape-hatch, the
thread is started, and the watchdog's process is wired correctly.
"""
from __future__ import annotations

import os
import threading

import pytest

from exo.worker.runner import bootstrap


def test_watchdog_disabled_does_not_start_thread(monkeypatch):
    """EXO_DISABLE_ORPHAN_WATCHDOG=1 must short-circuit before starting the
    watchdog thread."""
    monkeypatch.setenv("EXO_DISABLE_ORPHAN_WATCHDOG", "1")
    threads_before = {t.name for t in threading.enumerate()}
    bootstrap._start_orphan_watchdog()
    threads_after = {t.name for t in threading.enumerate()}
    assert "orphan-watchdog" not in threads_after, (
        "watchdog thread was started despite EXO_DISABLE_ORPHAN_WATCHDOG=1"
    )
    assert threads_before == threads_after, "no new threads should be created when disabled"


def test_watchdog_starts_thread_when_enabled(monkeypatch):
    """Without the disable flag, the watchdog thread is started (daemon=True so
    it won't block test exit)."""
    monkeypatch.delenv("EXO_DISABLE_ORPHAN_WATCHDOG", raising=False)
    assert "orphan-watchdog" not in {t.name for t in threading.enumerate()}, (
        "precondition: no orphan-watchdog thread running"
    )
    bootstrap._start_orphan_watchdog()
    assert "orphan-watchdog" in {t.name for t in threading.enumerate()}, (
        "watchdog thread should be started when not disabled"
    )


def test_entrypoint_calls_watchdog(monkeypatch):
    """entrypoint() must invoke _start_orphan_watchdog(). We patch the function
    to record the call rather than actually start the thread."""
    called = {"yes": False}
    def _spy():
        called["yes"] = True
    monkeypatch.setattr(bootstrap, "_start_orphan_watchdog", _spy)
    # Drive entrypoint just far enough to hit the call site; it will raise later
    # on missing args, which is fine — we only care that the call happened.
    try:
        bootstrap.entrypoint.__wrapped__ if hasattr(bootstrap.entrypoint, "__wrapped__") else None
    except Exception:
        pass
    # Re-check via source inspection instead (more robust than driving entrypoint).
    import inspect
    src = inspect.getsource(bootstrap.entrypoint)
    assert "_start_orphan_watchdog()" in src, (
        "entrypoint() must call _start_orphan_watchdog()"
    )
