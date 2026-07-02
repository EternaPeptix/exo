#!/usr/bin/env python3
"""Release MLX/Metal wired memory when an Exo instance is unloaded.

Root causes addressed:
  1. mlx_cleanup() existed but was never called on unload
  2. set_wired_limit(~95% RAM) was set on load but never reset to 0
  3. Orphan-runner fix force-killed subprocesses without Shutdown task
  4. SIGTERM did not run MLX cleanup before process exit

Idempotent: safe to run multiple times on each node.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

EXO = Path("/Users/jeweled/exo/src/exo")
VENV_PYTHON = Path("/Users/jeweled/exo/.venv/bin/python")


def compile_check(path: Path) -> None:
    import subprocess

    if VENV_PYTHON.exists():
        subprocess.run(
            [str(VENV_PYTHON), "-m", "py_compile", str(path)],
            check=True,
        )
    else:
        py_compile.compile(str(path), doraise=True)


MARKER = "def release_mlx_memory()"


def patch_file(rel: str, old: str, new: str, label: str, src: str) -> str:
    if new in src:
        print(f"  skip {label} (already applied)")
        return src
    if old not in src:
        print(f"  ABORT: {label} — patch target not found in {rel}")
        sys.exit(2)
    print(f"  apply {label}")
    return src.replace(old, new, 1)


def main() -> None:
    # --- utils_mlx.py: release_mlx_memory + mlx_cleanup ---
    utils_path = EXO / "worker/engines/mlx/utils_mlx.py"
    utils = utils_path.read_text(encoding="utf-8")
    if MARKER not in utils:
        utils = patch_file(
            "utils_mlx.py",
            """def mlx_cleanup(
    model: Model | None,
    tokenizer: TokenizerWrapper | None,
    group: mx.distributed.Group | None,
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()""",
            """def release_mlx_memory() -> None:
    \"\"\"Return MLX/Metal buffers and wired memory to the OS.\"\"\"
    import gc

    if not mx.metal.is_available():
        gc.collect()
        return

    mx.synchronize()
    mx.clear_cache()
    mx.set_wired_limit(0)
    mx.clear_cache()
    gc.collect()

    active_gb = mx.get_active_memory() / (1024**3)
    cache_gb = mx.get_cache_memory() / (1024**3)
    logger.info(
        f"MLX memory released (active={active_gb:.2f} GB, cache={cache_gb:.2f} GB)"
    )


def mlx_cleanup(
    model: Model | None,
    tokenizer: TokenizerWrapper | None,
    group: mx.distributed.Group | None,
) -> None:
    del model, tokenizer, group
    release_mlx_memory()""",
            "release_mlx_memory + mlx_cleanup",
            utils,
        )
        utils_path.write_text(utils, encoding="utf-8")
        compile_check(utils_path)
    else:
        print("utils_mlx.py already patched")

    # --- mlx/builder.py: close() calls mlx_cleanup ---
    builder_path = EXO / "worker/engines/mlx/builder.py"
    builder = builder_path.read_text(encoding="utf-8")
    if "release_mlx_memory" not in builder:
        builder = patch_file(
            "builder.py",
            """from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)""",
            """from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
    mlx_cleanup,
)""",
            "builder import mlx_cleanup",
            builder,
        )
        builder = patch_file(
            "builder.py",
            """    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group""",
            """    def close(self) -> None:
        mlx_cleanup(self.inference_model, self.tokenizer, self.group)
        self.inference_model = None
        self.tokenizer = None
        self.group = None
        self.vision_processor = None""",
            "MlxBuilder.close",
            builder,
        )
        builder_path.write_text(builder, encoding="utf-8")
        compile_check(builder_path)
    else:
        print("builder.py already patched")

    # --- batch_generator.py: close() releases MLX memory ---
    batch_path = EXO / "worker/runner/llm_inference/batch_generator.py"
    batch = batch_path.read_text(encoding="utf-8")
    if "release_mlx_memory" not in batch:
        batch = patch_file(
            "batch_generator.py",
            'from exo.worker.engines.mlx.utils_mlx import mlx_force_oom',
            'from exo.worker.engines.mlx.utils_mlx import mlx_force_oom, release_mlx_memory',
            "batch_generator import",
            batch,
        )
        batch = patch_file(
            "batch_generator.py",
            """    def close(self) -> None:
        del self.model, self.tokenizer, self.group

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,""",
            """    def close(self) -> None:
        del self.model, self.tokenizer, self.group
        release_mlx_memory()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,""",
            "SequentialGenerator.close",
            batch,
        )
        batch = patch_file(
            "batch_generator.py",
            """    def close(self) -> None:
        self._gen.close()
        del self.model, self.tokenizer, self.group""",
            """    def close(self) -> None:
        self._gen.close()
        del self.model, self.tokenizer, self.group
        release_mlx_memory()""",
            "BatchGenerator.close",
            batch,
        )
        batch_path.write_text(batch, encoding="utf-8")
        compile_check(batch_path)
    else:
        print("batch_generator.py already patched")

    # --- bootstrap.py: SIGTERM/SIGINT MLX cleanup before exit ---
    bootstrap_path = EXO / "worker/runner/bootstrap.py"
    bootstrap = bootstrap_path.read_text(encoding="utf-8")
    if "_install_mlx_signal_handlers" not in bootstrap:
        bootstrap = patch_file(
            "bootstrap.py",
            """import os
import resource
import traceback""",
            """import os
import resource
import signal
import sys
import traceback""",
            "bootstrap imports",
            bootstrap,
        )
        bootstrap = patch_file(
            "bootstrap.py",
            """logger: "loguru.Logger" = loguru.logger


@dataclass(frozen=True)
class RunnerTerminationError:""",
            """logger: "loguru.Logger" = loguru.logger


def _install_mlx_signal_handlers() -> None:
    \"\"\"Ensure Metal buffers are released when the runner is SIGTERM'd.\"\"\"

    def _cleanup_and_exit(signum: int, _frame: object) -> None:
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                mx.synchronize()
                mx.clear_cache()
                mx.set_wired_limit(0)
                mx.clear_cache()
        except Exception as exc:
            logger.warning(f"MLX cleanup on signal {signum} failed: {exc}")
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _cleanup_and_exit)


@dataclass(frozen=True)
class RunnerTerminationError:""",
            "bootstrap signal handlers",
            bootstrap,
        )
        bootstrap = patch_file(
            "bootstrap.py",
            """    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger""",
            """    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    _install_mlx_signal_handlers()

    # Import main after setting global logger""",
            "bootstrap call signal handlers",
            bootstrap,
        )
        bootstrap_path.write_text(bootstrap, encoding="utf-8")
        compile_check(bootstrap_path)
    else:
        print("bootstrap.py already patched")

    # --- runner.py: always release MLX in main() finally ---
    runner_path = EXO / "worker/runner/runner.py"
    runner = runner_path.read_text(encoding="utf-8")
    if "release_mlx_memory" not in runner:
        runner = patch_file(
            "runner.py",
            """        finally:
            if self._prefill_server is not None:
                self._prefill_server.stop()
                self._prefill_server = None
            self.task_receiver.close()
            if self._task_reader_thread is not None:
                self._task_reader_thread.join(timeout=5)
                self._task_reader_thread = None""",
            """        finally:
            if self._prefill_server is not None:
                self._prefill_server.stop()
                self._prefill_server = None
            if not isinstance(self.current_status, RunnerShutdown):
                with contextlib.suppress(Exception):
                    self.generator.close()
            else:
                from exo.worker.engines.mlx.utils_mlx import release_mlx_memory

                with contextlib.suppress(Exception):
                    release_mlx_memory()
            self.task_receiver.close()
            if self._task_reader_thread is not None:
                self._task_reader_thread.join(timeout=5)
                self._task_reader_thread = None""",
            "runner.main finally",
            runner,
        )
        if "import contextlib" not in runner:
            runner = patch_file(
                "runner.py",
                "import queue",
                "import contextlib\nimport queue",
                "runner contextlib import",
                runner,
            )
        runner_path.write_text(runner, encoding="utf-8")
        compile_check(runner_path)
    else:
        print("runner.py already patched")

    # --- main.py: graceful shutdown + longer timeout ---
    main_path = EXO / "worker/main.py"
    main = main_path.read_text(encoding="utf-8")
    if "_graceful_shutdown_runner" not in main:
        if "self._stopped: anyio.Event = anyio.Event()" in main:
            main = patch_file(
                "main.py",
                "        self._stopped: anyio.Event = anyio.Event()",
                """        self._stopped: anyio.Event = anyio.Event()
        self._shutting_down_runners: set[RunnerId] = set()""",
                "Worker _shutting_down_runners field",
                main,
            )
        main = patch_file(
            "main.py",
            """                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)
                    for runner_id, runner in list(self.runners.items()):
                        if (
                            runner.bound_instance.instance.instance_id
                            == event.instance_id
                        ):
                            logger.info(
                                f"InstanceDeleted: shutting down runner {runner_id}"
                            )
                            self.runners.pop(runner_id, None)
                            runner.shutdown()""",
            """                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)
                    for runner_id, runner in list(self.runners.items()):
                        if (
                            runner.bound_instance.instance.instance_id
                            == event.instance_id
                        ):
                            self._schedule_runner_shutdown(
                                runner_id, event.instance_id
                            )""",
            "InstanceDeleted graceful shutdown",
            main,
        )
        main = patch_file(
            "main.py",
            """    async def _reconcile_orphaned_runners(self) -> None:
        while True:
            await anyio.sleep(1)
            for runner_id, runner in list(self.runners.items()):
                instance_id = runner.bound_instance.instance.instance_id
                if instance_id not in self.state.instances:
                    logger.warning(
                        f"Orphaned runner {runner_id} for missing instance "
                        f"{instance_id}; shutting down"
                    )
                    self.runners.pop(runner_id, None)
                    runner.shutdown()""",
            """    def _schedule_runner_shutdown(
        self, runner_id: RunnerId, instance_id: InstanceId
    ) -> None:
        if runner_id in self._shutting_down_runners:
            return
        if runner_id not in self.runners:
            return
        self._shutting_down_runners.add(runner_id)
        self._tg.start_soon(self._graceful_shutdown_runner, runner_id, instance_id)

    async def _graceful_shutdown_runner(
        self, runner_id: RunnerId, instance_id: InstanceId
    ) -> None:
        runner = self.runners.pop(runner_id, None)
        if runner is None:
            self._shutting_down_runners.discard(runner_id)
            return

        shutdown_task = Shutdown(instance_id=instance_id, runner_id=runner_id)
        logger.info(
            f"Graceful MLX shutdown for runner {runner_id} "
            f"(instance {instance_id})"
        )
        try:
            with fail_after(60):
                await runner.start_task(shutdown_task)
        except TimeoutError:
            logger.warning(
                f"Graceful shutdown timed out for runner {runner_id}; "
                "force killing subprocess"
            )
        finally:
            runner.shutdown()
            self._shutting_down_runners.discard(runner_id)

    async def _reconcile_orphaned_runners(self) -> None:
        while True:
            await anyio.sleep(1)
            for runner_id, runner in list(self.runners.items()):
                instance_id = runner.bound_instance.instance.instance_id
                if instance_id not in self.state.instances:
                    logger.warning(
                        f"Orphaned runner {runner_id} for missing instance "
                        f"{instance_id}; scheduling graceful shutdown"
                    )
                    self._schedule_runner_shutdown(runner_id, instance_id)""",
            "graceful shutdown helpers",
            main,
        )
        main = patch_file(
            "main.py",
            """                case Shutdown(runner_id=runner_id):
                    runner = self.runners.pop(runner_id)
                    try:
                        with fail_after(3):
                            await runner.start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                    finally:
                        runner.shutdown()""",
            """                case Shutdown(runner_id=runner_id):
                    runner = self.runners.pop(runner_id)
                    try:
                        with fail_after(60):
                            await runner.start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                    finally:
                        runner.shutdown()""",
            "Shutdown timeout 3->60s",
            main,
        )
        main_path.write_text(main, encoding="utf-8")
        compile_check(main_path)
    else:
        print("main.py already patched")

    print("all MLX memory-release patches applied OK")


if __name__ == "__main__":
    main()
