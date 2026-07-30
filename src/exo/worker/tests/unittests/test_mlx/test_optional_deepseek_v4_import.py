from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _run_isolated_import(script: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_auto_parallel_imports_when_optional_deepseek_v4_is_absent() -> None:
    result = _run_isolated_import(
        """
        import importlib.abc
        import sys

        class DeepseekV4Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "mlx_lm.models.deepseek_v4":
                    raise ModuleNotFoundError(
                        "optional DeepSeek V4 blocked for test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, DeepseekV4Blocker())

        import mlx.nn as nn
        from exo.worker.engines.mlx import auto_parallel, cache, kimi_k3_pipeline
        from exo.worker.engines.mlx.generator import generate
        from exo.worker.runner.llm_inference import model_output_parsers

        assert not auto_parallel.DEEPSEEK_V4_AVAILABLE
        assert not cache.DEEPSEEK_V4_CACHE_AVAILABLE
        assert issubclass(auto_parallel.DeepseekV4Model, nn.Module)
        assert kimi_k3_pipeline.KimiK3PipelineFirstLayer is not None
        assert generate.mlx_generate is not None
        assert issubclass(model_output_parsers.DeepseekV4Model, nn.Module)
        """
    )
    assert result.returncode == 0, result.stderr


def test_auto_parallel_uses_deepseek_v4_types_when_module_is_present() -> None:
    result = _run_isolated_import(
        """
        import sys
        import types

        import mlx.nn as nn

        module = types.ModuleType("mlx_lm.models.deepseek_v4")

        class DeepseekV4MoE(nn.Module):
            pass

        class V4Attention(nn.Module):
            pass

        class Model(nn.Module):
            pass

        class DeepseekV4Cache:
            pass

        class CompressorBranch:
            pass

        module.DeepseekV4MoE = DeepseekV4MoE
        module.V4Attention = V4Attention
        module.Model = Model
        module.DeepseekV4Cache = DeepseekV4Cache
        module._CompressorBranch = CompressorBranch
        sys.modules[module.__name__] = module

        from exo.worker.engines.mlx import auto_parallel, cache
        from exo.worker.runner.llm_inference import model_output_parsers

        assert auto_parallel.DEEPSEEK_V4_AVAILABLE
        assert cache.DEEPSEEK_V4_CACHE_AVAILABLE
        assert auto_parallel.DeepseekV4MoE is DeepseekV4MoE
        assert auto_parallel.V4Attention is V4Attention
        assert auto_parallel.DeepseekV4Model is Model
        assert cache.DeepseekV4Cache is DeepseekV4Cache
        assert cache.CompressorBranch is CompressorBranch
        assert model_output_parsers.DeepseekV4Model is Model
        """
    )
    assert result.returncode == 0, result.stderr
