"""Narrow compatibility boundary for EXO's optional DeepSeek-V4 support."""

import mlx.nn as nn


class _UnavailableDeepseekV4MoE(nn.Module):
    """Sentinel used when this optional experimental model is not installed."""


class _UnavailableV4Attention(nn.Module):
    """Sentinel used when this optional experimental model is not installed."""


class _UnavailableDeepseekV4Model(nn.Module):
    """Sentinel used when this optional experimental model is not installed."""


class _UnavailableDeepseekV4Cache:
    """Sentinel for cache dispatch when optional DeepSeek V4 is unavailable."""


class _UnavailableCompressorBranch:
    """Sentinel for cache helpers when optional DeepSeek V4 is unavailable."""


try:
    from mlx_lm.models.deepseek_v4 import (
        DeepseekV4Cache,
        DeepseekV4MoE,
        V4Attention,
    )
    from mlx_lm.models.deepseek_v4 import (
        Model as DeepseekV4Model,
    )
    from mlx_lm.models.deepseek_v4 import (
        _CompressorBranch as CompressorBranch,  # type: ignore
    )
except ModuleNotFoundError as error:
    if error.name != "mlx_lm.models.deepseek_v4":
        raise
    DeepseekV4MoE = _UnavailableDeepseekV4MoE
    V4Attention = _UnavailableV4Attention
    DeepseekV4Model = _UnavailableDeepseekV4Model
    DeepseekV4Cache = _UnavailableDeepseekV4Cache
    CompressorBranch = _UnavailableCompressorBranch
    deepseek_v4_available = False
else:
    deepseek_v4_available = True

DEEPSEEK_V4_AVAILABLE = deepseek_v4_available

__all__ = [
    "CompressorBranch",
    "DEEPSEEK_V4_AVAILABLE",
    "DeepseekV4Cache",
    "DeepseekV4Model",
    "DeepseekV4MoE",
    "V4Attention",
]
