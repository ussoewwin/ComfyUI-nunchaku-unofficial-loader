"""Public INT8 Triton W8A8 wrappers (Plan B).

Soft-imports the fused kernel implementation. Missing Triton must never crash
graph load — callers check ``is_triton_int8_available()`` and fall back.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

LOG_PREFIX = "[HSWQ INT8 Triton]"

_AVAILABLE = False
_IMPORT_ERROR: Optional[BaseException] = None
_LOGGED_UNAVAILABLE = False

triton_quantize_rowwise = None
triton_int8_linear = None
triton_int8_linear_per_row = None

try:
    from ._int8_triton_kernels_impl import (  # noqa: F401
        triton_int8_linear as _triton_int8_linear,
        triton_int8_linear_per_row as _triton_int8_linear_per_row,
        triton_quantize_rowwise as _triton_quantize_rowwise,
    )

    triton_quantize_rowwise = _triton_quantize_rowwise
    triton_int8_linear = _triton_int8_linear
    triton_int8_linear_per_row = _triton_int8_linear_per_row
    _AVAILABLE = True
except Exception as exc:  # noqa: BLE001 — soft disable for public installs
    _IMPORT_ERROR = exc
    _AVAILABLE = False


def is_triton_int8_available() -> bool:
    """True when Triton imported and fused INT8 kernels are loadable."""
    global _LOGGED_UNAVAILABLE
    if _AVAILABLE:
        return True
    if not _LOGGED_UNAVAILABLE:
        _LOGGED_UNAVAILABLE = True
        logging.warning(
            "%s UNAVAILABLE — falling back to eager/_int_mm (%s)",
            LOG_PREFIX,
            _IMPORT_ERROR,
        )
    return False


def triton_import_error() -> Optional[BaseException]:
    return _IMPORT_ERROR


def cuda_sm_ok_for_triton_int8(min_sm: tuple[int, int] = (8, 0)) -> bool:
    """Ampere+ by default (plan success criterion). Soft-check only."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() >= min_sm
    except Exception:
        return False


def fused_int8_linear(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dispatch scalar vs per-row weight scale fused Triton INT8 Linear.

    Caller must enforce ``M > 16`` and accelerate toggle / availability.
    """
    if not _AVAILABLE or triton_int8_linear is None or triton_int8_linear_per_row is None:
        raise RuntimeError(f"{LOG_PREFIX} kernels not available: {_IMPORT_ERROR}")
    if compute_dtype is None:
        compute_dtype = x.dtype
    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([float(weight_scale)], device=x.device, dtype=torch.float32)
    if weight_scale.numel() == 1:
        return triton_int8_linear(x, weight_q, weight_scale, bias, compute_dtype=compute_dtype)
    return triton_int8_linear_per_row(x, weight_q, weight_scale, bias, compute_dtype=compute_dtype)
