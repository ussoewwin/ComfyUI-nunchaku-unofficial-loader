"""
HSWQ-owned NVFP4 Linear Tensor Core forward path.

Stock MixedPrecision Linear inference does:
  reshape → QuantizedTensor.from_float(act) → F.linear → often aten.addmm
  → unregistered → full dequant (slow), or wrong reshape via weight.shape[0]
  (QT storage dim).

This module owns the full inference path for HSWQ NVFP4 (+ optional ConvRot):
  1) reshape act to 2D
  2) FULL ConvRot via dense Hadamard GEMM (butterfly is slower for gs=256)
  3) cast weight/bias when off-device
  4) pooled CUDA quantize_nvfp4 (no per-call alloc)
  5) pooled cuBLAS scaled_mm_nvfp4
  6) reshape with module.out_features (never QT storage shape[0])

Never edits ComfyUI-master; installed via monkey-patch on MixedPrecision Linear.
"""
from __future__ import annotations

import logging
import os

from .nvfp4_hadamard import build_hadamard
from .nvfp4_runtime import (
    ensure_act_scale,
    clear_nvfp4_cudagraphs,
    nvfp4_quant_mm_cudagraph,
    quantize_nvfp4_act_pooled,
    rotate_last_dim_pooled,
    scaled_mm_nvfp4_pooled,
    _GRAPH_MAX_M,
)

logger = logging.getLogger(__name__)

# Counters for bench / diagnostics (reset per run if needed)
_TC_HITS = 0
_DEQUANT_FALLBACKS = 0


def reset_nvfp4_forward_stats() -> None:
    global _TC_HITS, _DEQUANT_FALLBACKS
    _TC_HITS = 0
    _DEQUANT_FALLBACKS = 0


def nvfp4_forward_stats() -> dict:
    return {"scaled_mm_hits": _TC_HITS, "dequant_fallbacks": _DEQUANT_FALLBACKS}


def _slice_nvfp4_mm_out(result, orig_m: int, orig_n: int):
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


def scaled_mm_nvfp4_linear(input_qt, weight_qt, bias):
    """Kitchen / tritant NVFP4 linear (QT path; used as fallback)."""
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    import torch.nn.functional as F
    import comfy_kitchen as ck
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(input_qt, QuantizedTensor)
        and isinstance(weight_qt, QuantizedTensor)
        and input_qt._layout_cls == "TensorCoreNVFP4Layout"
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if input_qt._qdata.dim() != 2:
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if getattr(input_qt._params, "transposed", False) or getattr(
        weight_qt._params, "transposed", False
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    a_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_qt)
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    out_dtype = input_qt._params.orig_dtype
    try:
        result = ck.scaled_mm_nvfp4(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        orig_m = input_qt._params.orig_shape[0]
        orig_n = weight_qt._params.orig_shape[0]  # (out, in)
        _TC_HITS += 1
        return _slice_nvfp4_mm_out(result, orig_m, orig_n)
    except (RuntimeError, TypeError) as e:
        logger.warning("[HSWQ NVFP4] scaled_mm_nvfp4 failed: %s — F.linear dequant", e)
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)


def _plain_weight_cached(module, weight_qt):
    """Cache get_plain_tensors on the module (weight QT identity stable after load)."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    cached = getattr(module, "_hswq_nvfp4_w_plain", None)
    if cached is not None and cached[0] is weight_qt._qdata:
        return cached[1], cached[2], cached[3], cached[4]
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    orig_n = int(weight_qt._params.orig_shape[0])
    module._hswq_nvfp4_w_plain = (
        weight_qt._qdata,
        w_qdata,
        scale_b,
        block_scale_b,
        orig_n,
    )
    return w_qdata, scale_b, block_scale_b, orig_n


def _tc_forward_pooled(module, input_2d, weight_qt, bias, act_scale, out_dtype):
    """Act float → pooled NVFP4 quant → pooled cuBLAS mm (no QT alloc).

    Prefers CUDA Graph (quantize+mm) after first capture per shape/weight; falls
    back to eager pooled kernels if capture/replay fails.
    """
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(weight_qt, QuantizedTensor)
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return None
    if getattr(weight_qt._params, "transposed", False):
        _DEQUANT_FALLBACKS += 1
        return None

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    orig_m, orig_k = int(input_2d.shape[0]), int(input_2d.shape[1])
    needs_padding = TensorCoreNVFP4Layout.get_padded_shape((orig_m, orig_k)) != (
        orig_m,
        orig_k,
    )

    scale_a = ensure_act_scale(input_2d, act_scale)
    try:
        w_qdata, scale_b, block_scale_b, orig_n = _plain_weight_cached(module, weight_qt)

        # Calib input_scale and placeholder ones are static — always cache
        # alpha. Recomputing scale_a*scale_b every Linear (~18k/sample) was
        # pure waste on FULL ConvRot (every layer has input_scale).
        cached_alpha = getattr(module, "_hswq_nvfp4_alpha", None)
        if cached_alpha is None:
            alpha = scale_a * scale_b
            if alpha.dtype != torch.float32:
                alpha = alpha.to(dtype=torch.float32)
            if alpha.dim() == 0:
                alpha = alpha.reshape(1)
            module._hswq_nvfp4_alpha = alpha
        else:
            alpha = cached_alpha

        # CUDA Graph is OFF by default: shape-shared replay copies full weight
        # every call and was slower than eager (13.05s vs ~11.8s). Opt-in:
        # HSWQ_NVFP4_CUDAGRAPH=1
        use_cg = (
            os.environ.get("HSWQ_NVFP4_CUDAGRAPH", "").strip() == "1"
            and orig_m <= _GRAPH_MAX_M
            and not getattr(module, "_hswq_nvfp4_no_cudagraph", False)
        )
        if use_cg:
            try:
                result = nvfp4_quant_mm_cudagraph(
                    input_2d,
                    w_qdata=w_qdata,
                    weight_scale=scale_b,
                    block_scale_w=block_scale_b,
                    scale_a=scale_a,
                    bias=bias,
                    out_dtype=out_dtype,
                    alpha=alpha,
                    pad_16x=needs_padding,
                    orig_n=orig_n,
                )
                _TC_HITS += 1
                return result
            except torch.cuda.OutOfMemoryError:
                clear_nvfp4_cudagraphs()
                torch.cuda.empty_cache()
                logger.warning(
                    "[HSWQ NVFP4] CUDA Graph OOM — cache cleared; eager pooled"
                )
            except (RuntimeError, TypeError, ValueError) as e:
                if "out of memory" in str(e).lower():
                    clear_nvfp4_cudagraphs()
                    torch.cuda.empty_cache()
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph OOM (%s); eager pooled", e
                    )
                else:
                    module._hswq_nvfp4_no_cudagraph = True
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph disabled for module (%s); eager pooled",
                        e,
                    )

        a_qdata, block_scale_a, _pr, _pc = quantize_nvfp4_act_pooled(
            input_2d, scale_a, pad_16x=needs_padding
        )
        result = scaled_mm_nvfp4_pooled(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
            alpha=alpha,
            orig_m=orig_m,
            orig_n=orig_n,
        )
        _TC_HITS += 1
        return result
    except (RuntimeError, TypeError, ValueError) as e:
        logger.warning("[HSWQ NVFP4] pooled TC path failed: %s", e)
        _DEQUANT_FALLBACKS += 1
        return None


def make_nvfp4_linear_forward(stock_forward):
    """
    Return a Linear.forward replacement.

    For modules flagged ``_hswq_nvfp4`` (set at load), run the HSWQ TC path.
    All other layers keep stock_forward unchanged.
    """
    import torch
    import comfy.model_management
    from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight

    def forward_nvfp4(self, input, *args, **kwargs):
        if not getattr(self, "_hswq_nvfp4", False) or getattr(self, "_full_precision_mm", False):
            return stock_forward(self, input, *args, **kwargs)

        # Training / LoRA / forced cast: fall back to stock (still ConvRot-armed there if wrap)
        if input.requires_grad or getattr(self, "comfy_force_cast_weights", False):
            return stock_forward(self, input, *args, **kwargs)
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return stock_forward(self, input, *args, **kwargs)

        run_every_op()
        input_shape = input.shape
        compute_dtype = input.dtype

        # 1) Reshape ≥3D → 2D first (same last-dim math; cheaper than rotating ND)
        reshaped_nd = input.ndim >= 3
        input_2d = input.reshape(-1, input_shape[-1]) if reshaped_nd else input
        if input_2d.ndim != 2:
            return stock_forward(self, input, *args, **kwargs)

        # 2) FULL ConvRot: dense Hadamard GEMM (gs=256 butterfly is ~15x slower)
        if getattr(self, "_hswq_nvfp4_convrot", False):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_H", None)
            if h is None or h.device != input_2d.device or h.dtype != input_2d.dtype:
                h = build_hadamard(gs, device=input_2d.device, dtype=input_2d.dtype)
                self._hswq_nvfp4_H = h
            input_2d = rotate_last_dim_pooled(input_2d, h, gs)

        # 3) Weight / bias: skip cast_bias_weight when already on-device QT
        #    (cast+sync every Linear was a major share of NVFP4 > FP16 wall time).
        offload_stream = None
        weight = self.weight
        if isinstance(weight, torch.nn.Parameter):
            weight = weight.data
        bias = self.bias.data if self.bias is not None else None
        need_cast = weight.device != input_2d.device or (
            bias is not None and bias.device != input_2d.device
        )
        if need_cast or hasattr(self, "_v"):
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input_2d,
                offloadable=True,
                compute_dtype=compute_dtype,
                want_requant=True,
            )

        scale = getattr(self, "input_scale", None)
        if scale is not None:
            if isinstance(scale, torch.nn.Parameter):
                scale = scale.data
            if scale.device != input.device:
                scale = comfy.model_management.cast_to_device(scale, input.device, None)

        layout = getattr(self, "layout_type", None)
        if layout is None:
            if offload_stream is not None:
                uncast_bias_weight(self, weight, bias, offload_stream)
            return stock_forward(self, input, *args, **kwargs)

        # 4) Pooled Tensor Core path (no QuantizedTensor.from_float alloc)
        out_2d = _tc_forward_pooled(
            self, input_2d, weight, bias, scale, compute_dtype
        )
        if out_2d is None:
            # Fallback: stock QT path
            from comfy.quant_ops import QuantizedTensor

            q_input = QuantizedTensor.from_float(input_2d, layout, scale=scale)
            out_2d = scaled_mm_nvfp4_linear(q_input, weight, bias)

        # 5) Restore rank with logical out_features (never QT storage shape[0])
        if reshaped_nd:
            out = out_2d.reshape((*input_shape[:-1], int(self.out_features)))
        else:
            out = out_2d

        if offload_stream is not None:
            uncast_bias_weight(self, weight, bias, offload_stream)
        return out

    forward_nvfp4._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    return forward_nvfp4
