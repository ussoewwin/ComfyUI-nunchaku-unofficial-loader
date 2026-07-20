"""
Patch inside ComfyUI-HSWQ-Loader-and-Tools to prevent mat1/mat2 shape
mismatches when running Z-Image + FP8 E4M3 + torch.compile.

- LoRA: skip applying a layer when reshape or LoRA output numel does not match weight.
- comfy.ops Linear: while torch.compile is running, or when
  input.shape[-1] != weight.shape[1], skip the 3D/QuantizedTensor path and,
  on mismatch, slice the input to weight.shape[1] to avoid a crash.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False


def _apply_ops_patch() -> bool:
    """Wrap comfy.ops.mixed_precision_ops; skip 3D/FP8 input path under compile / shape mismatch."""
    try:
        import torch
        import comfy.ops as ops_module
    except ImportError:
        return False

    original_mixed_precision_ops = getattr(ops_module, "mixed_precision_ops", None)
    if original_mixed_precision_ops is None or not callable(original_mixed_precision_ops):
        return False

    def patched_mixed_precision_ops(quant_config=None, compute_dtype=None, full_precision_mm=False, disabled=None):
        if quant_config is None:
            quant_config = {}
        if disabled is None:
            disabled = []
        result = original_mixed_precision_ops(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        Linear = getattr(result, "Linear", None)
        if Linear is None:
            return result
        _orig_forward = getattr(Linear, "forward", None)
        if _orig_forward is None or not callable(_orig_forward):
            return result

        run_every_op = getattr(ops_module, "run_every_op", None)
        if run_every_op is None:
            return result

        def _forward_compile_safe(self, input, *args, **kwargs):
            run_every_op()
            in_features = getattr(self, "weight", None)
            in_features = in_features.shape[1] if in_features is not None else None
            compiling = torch.compiler.is_compiling()
            shape_mismatch = in_features is not None and input.shape[-1] != in_features
            if compiling or shape_mismatch:
                if shape_mismatch:
                    logger.warning(
                        "ComfyUI-nunchaku: Linear input.shape[-1]=%s != weight.shape[1]=%s, slicing input [z_image/FP8/torch.compile compat]",
                        input.shape[-1],
                        in_features,
                    )
                    input = input[..., :in_features].contiguous()
                if input.ndim == 3:
                    input_shape = input.shape
                    input = input.reshape(-1, input_shape[2])
                    compute_dtype = getattr(input, "dtype", None)
                    output = self.forward_comfy_cast_weights(input, compute_dtype, want_requant=False)
                    output = output.reshape((input_shape[0], input_shape[1], self.weight.shape[0]))
                else:
                    compute_dtype = getattr(input, "dtype", None)
                    output = self.forward_comfy_cast_weights(input, compute_dtype, want_requant=False)
                return output
            return _orig_forward(self, input, *args, **kwargs)

        Linear.forward = _forward_compile_safe
        return result

    ops_module.mixed_precision_ops = patched_mixed_precision_ops
    return True


def _apply_lora_patch() -> bool:
    """Wrap LoraDiff.calculate_weight; for z_image/FP8 skip on reshape or numel mismatch."""
    try:
        import torch
        import comfy.weight_adapter.lora as lora_module
    except ImportError:
        return False

    LoraDiff = getattr(lora_module, "LoraDiff", None)
    if LoraDiff is None:
        return False
    _original_calculate_weight = getattr(LoraDiff, "calculate_weight", None)
    if _original_calculate_weight is None or not callable(_original_calculate_weight):
        return False

    def _patched_calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=None,
        original_weight=None,
    ):
        if intermediate_dtype is None:
            intermediate_dtype = torch.float32
        v = self.weights
        reshape = v[5]
        try:
            from .comfy_quant_int8 import record_lora_shape_skip
        except ImportError:
            record_lora_shape_skip = None

        if reshape is not None and tuple(reshape) != weight.shape:
            reason = f"reshape {list(reshape)} != weight.shape {list(weight.shape)}"
            logger.warning(
                "LoRA %s: skipping %s (%s) [z_image/FP8/torch.compile compat]",
                self.name, key, reason,
            )
            if record_lora_shape_skip is not None:
                record_lora_shape_skip(getattr(self, "name", "?"), key, reason)
            return weight
        try:
            lora_diff_flat = torch.mm(v[0].flatten(start_dim=1), v[1].flatten(start_dim=1))
            if lora_diff_flat.numel() != weight.numel():
                reason = (
                    f"lora output size {lora_diff_flat.numel()} != weight size {weight.numel()}"
                )
                logger.warning(
                    "LoRA %s: skipping %s (%s) [z_image/FP8/torch.compile compat]",
                    self.name, key, reason,
                )
                if record_lora_shape_skip is not None:
                    record_lora_shape_skip(getattr(self, "name", "?"), key, reason)
                return weight
        except Exception as e:
            reason = f"error during lora_diff_flat check: {e}"
            logger.warning(
                "LoRA %s: skipping %s (%s) [z_image/FP8/torch.compile compat]",
                self.name, key, reason,
            )
            if record_lora_shape_skip is not None:
                record_lora_shape_skip(getattr(self, "name", "?"), key, reason)
            return weight
        return _original_calculate_weight(
            self,
            weight=weight,
            key=key,
            strength=strength,
            strength_model=strength_model,
            offset=offset,
            function=function,
            intermediate_dtype=intermediate_dtype,
            original_weight=original_weight,
        )

    LoraDiff.calculate_weight = _patched_calculate_weight
    return True


def _apply_rmsnorm_patch() -> bool:
    """
    Wrap comfy.ops.RMSNorm.forward_comfy_cast_weights so a mismatch between
    normalized_shape[0] and input.shape[-1] does not crash.

    - Matching shapes: call the original implementation.
    - Mismatch: slice/pad weight to the input last-dim, then call torch.rms_norm.
    """
    try:
        import torch
        import comfy.ops as ops_module
    except ImportError:
        return False

    RMSNorm = getattr(ops_module, "RMSNorm", None)
    if RMSNorm is None:
        return False

    _orig_forward_comfy = getattr(RMSNorm, "forward_comfy_cast_weights", None)
    if _orig_forward_comfy is None or getattr(_orig_forward_comfy, "_hswq_rmsnorm_patched", False):
        return False

    def _forward_comfy_cast_weights_safe(self, input):
        norm_shape = getattr(self, "normalized_shape", None)
        target_dim = None
        if isinstance(norm_shape, (tuple, list)) and len(norm_shape) > 0:
            try:
                target_dim = int(norm_shape[0])
            except Exception:
                target_dim = None
        elif isinstance(norm_shape, int):
            target_dim = norm_shape

        last_dim = input.shape[-1]

        if target_dim is not None and last_dim != target_dim:
            logger.warning(
                "ComfyUI-nunchaku: RMSNorm input.shape[-1]=%s != normalized_shape[0]=%s, adjusting to input dim [z_image/FP8/torch.compile compat]",
                last_dim,
                target_dim,
            )
            weight = getattr(self, "weight", None)
            eff_weight = None
            if weight is not None:
                if weight.shape[-1] >= last_dim:
                    eff_weight = weight[..., :last_dim]
                else:
                    pad = last_dim - weight.shape[-1]
                    eff_weight = torch.nn.functional.pad(weight, (0, pad), value=1.0)
            return torch.rms_norm(input, last_dim, eff_weight, self.eps)

        return _orig_forward_comfy(self, input)

    setattr(_forward_comfy_cast_weights_safe, "_hswq_rmsnorm_patched", True)
    RMSNorm.forward_comfy_cast_weights = _forward_comfy_cast_weights_safe
    return True


def apply_zimage_fp8_torchcompile_patches() -> bool:
    """
    Apply Z-Image FP8 E4M3 + torch.compile compatibility patches.
    Skip if already applied. Returns whether patches are in effect.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return True
    ok_ops = _apply_ops_patch()
    ok_lora = _apply_lora_patch()
    ok_rmsnorm = _apply_rmsnorm_patch()
    if ok_ops:
        logger.info(
            "ComfyUI-nunchaku: applied z_image FP8/torch.compile ops patch (skip 3D/QuantizedTensor path when compiling or shape mismatch)"
        )
    if ok_lora:
        logger.info(
            "ComfyUI-nunchaku: applied z_image FP8/torch.compile LoRA compat patch (reshape/numel skip)"
        )
    if ok_rmsnorm:
        logger.info(
            "ComfyUI-nunchaku: applied z_image FP8/torch.compile RMSNorm compat patch (normalized_shape/input mismatch safe)"
        )
    _PATCHES_APPLIED = ok_ops or ok_lora or ok_rmsnorm
    return _PATCHES_APPLIED
