# HSWQ&Nunchaku Ultimate SD Upscale ノード用: テンソル形状・量子化まわりのランタイムパッチ
# 本ノード実行時に発生する copy_ / FP8 bias / embedder / Lumina modulate・apply_gate のエラーを
# 当拡張内のインポートとパッチ適用で対処する（他拡張・site-packages・本体は編集しない）

from __future__ import annotations
import logging
import torch

logger = logging.getLogger("nunchaku.usdu_compat_patches")


def patch_comfy_kitchen_copy_guard():
    """QuantizedTensor copy_ の shape 不一致時は copy せず dst を返す。"""
    try:
        import comfy_kitchen.tensor.base as base
    except ImportError:
        return
    if not hasattr(base, "_handle_copy_") or not hasattr(base, "_DISPATCH_TABLE"):
        return
    _orig = base._handle_copy_

    def _handle_copy_safe(qt, args, kwargs):
        dst, src = args[0], args[1]
        if not isinstance(src, base.QuantizedTensor):
            return _orig(qt, args, kwargs)
        if dst._qdata.shape != src._qdata.shape:
            return dst
        try:
            return _orig(qt, args, kwargs)
        except RuntimeError:
            return dst

    base._handle_copy_ = _handle_copy_safe
    base._DISPATCH_TABLE[torch.ops.aten.copy_.default] = _handle_copy_safe
    logger.info("[Nunchaku USDU] Applied comfy_kitchen copy_ guard (size mismatch -> return dst).")


def patch_comfy_kitchen_fp8_linear_bias_guard():
    """FP8 linear/addmm で bias と weight の out_features 不一致時は dequant パスへ。"""
    try:
        import comfy_kitchen.tensor.base as base
        import comfy_kitchen.tensor.fp8 as fp8
    except ImportError:
        return
    op_linear = torch.ops.aten.linear.default
    if op_linear not in base._LAYOUT_DISPATCH_TABLE:
        return
    layout_cls = fp8.TensorCoreFP8Layout
    if layout_cls not in base._LAYOUT_DISPATCH_TABLE[op_linear]:
        return
    _orig_linear = base._LAYOUT_DISPATCH_TABLE[op_linear][layout_cls]

    def _fp8_linear_bias_safe(qt, args, kwargs):
        input_tensor = args[0] if len(args) > 0 else None
        weight = args[1] if len(args) > 1 else None
        bias = args[2] if len(args) > 2 else None
        if (
            bias is not None
            and isinstance(input_tensor, base.QuantizedTensor)
            and isinstance(weight, base.QuantizedTensor)
        ):
            w_qdata, _ = fp8.TensorCoreFP8Layout.get_plain_tensors(weight)
            out_features = w_qdata.shape[0]
            if getattr(bias, "shape", ()) and bias.numel() != out_features:
                inp_dq, w_dq, bias_dq = base.dequantize_args((input_tensor, weight, bias))
                if bias_dq is not None and bias_dq.numel() != out_features:
                    bias_dq = None
                return torch.nn.functional.linear(inp_dq, w_dq, bias_dq)
        return _orig_linear(qt, args, kwargs)

    base._LAYOUT_DISPATCH_TABLE[op_linear][layout_cls] = _fp8_linear_bias_safe

    op_addmm = torch.ops.aten.addmm.default
    if op_addmm in base._LAYOUT_DISPATCH_TABLE and layout_cls in base._LAYOUT_DISPATCH_TABLE[op_addmm]:
        _orig_addmm = base._LAYOUT_DISPATCH_TABLE[op_addmm][layout_cls]

        def _fp8_addmm_bias_safe(qt, args, kwargs):
            bias, input_tensor, weight = args[0], args[1], args[2]
            if (
                bias is not None
                and isinstance(input_tensor, base.QuantizedTensor)
                and isinstance(weight, base.QuantizedTensor)
            ):
                w_qdata, _ = fp8.TensorCoreFP8Layout.get_plain_tensors(weight)
                out_features = w_qdata.shape[1]
                if getattr(bias, "shape", ()) and bias.numel() != out_features:
                    bias_dq, inp_dq, w_dq = base.dequantize_args(args)
                    if bias_dq is not None and bias_dq.numel() != out_features:
                        bias_dq = None
                    return torch.addmm(
                        bias_dq if bias_dq is not None else torch.zeros(out_features, device=inp_dq.device, dtype=inp_dq.dtype),
                        inp_dq, w_dq
                    )
            return _orig_addmm(qt, args, kwargs)

        base._LAYOUT_DISPATCH_TABLE[op_addmm][layout_cls] = _fp8_addmm_bias_safe

    logger.info("[Nunchaku USDU] Applied comfy_kitchen FP8 linear/addmm bias guard.")


def patch_control_embedder_linear_weight_transpose():
    """manual_cast.Linear で weight が (in, out) のときだけ転置して F.linear。"""
    try:
        import comfy.ops as ops
    except ImportError:
        return
    if not hasattr(ops, "manual_cast") or not hasattr(ops.manual_cast, "Linear"):
        return
    linear_cls = ops.manual_cast.Linear
    if not hasattr(linear_cls, "forward_comfy_cast_weights"):
        return

    def _forward_comfy_cast_weights_safe(self, input):
        weight, bias, offload_stream = ops.cast_bias_weight(self, input, offloadable=True)
        if (
            weight.dim() == 2
            and input.shape[-1] == weight.shape[0]
            and weight.shape[1] != input.shape[-1]
        ):
            weight = weight.T
        x = torch.nn.functional.linear(input, weight, bias)
        ops.uncast_bias_weight(self, weight, bias, offload_stream)
        return x

    linear_cls.forward_comfy_cast_weights = _forward_comfy_cast_weights_safe
    logger.info("[Nunchaku USDU] Applied control embedder linear weight-transpose guard.")


def patch_lumina_modulate_scale_guard():
    """Lumina modulate / apply_gate で scale・gate と x の last dim 不一致時のみ pad/trim。"""
    try:
        import comfy.ldm.lumina.model as lumina_model
    except ImportError:
        return
    if not hasattr(lumina_model, "modulate"):
        return
    _orig_modulate = lumina_model.modulate

    def _modulate_safe(x, scale, timestep_zero_index=None):
        if timestep_zero_index is not None:
            return _orig_modulate(x, scale, timestep_zero_index=timestep_zero_index)
        if x.dim() < 1 or scale.dim() < 1 or scale.size(-1) == x.size(-1):
            return _orig_modulate(x, scale, timestep_zero_index=timestep_zero_index)
        target_dim = x.size(-1)
        if scale.size(-1) < target_dim:
            scale = torch.nn.functional.pad(scale, (0, target_dim - scale.size(-1)))
        else:
            scale = scale[..., :target_dim].contiguous()
        return x * (1 + scale.unsqueeze(1))

    lumina_model.modulate = _modulate_safe

    if hasattr(lumina_model, "apply_gate"):
        _orig_apply_gate = lumina_model.apply_gate

        def _apply_gate_safe(gate, x, timestep_zero_index=None):
            if timestep_zero_index is not None:
                return _orig_apply_gate(gate, x, timestep_zero_index=timestep_zero_index)
            if x.dim() < 1 or gate.dim() < 1 or gate.size(-1) == x.size(-1):
                return _orig_apply_gate(gate, x, timestep_zero_index=timestep_zero_index)
            target_dim = x.size(-1)
            if gate.size(-1) < target_dim:
                gate = torch.nn.functional.pad(gate, (0, target_dim - gate.size(-1)))
            else:
                gate = gate[..., :target_dim].contiguous()
            return gate * x

        lumina_model.apply_gate = _apply_gate_safe

    logger.info("[Nunchaku USDU] Applied Lumina modulate & apply_gate scale-dimension guard.")


def apply_usdu_compat_patches():
    """HSWQ&Nunchaku Ultimate SD Upscale 用の互換パッチを一括適用する。"""
    patch_comfy_kitchen_copy_guard()
    patch_comfy_kitchen_fp8_linear_bias_guard()
    patch_control_embedder_linear_weight_transpose()
    patch_lumina_modulate_scale_guard()
