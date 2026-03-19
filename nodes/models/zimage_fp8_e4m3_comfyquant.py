"""
Z Image FP8 E4M3 (ComfyUI comfy_quant 形式) 専用 Loader。

- comfy_quant / weight_scale を含むチェックポイントを comfy.sd.load_checkpoint_guess_config で読む。
- z_image + FP8 + torch.compile 互換パッチは適用する（ops/LoRA の安全化）。
"""
from __future__ import annotations

import logging
import os

import comfy.ops
import comfy.sd
import torch
from nunchaku.utils import get_gpu_memory

from ...patches.zimage_fp8_torchcompile import apply_zimage_fp8_torchcompile_patches
from ..utils import get_filename_list, get_full_path_or_raise

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HSWQZImageFP8E4M3ComfyQuantLoader:
    """
    comfy_quant 形式の Z Image を FP8 E4M3 で読む専用 Loader。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    get_filename_list("checkpoints"),
                    {"tooltip": "comfy_quant 形式の Z Image チェックポイント（checkpoints）。"},
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "CPU オフロード。'auto' は VRAM 15GB 未満で有効。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "HSWQ-ussoewwin"
    TITLE = "HSWQ Z Image FP8 E4M3 Loader (comfy_quant)"

    def load_checkpoint(self, ckpt_name: str, cpu_offload: str, **kwargs):
        apply_zimage_fp8_torchcompile_patches()

        ckpt_path = get_full_path_or_raise("checkpoints", ckpt_name)

        if cpu_offload == "auto":
            cpu_offload_enabled = get_gpu_memory() < 15
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
        else:
            cpu_offload_enabled = False

        model_options = {
            "cpu_offload_enabled": cpu_offload_enabled,
            "dtype": torch.float8_e4m3fn,
            "custom_operations": getattr(comfy.ops, "fp8_ops", comfy.ops.manual_cast),
            "fp8_optimizations": True,
        }

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=False,
            output_clip=False,
            model_options=model_options,
        )
        model = out[0]

        if hasattr(model, "model") and hasattr(model.model, "manual_cast_dtype"):
            model.model.manual_cast_dtype = torch.bfloat16

        return (model,)

