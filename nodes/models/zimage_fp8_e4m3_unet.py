from __future__ import annotations

import logging
import os

import torch
import comfy.utils
from comfy import sd as comfy_sd

from ...nodes.utils import get_filename_list, get_full_path_or_raise


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HSWQZImageFP8E4M3UNetLoader:
    """
    Zimage FP8 E4M3 用 UNet ローダー（HSWQ 専用）。

    - Nunchaku 系クラス・モジュールには一切依存しない
    - ComfyUI 本体の `sd.load_diffusion_model_state_dict` に state_dict をそのまま渡すだけ
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {
                        "tooltip": (
                            "Zimage FP8 E4M3 用 HSWQ UNet モデル "
                            "（ComfyUI 本体の UNet ローダーで読める .safetensors / .ckpt）"
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HSWQ-ussoewwin"
    TITLE = "HSWQ FP8 E4M3 UNet Loader"

    def load_model(
        self,
        model_name: str,
        **kwargs,
    ):
        # diffusion_models からファイルパス解決
        model_path = get_full_path_or_raise("diffusion_models", model_name)

        # state_dict + metadata を取得（中身は一切いじらない）
        sd, metadata = comfy.utils.load_torch_file(
            model_path,
            return_metadata=True,
        )

        # ComfyUI 本体の UNet ローダーに渡すオプション
        model_options = {
            # FP8 E4M3 前提
            "dtype": torch.float8_e4m3fn,
            # comfy.ops.pick_operations() に fp8_ops を選ばせるためのフラグ
            "fp8_optimizations": True,
        }

        logger.info(
            "HSWQ Zimage FP8 E4M3 UNet: loading via comfy.sd.load_diffusion_model_state_dict (%s)",
            model_path,
        )

        # 標準 UNet ローダーに state_dict をそのまま渡す。Nunchaku は一切関係なし。
        model = comfy_sd.load_diffusion_model_state_dict(
            sd,
            model_options=model_options,
            metadata=metadata,
        )

        return (model,)

