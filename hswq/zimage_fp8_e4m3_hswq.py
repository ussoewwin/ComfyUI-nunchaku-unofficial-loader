"""
HSWQ Z-Image-Turbo FP8(E4M3) 専用ローダー。

Nunchaku 本体用の `nodes/models/zimage.py` には一切手を入れず、
FP8 で動かしたいときだけ、この別ファイルのノードを使う。
"""

from __future__ import annotations

import logging

import comfy.utils
from nunchaku.utils import get_gpu_memory

from ..nodes.models.zimage import load_diffusion_model_state_dict
from ..nodes.utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)


class HSWQZImageFP8E4M3DiTLoader:
    """
    HSWQ Z-Image-Turbo FP8(E4M3) 専用ローダー。

    - Nunchaku 用 `nodes/models/zimage.py` には触れない
    - クラス名／ノード名は Nunchaku ではなく HSWQ
    - 実際のロード処理は元の `load_diffusion_model_state_dict` を使い回す
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {"tooltip": "HSWQ Z-Image-Turbo FP8(E4M3) model."},
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Whether to enable CPU offload for the transformer model. "
                        "'auto' will enable it if the GPU memory is less than 15G.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HSWQ"
    TITLE = "HSWQ Z-Image-Turbo FP8(E4M3) DiT Loader"

    def load_model(
        self,
        model_name: str,
        cpu_offload: str,
        **kwargs,
    ):
        model_path = get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)

        if cpu_offload == "auto":
            if get_gpu_memory() < 15:  # 15GB threshold
                cpu_offload_enabled = True
                logger.info("[HSWQ] VRAM < 15GiB, enabling CPU offload")
            else:
                cpu_offload_enabled = False
                logger.info("[HSWQ] VRAM > 15GiB, disabling CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
            logger.info("[HSWQ] Enabling CPU offload")
        else:
            assert cpu_offload == "disable", "Invalid CPU offload option"
            cpu_offload_enabled = False
            logger.info("[HSWQ] Disabling CPU offload")

        # Nunchaku 側の load_diffusion_model_state_dict をそのまま使う。
        # FP8 最適化の有無などは、元の実装（metadata / quantization_config）に従う。
        model = load_diffusion_model_state_dict(
            sd, metadata=metadata, model_options={"cpu_offload_enabled": cpu_offload_enabled}
        )

        return (model,)

