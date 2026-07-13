"""
HSWQ Z-Image-Turbo FP8(E4M3) dedicated loader.

Does not modify upstream Nunchaku `nodes/models/zimage.py`.
Use this separate node only when you want FP8.
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
    HSWQ Z-Image-Turbo FP8(E4M3) dedicated loader.

    - Does not touch Nunchaku `nodes/models/zimage.py`
    - Class/node names are HSWQ, not Nunchaku
    - Reuses upstream `load_diffusion_model_state_dict` for loading
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

        # Use Nunchaku-side load_diffusion_model_state_dict as-is.
        # FP8 optimization follows upstream (metadata / quantization_config).
        model = load_diffusion_model_state_dict(
            sd, metadata=metadata, model_options={"cpu_offload_enabled": cpu_offload_enabled}
        )

        return (model,)

