"""
This module provides Z-Image-Turbo node classes (from SDXL equivalents):
- ZImageTurboDiTLoader
- ZImageTurboIntegratedLoader
"""

import logging
import os

import comfy.sd
import comfy.utils
import torch
from comfy import model_detection, model_management
from nunchaku.utils import get_gpu_memory

from ..utils import get_filename_list, get_full_path_or_raise
from .zimage import load_diffusion_model_state_dict as load_zimage_diffusion_model_state_dict

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ZImageTurboDiTLoader:
    """
    Loader for Z-Image-Turbo models (DiT Loader).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {"tooltip": "Z-Image-Turbo model (transformer-only safetensors)."},
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
            "optional": {
                "num_blocks_on_gpu": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 60,
                        "tooltip": (
                            "When CPU offload is enabled, how many transformer blocks remain on GPU memory."
                        ),
                    },
                ),
                "use_pin_memory": (
                    ["enable", "disable"],
                    {
                        "default": "disable",
                        "tooltip": "Use pinned memory for transformer blocks when CPU offload is enabled.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/zit"
    TITLE = "ZIT DiT Loader"

    def load_model(
        self, model_name: str, cpu_offload: str, num_blocks_on_gpu: int = 1, use_pin_memory: str = "disable", **kwargs
    ):
        model_path = get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)

        if cpu_offload == "auto":
            cpu_offload_enabled = get_gpu_memory() < 15
            logger.info("VRAM < 15GiB, enabling CPU offload" if cpu_offload_enabled else "VRAM > 15GiB, disabling CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
        else:
            cpu_offload_enabled = False

        model = load_zimage_diffusion_model_state_dict(
            sd, metadata=metadata, model_options={"cpu_offload_enabled": cpu_offload_enabled}
        )
        return (model,)


class ZImageTurboIntegratedLoader:
    """
    Loader for unified Z-Image-Turbo checkpoints (transformer + CLIP in one file).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (get_filename_list("checkpoints"), {"tooltip": "Unified Z-Image-Turbo checkpoint file (transformer + CLIP)."}),
            },
            "optional": {
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Whether to enable CPU offload for the transformer model.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_integrated_model"
    CATEGORY = "loaders/zit"
    TITLE = "ZIT Integrated Loader"

    def load_integrated_model(self, ckpt_name: str, cpu_offload: str = "auto", **kwargs):
        from comfy import supported_models as comfy_supported_models

        from ..utils import folder_paths

        ckpt_path = get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        if metadata is None:
            metadata = {}

        if cpu_offload == "auto":
            cpu_offload_enabled = get_gpu_memory() < 15
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
        else:
            cpu_offload_enabled = False

        model_options = {"cpu_offload_enabled": cpu_offload_enabled}
        model = load_zimage_diffusion_model_state_dict(sd, metadata=metadata, model_options=model_options)

        # Detect model config for CLIP (Z-Image-Turbo)
        clip_model_config = model_detection.model_config_from_unet(sd, "", metadata)
        if clip_model_config is None:
            logger.warning("Could not detect model config from checkpoint keys. Assuming Z-Image-Turbo.")
            clip_model_config = comfy_supported_models.ZImage(sd)

        clip = None
        clip_target = clip_model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = clip_model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = comfy.utils.calculate_parameters(clip_sd)
                clip = comfy.sd.CLIP(
                    clip_target,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    parameters=parameters,
                    state_dict=clip_sd,
                )
            else:
                logger.warning("No CLIP weights found in Z-Image-Turbo checkpoint.")
        if clip is None:
            logger.warning("Failed to load CLIP from integrated Z-Image-Turbo checkpoint.")

        return (model, clip)
