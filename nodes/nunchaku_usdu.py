"""
Nunchaku-dedicated copy of Ultimate SD Upscale nodes.

This file is intentionally a 1:1 copy of `ComfyUI_UltimateSDUpscale/nodes.py`,
with a single functional change:

- `shared.batch_as_tensor` is forced to float32 (and clamped) before the internal upscaler runs.

No other behavior changes are intended.
"""

import logging
import torch
import comfy
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import to avoid errors if ComfyUI_UltimateSDUpscale is not installed
usdu = None
tensor_to_pil = None
pil_to_tensor = None
StableDiffusionProcessing = None
shared = None
UpscalerData = None
MODES = {}
SEAM_FIX_MODES = {}

def _ensure_imports():
    global usdu, tensor_to_pil, pil_to_tensor, StableDiffusionProcessing, shared, UpscalerData, MODES, SEAM_FIX_MODES
    if usdu is None:
        try:
            import sys
            import os
            
            # Get ComfyUI_UltimateSDUpscale directory
            current_file = os.path.realpath(__file__)
            nodes_dir = os.path.dirname(current_file)  # .../ComfyUI-nunchaku-unofficial-loader/nodes
            loader_dir = os.path.dirname(nodes_dir)  # .../ComfyUI-nunchaku-unofficial-loader
            custom_nodes_dir = os.path.dirname(loader_dir)  # .../custom_nodes
            usdu_custom_node = os.path.join(custom_nodes_dir, "ComfyUI_UltimateSDUpscale")
            
            # Remove other custom_node paths from sys.path to avoid conflicts (like ComfyUI_UltimateSDUpscale does)
            custom_node_paths = [path for path in sys.path if "custom_node" in path and path != usdu_custom_node]
            original_sys_path = sys.path.copy()
            for path in custom_node_paths:
                if path in sys.path:
                    sys.path.remove(path)
            
            # Add ComfyUI_UltimateSDUpscale to sys.path first
            if usdu_custom_node not in sys.path:
                sys.path.insert(0, usdu_custom_node)
            
            # Store original modules to avoid conflicts
            original_modules = sys.modules.copy()
            modules_used = ["modules", "modules.devices", "modules.images", "modules.processing", "modules.scripts", "modules.shared", "modules.upscaler", "utils"]
            original_imported_modules = {}
            for module in modules_used:
                if module in sys.modules:
                    original_imported_modules[module] = sys.modules.pop(module)
            
            try:
                # Force reloading usdu_utils from USDU path
                import importlib.util
                spec = importlib.util.spec_from_file_location("usdu_utils", os.path.join(usdu_custom_node, "usdu_utils.py"))
                usdu_utils = importlib.util.module_from_spec(spec)
                sys.modules["usdu_utils"] = usdu_utils
                spec.loader.exec_module(usdu_utils)
                # Note: original USDU might expose tensor_to_pil/pil_to_tensor in usdu_utils
                tensor_to_pil = getattr(usdu_utils, "tensor_to_pil", None)
                if tensor_to_pil is None:
                    # If not found, try to find where they are defined.
                    # USDU structure changed? Let's fallback to modules.images or checking usdu_patch
                    pass

                # If the functions are not in usdu_utils.py, we might need to look elsewhere.
                # But typically they are used in USDU. 
                # Let's assume they are there as 'utils' import usually implies.
                # Wait, 'from utils import ...' in USDU source suggests a file named utils.py existed?
                # But ls showed usdu_utils.py. Maybe it was renamed.
                
                # Let's check if we can just import usdu_utils directly since it is in sys.path
                # But "from utils import" in previous log suggests it logic.
                
                # Let's stick to loading usdu_utils.py for now.
                tensor_to_pil = usdu_utils.tensor_to_pil
                pil_to_tensor = usdu_utils.pil_to_tensor

                from usdu_patch import usdu  # type: ignore
                # from utils import tensor_to_pil, pil_to_tensor  # type: ignore (REPLACED due to conflict)
                from modules.processing import StableDiffusionProcessing  # type: ignore
                import modules.shared as shared  # type: ignore
                from modules.upscaler import UpscalerData  # type: ignore
            finally:
                # Restore sys.path
                sys.path = original_sys_path
                # Restore original modules
                sys.modules.update(original_imported_modules)
            
            MODES = {
                "Linear": usdu.USDUMode.LINEAR,
                "Chess": usdu.USDUMode.CHESS,
                "None": usdu.USDUMode.NONE,
            }
            
            SEAM_FIX_MODES = {
                "None": usdu.USDUSFMode.NONE,
                "Band Pass": usdu.USDUSFMode.BAND_PASS,
                "Half Tile": usdu.USDUSFMode.HALF_TILE,
                "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
            }
        except ImportError as e:
            logger.error(f"Failed to import ComfyUI_UltimateSDUpscale: {e}")
            raise

MAX_RESOLUTION = 8192


def _to_fp32_image(image: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor to FP32 and ensure proper color range for Nunchaku SDXL.
    
    Nunchaku SDXL VAE decode output may be in a different color space or range.
    This function normalizes the input to ensure upscaler receives correct values.
    """
    t = image
    if torch.is_tensor(t):
        t = t.to(dtype=torch.float32)
        
        # Always normalize to maximize dynamic range
        # This fixes the issue where Nunchaku SDXL VAE outputs compressed range [0.15, 0.85]
        min_val = t.min().item()
        max_val = t.max().item()
        
        # Normalize to [0,1] range if there's a valid range
        if max_val > min_val:
            t = (t - min_val) / (max_val - min_val)
        else:
            t = torch.zeros_like(t)
        
        # Ensure values are in [0,1] range
        t = torch.clamp(t, 0.0, 1.0)
        t = t.contiguous()
    return t


def USDU_base_inputs():
    _ensure_imports()
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Sampling Params
        ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
        ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
        ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "The seed to use for image-to-image."})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
    ]

    optional = []
    return required, optional


def prepare_inputs(required: list, optional: list | None = None):
    inputs: dict = {}
    if required:
        inputs["required"] = {}
        for name, t in required:
            inputs["required"][name] = t
    if optional:
        inputs["optional"] = {}
        for name, t in optional:
            inputs["optional"][name] = t
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class NunchakuUltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        try:
            _ensure_imports()
            required, optional = USDU_base_inputs()
            return prepare_inputs(required, optional)
        except Exception as e:
            logger.error(f"Failed to initialize NunchakuUltimateSDUpscale INPUT_TYPES: {e}", exc_info=True)
            # Provide fallback with default modes to allow node registration even if imports fail
            # This ensures the node appears in the UI even if ComfyUI_UltimateSDUpscale is not properly installed
            fallback_modes = ["Linear", "Chess", "None"] if not MODES else list(MODES.keys())
            fallback_seam_modes = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"] if not SEAM_FIX_MODES else list(SEAM_FIX_MODES.keys())
            
            required = [
                ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
                ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
                ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
                ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
                ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
                ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
                ("seed", ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "The seed to use for image-to-image."})),
                ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
                ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
                ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
                ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
                ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
                ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
                ("mode_type", (fallback_modes, {"tooltip": "The tiling order to use for the redraw step."})),
                ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
                ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
                ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
                ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
                ("seam_fix_mode", (fallback_seam_modes, {"tooltip": "The seam fix mode to use."})),
                ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
                ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
                ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
                ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
                ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
                ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
            ]
            optional = []
            return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image and runs image-to-image on tiles from the input image."
    TITLE = "Nunchaku Ultimate SD Upscale"

    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_model,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        _ensure_imports()
        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.upscale_by = upscale_by
        self.seam_fix_mask_blur = seam_fix_mask_blur

        #
        # Set up A1111 patches
        #

        # Upscaler
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Normalize color range for Nunchaku SDXL VAE output before processing
        image = _to_fp32_image(image)

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        # Set batch_as_tensor
        shared.batch_as_tensor = image

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0],
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            upscale_by,
            force_uniform_tiles,
            tiled_decode,
            tile_width,
            tile_height,
            MODES[self.mode_type],
            SEAM_FIX_MODES[self.seam_fix_mode],
            custom_sampler,
            custom_sigmas,
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            # Running the script
            script = usdu.Script()
            _ = script.run(
                p=sdprocessing,
                _=None,
                tile_width=self.tile_width,
                tile_height=self.tile_height,
                mask_blur=self.mask_blur,
                padding=self.tile_padding,
                seams_fix_width=self.seam_fix_width,
                seams_fix_denoise=self.seam_fix_denoise,
                seams_fix_padding=self.seam_fix_padding,
                upscaler_index=0,
                save_upscaled_image=False,
                redraw_mode=MODES[self.mode_type],
                save_seams_fix_image=False,
                seams_fix_mask_blur=self.seam_fix_mask_blur,
                seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode],
                target_size_type=2,
                custom_width=None,
                custom_height=None,
                custom_scale=self.upscale_by,
            )

            # Return the resulting images
            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)
        finally:
            # Restore the original logging level
            logger.setLevel(old_level)

