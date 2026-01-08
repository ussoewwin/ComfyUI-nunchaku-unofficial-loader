"""
Nunchaku SDXL model base.

This module provides a wrapper for ComfyUI's SDXL model base.
"""

import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

import comfy
import comfy.model_management
import comfy.utils
import comfy.conds
from comfy.model_base import ModelType, SDXL

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

logger = logging.getLogger(__name__)




class NunchakuSDXL(SDXL):
    """
    Wrapper for the Nunchaku SDXL model.

    SDXL uses UNet2DConditionModel, so we inherit from SDXL and replace
    the UNet model with NunchakuSDXLUNet2DConditionModel.
    
    This class overrides _apply_model to convert ComfyUI's calling convention
    (context parameter) to diffusers' convention (encoder_hidden_states and added_cond_kwargs).

    Parameters
    ----------
    model_config : object
        Model configuration object.
    model_type : ModelType, optional
        Type of the model (default is ModelType.EPS).
    device : torch.device or str, optional
        Device to load the model onto.
    """

    def __init__(self, model_config, model_type=ModelType.EPS, device=None, **kwargs):
        """
        Initialize the NunchakuSDXL model.

        Parameters
        ----------
        model_config : object
            Model configuration object.
        model_type : ModelType, optional
            Type of the model (default is ModelType.EPS).
        device : torch.device or str, optional
            Device to load the model onto.
        **kwargs
            Additional keyword arguments.
        """
        # SDXL uses NunchakuSDXLUNet2DConditionModel
        unet_model = kwargs.get("unet_model", NunchakuSDXLUNet2DConditionModel)

        # Remove ComfyUI-specific keys from unet_config that UNet2DConditionModel doesn't accept
        # The UNet is already built in load_diffusion_model_state_dict, so we disable creation
        unet_config = model_config.unet_config.copy()
        unet_config["disable_unet_model_creation"] = True

        # Temporarily set unet_config to avoid passing invalid keys
        original_unet_config = model_config.unet_config
        model_config.unet_config = unet_config

        try:
            # IMPORTANT:
            # We must run SDXL.__init__ so attributes like noise_augmentor are created.
            # Using super(SDXL, self).__init__ skips SDXL and calls BaseModel directly, leaving SDXL fields unset.
            super().__init__(model_config, model_type, device=device)
        finally:
            # Restore original unet_config
            model_config.unet_config = original_unet_config

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the diffusion model with ComfyUI's calling convention.
        """
        dtype = self.get_dtype()

        # 1. Process inputs using model_sampling (Standard ComfyUI logic)
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        # 2. Force cast input latent 'xc' to model dtype (bf16)
        if xc.dtype != dtype:
            xc = xc.to(dtype)
        
        # 3. Process timestep 't'
        # Nunchaku expects bf16 timestep, avoiding float32 conversion overhead
        t = self.model_sampling.timestep(t)
        if t.dtype != dtype:
            t = t.to(dtype)

        # 4. Handle context (text encoder outputs)
        context = c_crossattn
        if context is not None:
            if hasattr(context, "dtype") and context.dtype != dtype:
                context = context.to(dtype)
            # Ensure device placement
            if hasattr(context, "device") and context.device != xc.device:
                context = context.to(device=xc.device)

        # 5. Handle extra conditions (text_embeds, time_ids)
        # Revert to direct kwargs handling to avoid strict pooled_output check in self.extra_conds
        # This matches the logic that was working previously.
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype") and extra.dtype != dtype:
                 extra = extra.to(dtype)
            # Ensure device placement
            if hasattr(extra, "device") and extra.device != xc.device:
                 extra = extra.to(device=xc.device)
            extra_conds[o] = extra

        text_embeds = extra_conds.get("text_embeds", None)
        time_ids = extra_conds.get("time_ids", None)
        
        if text_embeds is not None: 
             if text_embeds.dtype != dtype:
                 text_embeds = text_embeds.to(dtype)
             if hasattr(text_embeds, "device") and text_embeds.device != xc.device:
                 text_embeds = text_embeds.to(device=xc.device)
                 
        if time_ids is not None:
             if time_ids.dtype != dtype:
                 time_ids = time_ids.to(dtype)
             if hasattr(time_ids, "device") and time_ids.device != xc.device:
                 time_ids = time_ids.to(device=xc.device)
            
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}


        # ControlNet support:
        # ComfyUI ControlNet returns a dict with keys: {"input": [...], "middle": [...], "output": [...]}
        # (see comfy/controlnet.py ControlBase.control_merge).
        #
        # Diffusers UNet2DConditionModel expects ControlNet residuals as:
        # - down_block_additional_residuals: tuple[Tensor]
        # - mid_block_additional_residual: Tensor
        #
        # We map (ComfyUI ControlNet output for SDXL CLDM):
        # - control["output"] (filtered non-None) -> down_block_additional_residuals
        # - first non-None from control["middle"] -> mid_block_additional_residual
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if isinstance(control, dict):
            # For comfy/cldm/cldm.py SDXL ControlNet, outputs are under "output" and "middle"
            # (see comfy/cldm/cldm.py forward: returns {"middle": out_middle, "output": out_output}).
            inp = control.get("output", None)
            mid = control.get("middle", None)
            if isinstance(inp, list):
                down_list = [v for v in inp if v is not None]
                if len(down_list) > 0:
                    down_block_additional_residuals = tuple(
                        comfy.model_management.cast_to_device(v, xc.device, dtype) for v in down_list
                    )
            if isinstance(mid, list):
                for v in mid:
                    if v is not None:
                        mid_block_additional_residual = comfy.model_management.cast_to_device(v, xc.device, dtype)
                        break



        # Call diffusers UNet format
        # NunchakuSDXLUNet2DConditionModel is a diffusers UNet2DConditionModel
        if isinstance(self.diffusion_model, NunchakuSDXLUNet2DConditionModel):
            # Convert ComfyUI format to diffusers format
            model_output = self.diffusion_model(
                sample=xc,
                timestep=t,
                encoder_hidden_states=context,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                return_dict=False,
            )
            # diffusers returns a tuple (sample,), ComfyUI expects tensor
            if isinstance(model_output, tuple):
                model_output = model_output[0]
        else:
            # Fallback to base class implementation for non-Nunchaku models
            model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)

        if len(model_output) > 1 and not torch.is_tensor(model_output):
            model_output, _ = comfy.utils.pack_latents(model_output)

        result = self.model_sampling.calculate_denoised(sigma, model_output.float(), x)

        return result

    def extra_conds(self, **kwargs):
        """
        Build conditioning tensors for the sampler.

        ComfyUI's default SDXL path creates a single 'y' ADM tensor that includes embedded time information.
        Diffusers SDXL expects *raw* time_ids (B, 6) and text_embeds (B, 1280) in added_cond_kwargs.

        So for the Nunchaku SDXL (diffusers UNet) backend we generate:
        - c_crossattn: standard cross-attn tensor from clip
        - text_embeds: pooled_output from SDXL clip (CLIP-G pooled)
        - time_ids: raw 6-value SDXL time ids (height, width, crop_h, crop_w, target_h, target_w)
        - y: ComfyUI SDXL ADM tensor (required by many SDXL ControlNet implementations)
        """
        out = {}

        # inpaint concat support (keep same as BaseModel)
        concat_cond = self.concat_cond(**kwargs)
        if concat_cond is not None:
            out["c_concat"] = comfy.conds.CONDNoiseShape(concat_cond)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDCrossAttn(cross_attn)

        pooled_output = kwargs.get("pooled_output", None)
        if pooled_output is None:
            raise ValueError(
                "SDXL requires pooled_output (pooled text embedding) but it was not provided. "
                "Use a proper SDXL CLIP (recommended: DualCLIPLoader type=sdxl, or a standard SDXL checkpoint CLIP) "
                "so conditioning includes pooled_output."
            )
        out["text_embeds"] = comfy.conds.CONDRegular(pooled_output)

        # Also provide ComfyUI-style SDXL ADM ("y") for ControlNet compatibility.
        # Some ControlNet SDXL models error out when y is missing.
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = comfy.conds.CONDRegular(adm)

        # Raw SDXL time ids (6 values)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        device = kwargs.get("device", pooled_output.device)
        # diffusers expects float time ids (it applies add_time_proj internally)
        base = torch.tensor(
            [height, width, crop_h, crop_w, target_height, target_width],
            device=device,
            dtype=torch.float32,
        )
        time_ids = base.unsqueeze(0).repeat(pooled_output.shape[0], 1)
        out["time_ids"] = comfy.conds.CONDRegular(time_ids)



        return out

    def load_model_weights(self, sd: dict[str, torch.Tensor], unet_prefix: str = ""):
        """
        Load model weights into the diffusion model.

        Parameters
        ----------
        sd : dict of str to torch.Tensor
            State dictionary containing model weights.
        unet_prefix : str, optional
            Prefix for UNet weights (default is "").

        Raises
        ------
        ValueError
            If a required key is missing from the state dictionary.
        """
        diffusion_model = self.diffusion_model
        if isinstance(diffusion_model, NunchakuSDXLUNet2DConditionModel):
            # NunchakuSDXLUNet2DConditionModel handles its own loading
            diffusion_model.load_state_dict(sd, strict=False)
        else:
            state_dict = diffusion_model.state_dict()
            for k in state_dict.keys():
                if k not in sd:
                    raise ValueError(f"Key {k} not found in state_dict")
            diffusion_model.load_state_dict(sd, strict=True)

