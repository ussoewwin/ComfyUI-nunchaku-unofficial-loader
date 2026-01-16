import types
import comfy

class PatchKeys:
    ################## transformer_options patches ##################
    options_key = "patches_point"
    running_net_model = "running_net_model"
    # patches supported under patches_point
    dit_enter = "patch_dit_enter"
    dit_blocks_before = "patch_dit_blocks_before"
    dit_double_blocks_replace = "patch_dit_double_blocks_replace"
    dit_double_block_with_control_replace = "patch_dit_double_block_with_control_replace"
    dit_double_blocks_after = "patch_dit_double_blocks_after"
    dit_blocks_transition_replace = "patch_dit_blocks_transition_replace"
    dit_single_blocks_before = "patch_dit_single_blocks_before"
    dit_single_blocks_replace = "patch_dit_single_blocks_replace"
    dit_blocks_after = "patch_dit_blocks_after"
    dit_blocks_after_transition_replace = "patch_dit_final_layer_before_replace"
    dit_final_layer_before = "patch_dit_final_layer_before"
    dit_exit = "patch_dit_exit"
    ################## transformer_options patches ##################


def set_model_patch(model_patcher, options_key, patch, name):
    to = model_patcher.model_options["transformer_options"]
    if options_key not in to:
        to[options_key] = {}
    to[options_key][name] = to[options_key].get(name, []) + [patch]

def set_model_patch_replace(model_patcher, options_key, patch, name):
    to = model_patcher.model_options["transformer_options"]
    if options_key not in to:
        to[options_key] = {}
    to[options_key][name] = patch

def add_model_patch_option(model, patch_key):
    if 'transformer_options' not in model.model_options:
        model.model_options['transformer_options'] = {}
    to = model.model_options['transformer_options']
    if patch_key not in to:
        to[patch_key] = {}
    return to[patch_key]


def set_hook(diffusion_model, bak_method_name, new_method, old_method_name='forward_orig'):
    if new_method is not None:
        setattr(diffusion_model, bak_method_name, getattr(diffusion_model, old_method_name));
        setattr(diffusion_model ,old_method_name, types.MethodType(new_method, diffusion_model))


def clean_hook(diffusion_model, bak_method_name, old_method_name='forward_orig'):
    if hasattr(diffusion_model, bak_method_name):
        setattr(diffusion_model, old_method_name, getattr(diffusion_model, bak_method_name))
        delattr(diffusion_model, bak_method_name)


def is_nunchaku_sdxl_model(model):
    """Check if the model is a Nunchaku SDXL model."""
    try:
        from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
        if isinstance(model, NunchakuSDXLUNet2DConditionModel):
            return True
    except ImportError:
        pass
    return False

def is_nunchaku_zimage_model(model):
    """Check if the model is a Nunchaku Z-Image model."""
    try:
        from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
        if isinstance(model, NunchakuZImageTransformer2DModel):
            return True
    except ImportError:
        pass
    return False

def is_nunchaku_model(model):
    """Check if the model is any Nunchaku model (SDXL or Z-Image)."""
    return is_nunchaku_sdxl_model(model) or is_nunchaku_zimage_model(model)

