import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import folder_paths
import safetensors.torch
import comfy.sd
import comfy.utils
import comfy.ops

# --- Efficient FP8 Dequantization ---

def dequantize_fp8_weight(weight_fp8: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Dequantize FP8 weight to target dtype efficiently.
    Uses bfloat16 as intermediate to minimize VRAM (2 bytes vs 4 bytes for float32).
    FP8 (1 byte) -> bfloat16 (2 bytes) -> target_dtype
    """
    # Use bfloat16 as intermediate for VRAM efficiency (half the memory of float32)
    w = weight_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)
    if target_dtype != torch.bfloat16:
        w = w.to(target_dtype)
    return w

# --- Custom Layers for Runtime Scaling ---
# These classes inherit from nn.Linear/nn.Conv2d for LoRA compatibility
# They expose a 'weight' property that returns dequantized weights for LoRA patching

class HSWQLinear(nn.Linear):
    """
    FP8 Linear layer with runtime dequantization.
    - Inherits from nn.Linear for LoRA compatibility
    - Stores weights in FP8 (1 byte per element) for 50% VRAM savings vs FP16
    - Dequantizes to compute dtype only during forward pass
    - Exposes 'weight' property for LoRA compatibility
    """
    # ComfyUI LowVram compatibility attributes
    comfy_cast_weights = False
    weight_function = []
    bias_function = []
    
    def __init__(self, original_linear: nn.Linear, weight_tensor: torch.Tensor, scale: torch.Tensor):
        # Initialize parent with same dimensions but no actual weight creation
        # We use device='meta' to create a shell without allocating memory
        nn.Module.__init__(self)  # Skip nn.Linear.__init__ to avoid weight allocation
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Store FP8 weight (1 byte per element) - this is the VRAM saving
        self.register_buffer("weight_fp8", weight_tensor.detach().clone(), persistent=False)
        self.register_buffer("hswq_scale", scale.clone().detach(), persistent=False)
        
        # Store original dtype for weight property
        self._compute_dtype = torch.float16
        
        # ComfyUI LowVram compatibility - instance-level lists
        self.weight_function = []
        self.bias_function = []
        
        if original_linear.bias is not None:
            # Keep bias as a parameter for compatibility
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    @property
    def weight(self):
        """Return dequantized weight for LoRA compatibility."""
        return dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, self._compute_dtype)
    
    @weight.setter
    def weight(self, value):
        """Allow weight setting for LoRA compatibility."""
        if value is not None:
            # Store patched weight (handle both Tensor and Parameter)
            val = value.data if isinstance(value, nn.Parameter) else value
            if not hasattr(self, '_patched_weight') or self._patched_weight is None:
                self.register_buffer("_patched_weight", None, persistent=False)
            self._patched_weight = val.detach()
    
    def __setattr__(self, name, value):
        """Override to handle weight assignment without register_parameter error."""
        if name == 'weight' and isinstance(value, (torch.Tensor, nn.Parameter)):
            # Call the property setter directly to avoid recursion
            HSWQLinear.weight.fset(self, value)
            return
        super().__setattr__(name, value)
    
    def set_weight(self, weight, inplace_update=False, seed=None):
        """ComfyUI ModelPatcher calls this to apply LoRA patches."""
        if not hasattr(self, '_patched_weight') or self._patched_weight is None:
            self.register_buffer("_patched_weight", None, persistent=False)
        val = weight.data if isinstance(weight, nn.Parameter) else weight
        self._patched_weight = val.detach()
        if not hasattr(HSWQLinear, '_lora_log_count'):
            HSWQLinear._lora_log_count = 0
        HSWQLinear._lora_log_count += 1
        if HSWQLinear._lora_log_count <= 3:
            print(f"[HSWQ-LoRA] Linear layer patched: {self.out_features}x{self.in_features}")
        elif HSWQLinear._lora_log_count == 4:
            print(f"[HSWQ-LoRA] ... (more layers being patched)")

    def forward(self, input):
        target_dtype = input.dtype
        self._compute_dtype = target_dtype  # Update for weight property
        
        # If LoRA patched weight exists, use it instead of FP8 dequantization
        if hasattr(self, '_patched_weight') and self._patched_weight is not None:
            w = self._patched_weight.to(target_dtype)
        else:
            # Dequantize FP8 -> bfloat16 -> target_dtype (VRAM efficient path)
            w = dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, target_dtype)
        
        # Apply LowVram weight_function patches (ComfyUI dynamic LoRA)
        for f in self.weight_function:
            w = f(w)
        
        bias = self.bias.to(target_dtype) if self.bias is not None else None
        # Apply LowVram bias_function patches
        if bias is not None:
            for f in self.bias_function:
                bias = f(bias)
        
        return F.linear(input, w, bias)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Override to include dequantized weight in state_dict for LoRA compatibility."""
        # Add dequantized weight (this is what LoRA key mapping needs)
        destination[prefix + 'weight'] = self.weight if keep_vars else self.weight.detach()
        # Add bias if present
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Override to expose weight as a parameter for ModelPatcher compatibility."""
        # Yield the dequantized weight as if it were a parameter
        yield prefix + ('.' if prefix else '') + 'weight', self.weight
        if self.bias is not None:
            yield prefix + ('.' if prefix else '') + 'bias', self.bias

class HSWQConv2d(nn.Conv2d):
    """
    FP8 Conv2d layer with runtime dequantization.
    - Inherits from nn.Conv2d for LoRA compatibility
    - Stores weights in FP8 (1 byte per element) for 50% VRAM savings vs FP16
    - Dequantizes to compute dtype only during forward pass
    - Exposes 'weight' property for LoRA compatibility
    """
    # ComfyUI LowVram compatibility attributes
    comfy_cast_weights = False
    weight_function = []
    bias_function = []
    
    def __init__(self, original_conv: nn.Conv2d, weight_tensor: torch.Tensor, scale: torch.Tensor):
        # Initialize parent without weight allocation
        nn.Module.__init__(self)  # Skip nn.Conv2d.__init__ to avoid weight allocation
        
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.padding_mode = original_conv.padding_mode
        # Required for nn.Conv2d __repr__ compatibility
        self.transposed = getattr(original_conv, 'transposed', False)
        self.output_padding = getattr(original_conv, 'output_padding', (0, 0))
        
        # Store FP8 weight (1 byte per element) - this is the VRAM saving
        self.register_buffer("weight_fp8", weight_tensor.detach().clone(), persistent=False)
        self.register_buffer("hswq_scale", scale.clone().detach(), persistent=False)
        
        # Store original dtype for weight property
        self._compute_dtype = torch.float16
        
        # ComfyUI LowVram compatibility - instance-level lists
        self.weight_function = []
        self.bias_function = []
        
        if original_conv.bias is not None:
            # Keep bias as a parameter for compatibility
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    @property
    def weight(self):
        """Return dequantized weight for LoRA compatibility."""
        return dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, self._compute_dtype)
    
    @weight.setter
    def weight(self, value):
        """Allow weight setting for LoRA compatibility."""
        if value is not None:
            # Store patched weight (handle both Tensor and Parameter)
            val = value.data if isinstance(value, nn.Parameter) else value
            if not hasattr(self, '_patched_weight') or self._patched_weight is None:
                self.register_buffer("_patched_weight", None, persistent=False)
            self._patched_weight = val.detach()
    
    def __setattr__(self, name, value):
        """Override to handle weight assignment without register_parameter error."""
        if name == 'weight' and isinstance(value, (torch.Tensor, nn.Parameter)):
            # Call the property setter directly to avoid recursion
            HSWQConv2d.weight.fset(self, value)
            return
        super().__setattr__(name, value)
    
    def set_weight(self, weight, inplace_update=False, seed=None):
        """ComfyUI ModelPatcher calls this to apply LoRA patches."""
        if not hasattr(self, '_patched_weight') or self._patched_weight is None:
            self.register_buffer("_patched_weight", None, persistent=False)
        val = weight.data if isinstance(weight, nn.Parameter) else weight
        self._patched_weight = val.detach()
        if not hasattr(HSWQConv2d, '_lora_log_count'):
            HSWQConv2d._lora_log_count = 0
        HSWQConv2d._lora_log_count += 1
        if HSWQConv2d._lora_log_count <= 2:
            print(f"[HSWQ-LoRA] Conv2d layer patched: {self.out_channels}x{self.in_channels}")

    def forward(self, input):
        target_dtype = input.dtype
        self._compute_dtype = target_dtype  # Update for weight property
        
        # If LoRA patched weight exists, use it instead of FP8 dequantization
        if hasattr(self, '_patched_weight') and self._patched_weight is not None:
            w = self._patched_weight.to(target_dtype)
        else:
            # Dequantize FP8 -> bfloat16 -> target_dtype (VRAM efficient path)
            w = dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, target_dtype)
        
        # Apply LowVram weight_function patches (ComfyUI dynamic LoRA)
        for f in self.weight_function:
            w = f(w)
        
        bias = self.bias.to(target_dtype) if self.bias is not None else None
        # Apply LowVram bias_function patches
        if bias is not None:
            for f in self.bias_function:
                bias = f(bias)
        
        return F.conv2d(input, w, bias, self.stride, self.padding, self.dilation, self.groups)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Override to include dequantized weight in state_dict for LoRA compatibility."""
        # Add dequantized weight (this is what LoRA key mapping needs)
        destination[prefix + 'weight'] = self.weight if keep_vars else self.weight.detach()
        # Add bias if present
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Override to expose weight as a parameter for ModelPatcher compatibility."""
        # Yield the dequantized weight as if it were a parameter
        yield prefix + ('.' if prefix else '') + 'weight', self.weight
        if self.bias is not None:
            yield prefix + ('.' if prefix else '') + 'bias', self.bias

# --- Loader Node ---

class HSWQLoader:
    """
    HSWQ V2 (Scaled FP8) Full VRAM-Optimized Loader
    
    Loads models with .scale keys and replaces layers with custom HSWQ layers
    that store weights in FP8 but compute in FP16 (runtime descaling).
    This achieves:
    - 50% Storage Reduction
    - 50% VRAM Reduction (FP8 storage)
    - High Quality (Scaled reconstruction)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_hswq_checkpoint_vram_opt"
    CATEGORY = "HSWQ"
    TITLE = "HSWQ Scaled FP8 Loader (VRAM Opt)"

    def load_hswq_checkpoint_vram_opt(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        print(f"[HSWQ] Loading Scaled FP8 Model (VRAM Optimized): {ckpt_name}")
        
        # 1. Load state_dict to CPU first
        state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")
        
        scale_map = {} # key: weight_key, value: scale_tensor
        weight_map = {} # key: weight_key, value: fp8_weight_tensor (stored BEFORE dequantization)
        
        # Extract scales and weights, then DEQUANTIZE for ComfyUI's load_state_dict
        # This is crucial: PyTorch's load_state_dict cannot load FP8 tensors into FP16 nn.Parameters
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if key.endswith(".scale"):
                if key.endswith(".weight.scale"):
                    weight_key = key.replace(".weight.scale", ".weight")
                else:
                    weight_key = key.replace(".scale", ".weight")
                
                if weight_key in state_dict:
                    fp8_weight = state_dict[weight_key]
                    scale = state_dict[key]
                    
                    # Store original FP8 weight for HSWQ layer replacement later
                    scale_map[weight_key] = scale.clone()
                    weight_map[weight_key] = fp8_weight.clone()
                    
                    # DEQUANTIZE to FP16 for ComfyUI's load_state_dict compatibility
                    # This ensures the model loads correctly and LoRA can be applied
                    dequantized = dequantize_fp8_weight(fp8_weight, scale, torch.float16)
                    state_dict[weight_key] = dequantized
                    
                    # Mark scale key for removal (ComfyUI doesn't need it)
                    keys_to_remove.append(key)
        
        # Remove scale keys from state_dict
        for key in keys_to_remove:
            del state_dict[key]
        
        print(f"[HSWQ] Dequantized {len(scale_map)} FP8 weights to FP16 for model loading")
        
        # 2. Inject into ComfyUI loading mechanism
        # Monkey patch load_torch_file to avoid re-reading the file from disk
        original_loader = comfy.utils.load_torch_file
        
        def patched_loader(path, safe_load=False, device=None, return_metadata=False):
            if os.path.normpath(path) == os.path.normpath(ckpt_path):
                if return_metadata:
                    # safetensors header info might be lost here but usually empty dict is enough for inference 
                    # unless strict metadata checking is performed.
                    return state_dict, {} 
                return state_dict
            return original_loader(path, safe_load, device, return_metadata=return_metadata)
        
        comfy.utils.load_torch_file = patched_loader
        
        try:
            # 3. Standard Load (uses patched loader with DEQUANTIZED weights)
            # Now ComfyUI can properly load all weights and LoRA will work correctly
            model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, 
                output_vae=True, 
                output_clip=True, 
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
        finally:
            # Restore original loader
            comfy.utils.load_torch_file = original_loader

        # 4. Dynamic Layer Replacement
        print("[HSWQ] Replacing layers with FP8 Runtime-Descaling variants...")
        hswq_replaced_count = 0
        fp8_param_bytes = 0
        unet = model.model.diffusion_model
        
        # Collect modules to replace (avoid modifying during iteration)
        modules_to_replace = []
        for name, module in unet.named_modules():
            key_name = f"model.diffusion_model.{name}.weight"
            if key_name in scale_map:
                modules_to_replace.append((name, module, scale_map[key_name], weight_map[key_name]))
        
        for name, module, scale, weight_val in modules_to_replace:
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = unet.get_submodule(parent_name)
            else:
                parent_name = ""
                child_name = name
                parent = unet

            if isinstance(module, nn.Linear):
                new_layer = HSWQLinear(module, weight_val, scale)
                setattr(parent, child_name, new_layer)
                hswq_replaced_count += 1
                fp8_param_bytes += weight_val.numel()  # FP8 = 1 byte per element
            elif isinstance(module, nn.Conv2d):
                new_layer = HSWQConv2d(module, weight_val, scale)
                setattr(parent, child_name, new_layer)
                hswq_replaced_count += 1
                fp8_param_bytes += weight_val.numel()
        
        fp8_mb = fp8_param_bytes / (1024 * 1024)
        fp16_equivalent_mb = (fp8_param_bytes * 2) / (1024 * 1024)
        print(f"[HSWQ] Replaced {hswq_replaced_count} layers with FP8 storage.")
        print(f"[HSWQ] VRAM for FP8 weights: {fp8_mb:.1f} MB (vs {fp16_equivalent_mb:.1f} MB if FP16)")
        
        # Reset LoRA log counters for fresh tracking
        HSWQLinear._lora_log_count = 0
        HSWQConv2d._lora_log_count = 0
        
        # Verify state_dict contains weights for LoRA key mapping
        test_sd = model.model.state_dict()
        weight_keys = [k for k in test_sd.keys() if k.endswith('.weight') and 'diffusion_model' in k]
        print(f"[HSWQ] Model state_dict has {len(weight_keys)} diffusion_model weight keys for LoRA mapping")
        
        # Test that set_weight is discoverable
        test_layer = unet.get_submodule("input_blocks.1.0.out_layers.3")
        has_set_weight = hasattr(test_layer, 'set_weight') and callable(getattr(test_layer, 'set_weight', None))
        print(f"[HSWQ] HSWQ layers have set_weight method: {has_set_weight}")
        
        # Clear CPU memory
        del state_dict
        del weight_map
        del scale_map
        del test_sd
        import gc
        gc.collect()
        
        return (model, clip, vae)
