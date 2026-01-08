"""
Check manual_cast_dtype
"""
import torch
import json
import sys
sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")

from pathlib import Path
from safetensors.torch import load_file
from safetensors import safe_open
import comfy.model_management as model_management
import comfy.utils

model_path = Path(r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\r128_svdq_fp4_bluePencilXL_v031_integrated.safetensors")

with safe_open(str(model_path), framework="pt") as f:
    metadata = f.metadata()
sd = load_file(str(model_path))

parameters = comfy.utils.calculate_parameters(sd)
weight_dtype = comfy.utils.weight_dtype(sd)
load_device = model_management.get_torch_device()

print(f"weight_dtype: {weight_dtype}")
print(f"load_device: {load_device}")

# SDXL supported dtypes (from config)
supported_dtypes = [torch.float16, torch.bfloat16]
unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=supported_dtypes, weight_dtype=weight_dtype)
print(f"unet_dtype: {unet_dtype}")

manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, supported_dtypes)
print(f"manual_cast_dtype: {manual_cast_dtype}")

if manual_cast_dtype is not None:
    print("WARNING: manual_cast_dtype is set - this may cause slow inference!")
else:
    print("OK: manual_cast_dtype is None")
