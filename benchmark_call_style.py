"""
Load model using ComfyUI loader and measure by direct calls
"""

import torch
import time
import json
import sys
import os

sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")
os.chdir(r"D:\USERFILES\ComfyUI\ComfyUI")
sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader")

from pathlib import Path
from safetensors.torch import load_file
from safetensors import safe_open
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel, convert_sdxl_state_dict

print("=" * 60)
print("Via ComfyUI loader vs standalone comparison")
print("=" * 60)

gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}\n")

model_path = Path(r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\r128_svdq_fp4_bluePencilXL_v031_integrated.safetensors")

# Load using same method as standalone
print("Loading model (standalone method)...")
with safe_open(str(model_path), framework="pt") as f:
    metadata = f.metadata()

config = json.loads(metadata.get("config", "{}"))
quantization_config = json.loads(metadata.get("quantization_config", "{}"))
rank = quantization_config.get("rank", 128)
precision = quantization_config.get("precision", "nvfp4")

state_dict = load_file(str(model_path))

with torch.device("meta"):
    unet = NunchakuSDXLUNet2DConditionModel.from_config(config)
unet = unet.to(torch.bfloat16)
unet._patch_model(precision=precision, rank=rank)
unet = unet.to_empty(device="cuda")
converted_sd = convert_sdxl_state_dict(state_dict)
unet.load_state_dict(converted_sd, strict=False)
unet.eval()
print("Model loaded.\n")

# Test input
batch_size = 1
latent_size = 128
dtype = torch.bfloat16

sample = torch.randn(batch_size, 4, latent_size, latent_size, device="cuda", dtype=dtype)
timestep = torch.tensor([500.0], device="cuda")
context = torch.randn(batch_size, 77, 2048, device="cuda", dtype=dtype)
added_cond_kwargs = {
    "text_embeds": torch.randn(batch_size, 1280, device="cuda", dtype=dtype),
    "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda", dtype=torch.float32),
}

# Warmup
print("Warmup...")
with torch.no_grad():
    for _ in range(3):
        out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
        torch.cuda.synchronize()

# [1] Call with positional arguments
print("\n[1] Call with positional arguments:")
times = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
print(f"    Time: {sum(times)/len(times):.2f} ms/iter")

# [2] Call with keyword arguments (ComfyUI style)
print("\n[2] Call with keyword arguments (ComfyUI style):")
times = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        out = unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=context,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=None,
            mid_block_additional_residual=None,
            return_dict=False,
        )
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
print(f"    Time: {sum(times)/len(times):.2f} ms/iter")

# [3] return_dict=True (default)
print("\n[3] return_dict=True:")
times = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs, return_dict=True)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
print(f"    Time: {sum(times)/len(times):.2f} ms/iter")

print("\n" + "=" * 60)
