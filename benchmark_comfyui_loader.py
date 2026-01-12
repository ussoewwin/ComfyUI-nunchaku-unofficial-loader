"""
ComfyUIのローダー関数を使ってモデルをロードし、直接呼び出して測定
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
import comfy.model_management as model_management

# Import the loader function
from nodes.models.sdxl import load_diffusion_model_state_dict

print("=" * 60)
print("ComfyUIローダー関数 経由でロード → 直接呼び出し")
print("=" * 60)

gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}\n")

model_path = Path(r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\r128_svdq_fp4_bluePencilXL_v031_integrated.safetensors")

print("Loading model via ComfyUI loader...")
with safe_open(str(model_path), framework="pt") as f:
    metadata = f.metadata()
sd = load_file(str(model_path))

# Use the actual ComfyUI loader
model_patcher = load_diffusion_model_state_dict(sd, metadata)
print(f"Model loaded: {type(model_patcher.model.diffusion_model).__name__}")
print()

# Get the UNet
load_device = model_management.get_torch_device()
model_patcher.load(device_to=load_device)
unet = model_patcher.model.diffusion_model

# テスト入力
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

# 直接呼び出し
print("\n[1] ComfyUIローダー経由 → 直接呼び出し:")
times = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
print(f"    Time: {sum(times)/len(times):.2f} ms/iter")

# _apply_model経由
print("\n[2] _apply_model経由:")
model = model_patcher.model
sigma = torch.tensor([0.5], device="cuda", dtype=dtype)

# Warmup
with torch.no_grad():
    for _ in range(3):
        out = model._apply_model(sample, sigma, c_crossattn=context, text_embeds=added_cond_kwargs["text_embeds"], time_ids=added_cond_kwargs["time_ids"])
        torch.cuda.synchronize()

times = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        out = model._apply_model(sample, sigma, c_crossattn=context, text_embeds=added_cond_kwargs["text_embeds"], time_ids=added_cond_kwargs["time_ids"])
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
print(f"    Time: {sum(times)/len(times):.2f} ms/iter")

print("\n" + "=" * 60)
