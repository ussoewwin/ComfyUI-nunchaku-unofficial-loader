"""
Test Nunchaku SDXL with same flow as ComfyUI - direct implementation
"""

import torch
import time
import json
import sys
import os

sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")
os.chdir(r"D:\USERFILES\ComfyUI\ComfyUI")

def benchmark_comfyui_flow():
    from pathlib import Path
    from safetensors.torch import load_file
    from safetensors import safe_open
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel, convert_sdxl_state_dict
    import comfy.model_management as model_management
    from comfy.model_base import ModelType, SDXL
    
    print("=" * 60)
    print("Nunchaku SDXL: via _apply_model vs direct call")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print()

    model_path = Path(r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\r128_svdq_fp4_bluePencilXL_v031_integrated.safetensors")
    
    print("Loading model...")
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
    print("Model loaded.")
    print()

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

    # [1] Direct UNet call (standalone)
    print("[1] Direct UNet call (standalone)")
    with torch.no_grad():
        for _ in range(3):
            out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
    torch.cuda.synchronize()

    iters = 20
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
    torch.cuda.synchronize()
    time_direct = (time.time() - start) / iters * 1000
    print(f"    Time: {time_direct:.2f} ms/iter")
    print(f"    it/s: {1000 / time_direct:.2f}")

    # [2] Simulation via NunchakuSDXL._apply_model
    print()
    print("[2] Simulate _apply_model processing")
    
    # Simulate model_sampling behavior
    from comfy.model_sampling import ModelSamplingDiscrete
    import comfy.supported_models

    # Use SDXL model sampling
    model_sampling = ModelSamplingDiscrete(comfy.supported_models.SDXL({}))
    
    sigma = torch.tensor([0.5], device="cuda", dtype=dtype)
    
    with torch.no_grad():
        for _ in range(3):
            # Simulate _apply_model processing
            xc = model_sampling.calculate_input(sigma, sample)
            xc = xc.to(dtype)
            t_step = model_sampling.timestep(sigma).float()
            
            # UNet call
            model_output = unet(
                sample=xc,
                timestep=t_step,
                encoder_hidden_states=context,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            
            # calculate_denoised
            result = model_sampling.calculate_denoised(sigma, model_output.float(), sample)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            xc = model_sampling.calculate_input(sigma, sample)
            xc = xc.to(dtype)
            t_step = model_sampling.timestep(sigma).float()
            
            model_output = unet(
                sample=xc,
                timestep=t_step,
                encoder_hidden_states=context,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            
            result = model_sampling.calculate_denoised(sigma, model_output.float(), sample)
    torch.cuda.synchronize()
    time_apply_model = (time.time() - start) / iters * 1000
    print(f"    Time: {time_apply_model:.2f} ms/iter")
    print(f"    it/s: {1000 / time_apply_model:.2f}")

    print()
    print("=" * 60)
    print(f"Direct call:         {time_direct:.2f} ms")
    print(f"Via _apply_model:    {time_apply_model:.2f} ms")
    print(f"Overhead:            {time_apply_model - time_direct:.2f} ms ({(time_apply_model/time_direct - 1)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_comfyui_flow()
