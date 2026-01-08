import torch
import time
import json
import sys
import os

# Add ComfyUI-nunchaku to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
nunchaku_path = os.path.join(current_dir, "ComfyUI-nunchaku")
if nunchaku_path not in sys.path:
    sys.path.append(nunchaku_path)

from pathlib import Path
from safetensors.torch import load_file
from safetensors import safe_open
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel, convert_sdxl_state_dict

def benchmark():
    print("=" * 60)
    print("Nunchaku SDXL: ComfyUI Condition Benchmark")
    print("Conditions: Resolution 1216x1216, Batch Size 2 (CFG)")
    print("=" * 60)

    model_path = Path(r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\r128_svdq_fp4_bluePencilXL_v031_integrated.safetensors")
    
    # Load Model (Same as original benchmark)
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

    # Inputs for ComfyUI Condition
    batch_size = 2  # CFG doubles the batch size
    latent_size = 152  # 1216 / 8 = 152
    
    dtype = torch.bfloat16
    device = "cuda"

    sample = torch.randn((batch_size, 4, latent_size, latent_size), device=device, dtype=dtype)
    timestep = torch.tensor([999.0] * batch_size, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn((batch_size, 77, 2048), device=device, dtype=dtype)
    
    added_cond_kwargs = {
        "text_embeds": torch.randn((batch_size, 1280), device=device, dtype=dtype),
        "time_ids": torch.randn((batch_size, 6), device=device, dtype=dtype),
    }

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    
    # Benchmark
    print("Benchmarking...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    num_runs = 10
    with torch.no_grad():
        for _ in range(num_runs):
            _ = unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / num_runs * 1000
    
    print(f"Average Inference Time (BS={batch_size}, {latent_size*8}px): {avg_time:.2f} ms")

if __name__ == "__main__":
    benchmark()
