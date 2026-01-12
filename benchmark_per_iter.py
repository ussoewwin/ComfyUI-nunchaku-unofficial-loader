"""
各イテレーション後にsynchronize()して正確な時間を測定
"""

import torch
import time
import json
import sys
sys.path.insert(0, r"D:\USERFILES\ComfyUI\ComfyUI")

def benchmark_per_iter():
    from pathlib import Path
    from safetensors.torch import load_file
    from safetensors import safe_open
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel, convert_sdxl_state_dict
    
    print("=" * 60)
    print("Nunchaku SDXL: 各イテレーション後にsynchronize()")
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
    print("Model loaded.\n")

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

    # Warmup (3回、各回後にsync)
    print("Warmup...")
    with torch.no_grad():
        for _ in range(3):
            out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
            torch.cuda.synchronize()

    # [1] バッチ計測（最後にだけsync）- 元のベンチマーク方法
    print("\n[1] バッチ計測（最後にだけsync）:")
    iters = 20
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
    torch.cuda.synchronize()
    time_batch = (time.time() - start) / iters * 1000
    print(f"    Time: {time_batch:.2f} ms/iter")

    # [2] 各イテレーション計測（毎回sync）- 正確な計測
    print("\n[2] 各イテレーション計測（毎回sync）:")
    times = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.time()
            out = unet(sample, timestep, context, added_cond_kwargs=added_cond_kwargs)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
    time_per_iter = sum(times) / len(times)
    print(f"    Time: {time_per_iter:.2f} ms/iter")
    print(f"    Min:  {min(times):.2f} ms")
    print(f"    Max:  {max(times):.2f} ms")

    print("\n" + "=" * 60)
    print(f"バッチ計測: {time_batch:.2f} ms")
    print(f"個別計測:   {time_per_iter:.2f} ms")
    print(f"差:         {time_per_iter - time_batch:.2f} ms ({(time_per_iter/time_batch - 1)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_per_iter()
