"""
Speed comparison with and without LoRA
"""

import torch
import time
import json


def benchmark_with_without_lora():
    from pathlib import Path
    from safetensors.torch import load_file
    from safetensors import safe_open
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel, convert_sdxl_state_dict

    print("=" * 60)
    print("Nunchaku SDXL: Speed comparison with and without LoRA")
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

    # Input
    batch_size = 1
    latent_size = 128
    sample = torch.randn(batch_size, 4, latent_size, latent_size, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device="cuda")
    encoder_hidden_states = torch.randn(batch_size, 77, 2048, device="cuda", dtype=torch.bfloat16)
    added_cond_kwargs = {
        "text_embeds": torch.randn(batch_size, 1280, device="cuda", dtype=torch.bfloat16),
        "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda", dtype=torch.float32),
    }

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            out = unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    torch.cuda.synchronize()

    # Benchmark (no LoRA = pure UNet)
    iters = 20
    print("[1] Pure UNet (no LoRA runtime)")
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            out = unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    torch.cuda.synchronize()
    time_no_lora = (time.time() - start) / iters * 1000
    print(f"    Time: {time_no_lora:.2f} ms/iter")
    print(f"    it/s: {1000 / time_no_lora:.2f}")

    # LoRA simulation: simulate additional 2xGEMM
    # Mimic actual LoRA loader behavior
    print()
    print("[2] Simulated LoRA overhead (2x GEMM per attention layer)")
    
    # SDXL has approximately 70 attention layers
    # Each layer adds 2 GEMMs: (N, in) @ (in, rank) @ (rank, out)
    # Typical parameters: in=1024, out=1024, rank=64(LoRA), N=128*128=16384
    num_attn_layers = 70
    lora_rank = 64
    in_features = 1024
    out_features = 1024
    seq_len = 128 * 128  # 1024x1024 latent

    # Dummy LoRA matrices
    down_t = torch.randn(in_features, lora_rank, device="cuda", dtype=torch.bfloat16)
    up_t = torch.randn(lora_rank, out_features, device="cuda", dtype=torch.bfloat16)
    x_lora = torch.randn(seq_len, in_features, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        add = (x_lora @ down_t) @ up_t
    torch.cuda.synchronize()

    # Benchmark: additional overhead per step
    start = time.time()
    for _ in range(iters):
        for _ in range(num_attn_layers):
            add = (x_lora @ down_t) @ up_t
    torch.cuda.synchronize()
    time_lora_overhead = (time.time() - start) / iters * 1000
    print(f"    LoRA overhead per step: {time_lora_overhead:.2f} ms")

    print()
    print("[3] Estimated total with LoRA")
    estimated = time_no_lora + time_lora_overhead
    print(f"    Pure UNet:     {time_no_lora:.2f} ms")
    print(f"    LoRA overhead: {time_lora_overhead:.2f} ms")
    print(f"    Estimated:     {estimated:.2f} ms ({1000/estimated:.2f} it/s)")

    print()
    print("=" * 60)


if __name__ == "__main__":
    benchmark_with_without_lora()
