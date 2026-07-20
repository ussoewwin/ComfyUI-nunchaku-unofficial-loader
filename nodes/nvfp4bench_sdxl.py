import argparse
import torch

import transformers_clip_compat

transformers_clip_compat.apply()

from diffusers import StableDiffusionXLPipeline
import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim
import os
import gc
import time
import sys

# ComfyUI-native load via comfy.sd (QUANT_ALGOS float8_e4m3fn / int8_tensorwise).
# Do NOT require nodes.py — cloud checkouts of ComfyUI-master may lack root .py
# files while still shipping the comfy/ package (ops.py, sd.py, sample.py).
import logging
import types as _types

logging.getLogger("comfy").setLevel(logging.WARNING)


def _resolve_comfy_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = []
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(os.path.abspath(env))
    candidates.append(os.path.join(repo_root, "ComfyUI-master"))
    candidates.append(os.path.join(repo_root, "ComfyUI"))
    candidates.append(os.path.abspath("ComfyUI-master"))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(os.path.join(path, "comfy")):
            return path
    return candidates[0]


COMFY_PATH = _resolve_comfy_path()


def _ensure_comfy_complete():
    """Cloud checkouts of ComfyUI-master often ship a truncated tree.
    Restore missing root .py files and the entire comfy/ tree from upstream,
    Preserve existing ops.py if present. INT8 Conv2d support is NOT via
    permanently patched ops.py — see int8/comfy_quant_int8.py monkey-patch."""
    comfy_dir = os.path.join(COMFY_PATH, "comfy")
    # Always ensure comfy/__init__.py exists first.
    init_path = os.path.join(comfy_dir, "__init__.py")
    if not os.path.isfile(init_path):
        try:
            with open(init_path, "w") as _f:
                _f.write("")
        except Exception as e:
            print(f"[BENCH] create comfy/__init__.py failed: {e}")
    # Root-level sentinels (node_helpers.py etc. are required by comfy.sd).
    root_sentinels = ("node_helpers.py", "nodes.py", "main.py", "folder_paths.py",
                      "app.py", "server.py", "types.py")
    root_missing = [f for f in root_sentinels
                    if not os.path.isfile(os.path.join(COMFY_PATH, f))]
    comfy_sentinels = ("model_management.py", "memory_management.py",
                       "quant_ops.py", "model_patcher.py", "sd.py", "sample.py",
                       "utils.py", "cli_args.py", "options.py", "samplers.py",
                       "lora.py", "hooks.py", "latent_formats.py", "model_base.py",
                       "model_detection.py", "model_sampling.py",
                       "supported_models.py", "supported_models_base.py",
                       "clip_vision.py", "clip_model.py", "sd1_clip.py",
                       "sdxl_clip.py", "diffusers_convert.py", "diffusers_load.py",
                       "float.py", "gligen.py", "pinned_memory.py",
                       "patcher_extension.py", "rmsnorm.py", "nested_tensor.py",
                       "pixel_space_convert.py", "multigpu.py",
                       "model_prefetch.py", "sampler_helpers.py",
                       "deploy_environment.py", "comfy_api_env.py",
                       "lora_convert.py", "audio_encoders")
    comfy_missing = [f for f in comfy_sentinels
                     if not os.path.exists(os.path.join(comfy_dir, f))]
    if not root_missing and not comfy_missing:
        return
    print(f"[BENCH] ComfyUI incomplete (root missing {len(root_missing)}, comfy missing {len(comfy_missing)}); self-repairing...")
    sibling = os.path.join(os.path.dirname(COMFY_PATH), "_comfyui_full")
    if not os.path.isdir(os.path.join(sibling, "comfy")):
        import subprocess as _sp
        try:
            _sp.check_call(["git", "clone", "--depth", "1",
                            "https://github.com/comfyanonymous/ComfyUI.git", sibling],
                           stdout=_sp.DEVNULL, stderr=_sp.STDOUT)
            print(f"[BENCH] Cloned upstream ComfyUI -> {sibling}")
        except Exception as e:
            print(f"[BENCH] Clone failed: {e}; cannot self-repair.")
            return
    import shutil as _sh
    # Restore missing root .py files from upstream root.
    for f in root_missing:
        src = os.path.join(sibling, f)
        dst = os.path.join(COMFY_PATH, f)
        if os.path.isfile(src):
            try:
                _sh.copy2(src, dst)
                print(f"[BENCH] restored root/{f}")
            except Exception as e:
                print(f"[BENCH] restore root/{f} failed: {e}")
    if comfy_missing:
        src_comfy = os.path.join(sibling, "comfy")
        if not os.path.isdir(src_comfy):
            print(f"[BENCH] {src_comfy} missing; cannot repair comfy/.")
            return
        # Back up patched ops.py.
        patched_ops = None
        ops_path = os.path.join(comfy_dir, "ops.py")
        if os.path.isfile(ops_path):
            import tempfile as _tf
            patched_ops = _tf.mkstemp(suffix="_ops.py")[1]
            _sh.copy2(ops_path, patched_ops)
        try:
            if os.path.isdir(comfy_dir):
                _sh.rmtree(comfy_dir)
            _sh.copytree(src_comfy, comfy_dir)
            print(f"[BENCH] restored full comfy/ from upstream")
        except Exception as e:
            print(f"[BENCH] copytree failed: {e}")
            if patched_ops and os.path.isfile(patched_ops):
                os.unlink(patched_ops)
            return
        if patched_ops and os.path.isfile(patched_ops):
            _sh.copy2(patched_ops, ops_path)
            os.unlink(patched_ops)
            print(f"[BENCH] restored patched ops.py")
        if not os.path.isfile(init_path):
            with open(init_path, "w") as _f:
                _f.write("")
            print(f"[BENCH] created comfy/__init__.py")


_ensure_comfy_complete()

if COMFY_PATH not in sys.path:
    sys.path.insert(0, COMFY_PATH)


def _install_comfy_aimdo_stub():
    """ComfyUI-master hard-imports comfy_aimdo.*; cloud envs often lack it."""
    try:
        import comfy_aimdo.host_buffer  # noqa: F401
        return False
    except Exception:
        pass

    class _HostBuffer:
        def __init__(self, *args, **kwargs):
            pass

        def get_raw_address(self):
            return 0

        def read_file_slice(self, *args, **kwargs):
            return None

    class _ModelVBAR:
        def __init__(self, *args, **kwargs):
            pass

        def loaded_size(self):
            return 0

    class _VRAMBuffer:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return None

    class _ModelMMAP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("comfy_aimdo stub: ModelMMAP is unavailable (aimdo disabled)")

        def get(self):
            return 0

        def get_file_handle(self):
            return None

    pkg = _types.ModuleType("comfy_aimdo")
    pkg.__path__ = []

    host_buffer = _types.ModuleType("comfy_aimdo.host_buffer")
    host_buffer.HostBuffer = _HostBuffer
    host_buffer.read_file_to_device = lambda *a, **k: None

    model_vbar = _types.ModuleType("comfy_aimdo.model_vbar")
    model_vbar.ModelVBAR = _ModelVBAR
    model_vbar.vbar_fault = lambda *a, **k: None
    model_vbar.vbar_signature_compare = lambda *a, **k: False
    model_vbar.vbar_unpin = lambda *a, **k: None
    model_vbar.vbars_analyze = lambda *a, **k: 0
    model_vbar.vbars_reset_watermark_limits = lambda *a, **k: None

    torch_mod = _types.ModuleType("comfy_aimdo.torch")
    torch_mod.aimdo_to_tensor = lambda *a, **k: None
    torch_mod.hostbuf_to_tensor = lambda *a, **k: None

    vram_buffer = _types.ModuleType("comfy_aimdo.vram_buffer")
    vram_buffer.VRAMBuffer = _VRAMBuffer

    model_mmap = _types.ModuleType("comfy_aimdo.model_mmap")
    model_mmap.ModelMMAP = _ModelMMAP

    control = _types.ModuleType("comfy_aimdo.control")
    control.init = lambda *a, **k: False
    control.init_devices = lambda *a, **k: False
    control.analyze = lambda *a, **k: None
    control.set_log_debug = lambda *a, **k: None
    control.set_log_critical = lambda *a, **k: None
    control.set_log_error = lambda *a, **k: None
    control.set_log_warning = lambda *a, **k: None
    control.set_log_info = lambda *a, **k: None

    pkg.host_buffer = host_buffer
    pkg.model_vbar = model_vbar
    pkg.torch = torch_mod
    pkg.vram_buffer = vram_buffer
    pkg.model_mmap = model_mmap
    pkg.control = control

    sys.modules["comfy_aimdo"] = pkg
    sys.modules["comfy_aimdo.host_buffer"] = host_buffer
    sys.modules["comfy_aimdo.model_vbar"] = model_vbar
    sys.modules["comfy_aimdo.torch"] = torch_mod
    sys.modules["comfy_aimdo.vram_buffer"] = vram_buffer
    sys.modules["comfy_aimdo.model_mmap"] = model_mmap
    sys.modules["comfy_aimdo.control"] = control
    return True


_AIMDO_STUBBED = _install_comfy_aimdo_stub()


def _install_psutil_stub():
    """comfy.model_management imports psutil at module load. Cloud bench envs
    may lack it; provide a minimal stub so import succeeds without forcing
    a pip install."""
    try:
        import psutil  # noqa: F401
        return False
    except Exception:
        pass

    class _VMem:
        def __init__(self):
            self.total = 0
            self.available = 0
            self.percent = 0.0
            self.used = 0
            self.free = 0

    class _Swap:
        def __init__(self):
            self.total = 0
            self.used = 0
            self.free = 0
            self.percent = 0.0

    class _PSUtilStub:
        def virtual_memory(self):
            return _VMem()

        def swap_memory(self):
            return _Swap()

        def cpu_percent(self, *a, **k):
            return 0.0

        def cpu_count(self, *a, **k):
            return 1

        def disk_usage(self, *a, **k):
            return None

    import types as _t
    mod = _t.ModuleType("psutil")
    stub = _PSUtilStub()
    mod.virtual_memory = stub.virtual_memory
    mod.swap_memory = stub.swap_memory
    mod.cpu_percent = stub.cpu_percent
    mod.cpu_count = stub.cpu_count
    mod.disk_usage = stub.disk_usage
    sys.modules["psutil"] = mod
    return True


_PSUTIL_STUBBED = _install_psutil_stub()

# ComfyUI's cli_args.py calls parser.parse_args() at import time, which would
# swallow our bench argv (--fp16, --fp8, --prompt, ...). Temporarily clear
# sys.argv so ComfyUI parses an empty arg list, then restore it.
_saved_argv = list(sys.argv)
sys.argv = [sys.argv[0]] if sys.argv else []
_comfy_import_error = None
try:
    import comfy.options
    # Force ComfyUI to NOT parse our argv (belt-and-suspenders).
    comfy.options.args_parsing = False
    import comfy.model_management
    import comfy.ops
    import comfy.sample
    import comfy.sd
    import comfy.utils
except Exception as _e:
    _comfy_import_error = _e
    # Fall back to direct import without the options pre-import (some checkouts
    # ship comfy.model_management without a separate comfy.options module).
    try:
        import comfy.model_management
        import comfy.ops
        import comfy.sample
        import comfy.sd
        import comfy.utils
    except Exception:
        pass
finally:
    sys.argv = _saved_argv

if _comfy_import_error is not None and "comfy.ops" not in dir():
    print(f"Error: Could not import ComfyUI comfy package from {COMFY_PATH}: {type(_comfy_import_error).__name__}: {_comfy_import_error}")
    print("Ensure ComfyUI-master/comfy is present, or set COMFYUI_PATH.")
    sys.exit(1)


try:
    # INT8 comfy_quant: inject Quantized Conv2d + normalize comfy_quant JSON.
    # Local package only — never import from ComfyUI-nunchaku-unofficial-loader.
    _BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
    if _BENCH_DIR not in sys.path:
        sys.path.insert(0, _BENCH_DIR)
    from int8.comfy_quant_int8 import (  # noqa: E402
        apply_comfy_quant_int8_patches,
        checkpoint_looks_like_comfy_quant_int8,
        _int8_quant_conv_scope,
    )
    import int8.comfy_quant_int8 as _cq_int8  # noqa: E402
    from int4.comfy_quant_int4 import (  # noqa: E402
        apply_comfy_quant_int4_patches,
        checkpoint_looks_like_comfy_quant_int4,
    )
    import int4.comfy_quant_int4 as _cq_int4  # noqa: E402
    from nvfp4.comfy_quant_nvfp4 import (  # noqa: E402
        apply_comfy_quant_nvfp4_patches,
        checkpoint_looks_like_comfy_quant_nvfp4,
        nvfp4_forward_stats,
        reset_nvfp4_forward_stats,
    )
    import nvfp4.comfy_quant_nvfp4 as _cq_nvfp4  # noqa: E402

    apply_comfy_quant_int8_patches()
    apply_comfy_quant_int4_patches()
    apply_comfy_quant_nvfp4_patches()
    _mp_ops = comfy.ops.mixed_precision_ops()
    with _int8_quant_conv_scope():
        _has_conv_int8 = hasattr(comfy.ops.mixed_precision_ops(), "Conv2d")

    print(f"[BENCH] COMFY_PATH: {COMFY_PATH}")
    print(f"[BENCH] comfy.ops: {comfy.ops.__file__}")
    print(f"[BENCH] int8_tensorwise: {'int8_tensorwise' in comfy.ops.QUANT_ALGOS}")
    print(f"[BENCH] convrot_w4a4: {'convrot_w4a4' in comfy.ops.QUANT_ALGOS}")
    print(f"[BENCH] nvfp4: {'nvfp4' in comfy.ops.QUANT_ALGOS}")
    print(f"[BENCH] comfy_quant_int8 patched: {_cq_int8._PATCHES_APPLIED}")
    print(f"[BENCH] comfy_quant_int4 patched: {_cq_int4._PATCHES_APPLIED}")
    print(f"[BENCH] comfy_quant_nvfp4 patched: {_cq_nvfp4._PATCHES_APPLIED}")
    print(
        f"[BENCH] mixed_precision_ops Conv2d inject: "
        f"{getattr(comfy.ops.mixed_precision_ops, '_hswq_int8_conv_patched', False)}"
    )
    print(f"[BENCH] MixedPrecisionOps.Conv2d (default): {hasattr(_mp_ops, 'Conv2d')}")
    print(f"[BENCH] MixedPrecisionOps.Conv2d (INT8 scope): {_has_conv_int8}")
    print(f"[BENCH] patch file int8: {os.path.abspath(_cq_int8.__file__)}")
    print(f"[BENCH] patch file int4: {os.path.abspath(_cq_int4.__file__)}")
    print(f"[BENCH] patch file nvfp4: {os.path.abspath(_cq_nvfp4.__file__)}")
    print(f"[BENCH] comfy_aimdo stubbed: {_AIMDO_STUBBED}")
except Exception as e:
    print(f"Error: Could not import ComfyUI comfy package from {COMFY_PATH}: {type(e).__name__}: {e}")
    comfy_dir = os.path.join(COMFY_PATH, "comfy")
    if os.path.isdir(comfy_dir):
        try:
            listing = sorted(os.listdir(comfy_dir))
            print(f"comfy/ listing ({len(listing)} entries): {listing[:40]}")
            for key in ("__init__.py", "model_management.py", "ops.py", "sd.py", "sample.py"):
                p = os.path.join(comfy_dir, key)
                print(f"  {key}: exists={os.path.isfile(p)} size={os.path.getsize(p) if os.path.isfile(p) else 0}")
        except Exception as ex:
            print(f"Failed to list comfy/: {ex}")
    elif os.path.isdir(COMFY_PATH):
        try:
            print(f"COMFY_PATH listing: {os.listdir(COMFY_PATH)[:20]}")
        except Exception:
            pass
    else:
        print(f"COMFY_PATH does not exist: {COMFY_PATH}")
    # Print git HEAD for checkout staleness diagnosis
    try:
        import subprocess as _sp
        head = _sp.check_output(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log", "-1", "--oneline"], text=True, stderr=_sp.DEVNULL).strip()
        print(f"Repo HEAD: {head}")
    except Exception:
        pass
    print("Ensure ComfyUI-master/comfy is present, or set COMFYUI_PATH.")
    sys.exit(1)

# Enforce deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_pipeline(path, device="cuda"):
    print(f"Loading model: {os.path.basename(path)}...")
    try:
        ckpt_path = os.path.abspath(path)
        use_int8_scope = checkpoint_looks_like_comfy_quant_int8(ckpt_path)
        use_int4 = checkpoint_looks_like_comfy_quant_int4(ckpt_path)
        use_nvfp4 = checkpoint_looks_like_comfy_quant_nvfp4(ckpt_path)
        print(f"[BENCH] INT8 Conv2d load scope: {use_int8_scope}")
        print(f"[BENCH] INT4 convrot_w4a4 detect: {use_int4}")
        print(f"[BENCH] NVFP4 detect: {use_nvfp4}")
        if use_int8_scope:
            with _int8_quant_conv_scope():
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=None,
                )
        else:
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=None,
            )
        model, clip, vae = out[0], out[1], out[2]
        return model, clip, vae
    except Exception as e:
        import traceback
        print(f"Error loading model: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

def generate_image_fixed(model, clip, vae, prompt, seed, steps):
    # CLIP encode (same as nodes.CLIPTextEncode — no nodes.py required)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    # Empty latent 1024x1024 SDXL
    latent_image = torch.zeros(
        [1, 4, 1024 // 8, 1024 // 8],
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    latent = {"samples": latent_image, "downscale_ratio_spacial": 8}
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent["samples"], latent.get("downscale_ratio_spacial", None), None
    )
    noise = comfy.sample.prepare_noise(latent_image, seed, None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        7.0,
        "dpmpp_2m",
        "karras",
        positive,
        negative,
        latent_image,
        denoise=1.0,
        callback=None,
        disable_pbar=False,
        seed=seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    images = vae.decode(samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    img_array = 255.0 * images[0].detach().cpu().numpy()
    image = Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))

    return image, end_time - start_time

def calculate_metrics(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # MSE (mean squared error)
    mse = np.mean((arr1 - arr2) ** 2)

    # SSIM (structural similarity)
    score_ssim = ssim(arr1, arr2, win_size=3, channel_axis=2, data_range=255)

    return mse, score_ssim

def main():
    parser = argparse.ArgumentParser(description="Robust SDXL FP8 Fidelity Benchmark")
    parser.add_argument("--fp16", required=True, help="Path to Baseline (FP16) model")
    parser.add_argument("--fp8", required=True, help="Path to Quantized (FP8) model")
    parser.add_argument("--prompt", required=True, help="Benchmark prompt")
    parser.add_argument("--seed", type=int, default=123456789, help="Fixed seed for reproduction")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Benchmark Config ---")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"------------------------")

    # 1. FP16 (Baseline) Generation
    print("\n=== 1. Generating Baseline (FP16) ===")
    model16, clip16, vae16 = load_pipeline(args.fp16, device)
    img_fp16, time_fp16 = generate_image_fixed(model16, clip16, vae16, args.prompt, args.seed, args.steps)
    img_fp16.save("bench_result_fp16.png")
    print(f"FP16 Time: {time_fp16:.2f}s")

    # Full memory release (keep CLIP/VAE for FP8 side to isolate UNet difference)
    del model16
    gc.collect()
    torch.cuda.empty_cache()

    # 2. NVFP4 (Quantized) Generation
    print("\n=== 2. Generating Quantized (NVFP4) ===")
    reset_nvfp4_forward_stats()
    model8, clip8, vae8 = load_pipeline(args.fp8, device)
    img_fp8, time_fp8 = generate_image_fixed(model8, clip16, vae16, args.prompt, args.seed, args.steps)
    img_fp8.save("bench_result_fp8.png")
    _nvfp4_stats = nvfp4_forward_stats()
    print(f"NVFP4 Time: {time_fp8:.2f}s")
    print(
        f"[BENCH] NVFP4 TC scaled_mm_hits={_nvfp4_stats['scaled_mm_hits']} "
        f"dequant_fallbacks={_nvfp4_stats['dequant_fallbacks']}"
    )

    del model8, clip8, vae8, clip16, vae16
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Comparison
    print("\n=== 3. Calculating Metrics ===")

    # Size check (prevent error when models/settings differ)
    if img_fp16.size != img_fp8.size:
        print(f"Error: Image sizes do not match! FP16:{img_fp16.size}, NVFP4:{img_fp8.size}")
        print("Different models or settings used.")
        sys.exit(1)

    mse, score = calculate_metrics(img_fp16, img_fp8)

    print(f"--------------------------------------------------")
    print(f"MSE (Error): {mse:.4f} \t(0 is perfect match)")
    print(f"SSIM (Sim) : {score:.4f} \t(1.0 is perfect match)")
    print(
        f"NVFP4 TC   : hits={_nvfp4_stats['scaled_mm_hits']} "
        f"fallbacks={_nvfp4_stats['dequant_fallbacks']}"
    )
    print(f"--------------------------------------------------")

    # Grading logic
    if score > 0.98:
        grade = "PERFECT (S)"
    elif score > 0.95:
        grade = "EXCELLENT (A)"
    elif score > 0.90:
        grade = "GOOD (B)"
    else:
        grade = "WARNING (C)"

    print(f"Quality Grade: {grade}")

    # Difference image generation
    diff_img = ImageChops.difference(img_fp16, img_fp8)
    diff_img = ImageChops.multiply(diff_img, Image.new('RGB', diff_img.size, (10, 10, 10)))
    diff_img.save("bench_result_diff.png")
    print("Diff image saved: bench_result_diff.png")

if __name__ == "__main__":
    main()
