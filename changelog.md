# Changelog

## Version 3.2.2

- **Fixed**: INT8→Nunchaku VRAM handoff false-positive on non-SVDQ loads (including SDXL INT8 normal generation) — SVDQ detection no longer uses bare `"nunchaku" in __module__` (this extension’s INT8 Conv2d path contains that substring); handoff `_VER = 10` arms only for real Nunchaku SVDQ on the BaseModel, and native comfy_quant INT8 (any architecture) never arms handoff.
- See [Release Notes v3.2.2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.2.2) for details.

## Version 3.2.1

- **Fixed**: INT8 HSWQ (Dynamic VRAM) → Nunchaku SVDQ coexistence Abort — LowVramPatch and Dynamic LoRA bake restricted to `comfy.quant_ops.QuantizedTensor` only (never bare `torch.int8`); unidirectional VRAM handoff uses `detach(unpatch_all=True)` before SVDQ load.
- **Removed**: Reintroduced **HSWQ Pin Buffer Cache** again (not required for the Abort fix; Detailer-scoped pin pooling remains obsolete after AIMDO HostBuffer).
- **Docs**: Rewrote `md/HSWQ_INT8_NUNCHAKU_COEXISTENCE_GUIDE.md` for verified Abort causes vs PinCache correlation.
- See [Release Notes v3.2.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.2.1) for details.

## Version 3.2.0

- **Removed**: **HSWQ Pin Buffer Cache** (`nodes/hswq_pin_cache.py` and Detailer `hswq_pin_cache_scope`) — redundant after ComfyUI Dynamic VRAM / AIMDO `HostBuffer` updates (no thrashing `unpin` path). Batched Detailer three-phase flow kept; use native ComfyUI pin behavior.
- **Changed**: Display title forced to **HSWQ Checkpoint Loader (SDXL)** for the SDXL checkpoint loader node.
- See [Release Notes v3.2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.2.0) for details.

## Version 3.1.9

- **Added**: Native **comfy_quant INT8** (`int8_tensorwise`) load path for SDXL checkpoints — **HSWQ FP8/INT8 Loader (VRAM Opt)** auto-detects INT8 vs Scaled FP8; **HSWQ FP8 E4M3 UNet Loader** gains `int8_tensorwise` / auto-detect. Extension-side Conv2d quant support and INT8-safe LoRA bake under Dynamic VRAM.
- See [Release Notes v3.1.9](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.1.9) for details.

## Version 3.1.8

- **Added**: **HSWQ Save Image** (`NunchakuSaveImage`) — save `IMAGE` output as PNG or JPG (JPEG quality when JPG is selected).
- **Added**: **Nunchaku Ultimate SD Upscale** — `upscale_by` dropdown with **Auto** mode and `target_height` (default 4320) to derive scale from input height; fixed magnifications 0.05–4.00 remain available.
- See [Release Notes v3.1.8](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.1.8) for details.

## Version 3.1.7

- **Fixed**: Critical fix for severe output noise and `RuntimeError` in `NunchakuUltimateSDUpscale` when used with Lumina/HunYuan-DiT architectures. Corrected the conditioning tensor slicing logic to accurately extract T5/LLM features from concatenated tensors.
- See [Release Notes v3.1.7](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.1.7) for details.

## Version 3.1.3

- **Fixed**: Workaround for `RuntimeError` in `NunchakuUltimateSDUpscale` caused by a recent ComfyUI core change that concatenates multi-encoder conditioning along the feature dimension (e.g., 7680 instead of 2560) for Lumina/HunYuan-based models. Added automatic detection and truncation of these embeddings before sampling.
- See [Release Notes v3.1.3](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.1.3) for details.

## Version 3.1.2

- **Fixed**: Pin Buffer Cache (monkey-patch for `comfy.pinned_memory.pin_memory` / `unpin_memory`) is now enabled only while running `HSWQ Batched Detailer (SEGS)`. Outside of Detailer SEGS, the extension delegates back to ComfyUI's original pin/unpin behavior to avoid side effects in other nodes/workflows.

## Version 3.1.1

- **Fixed**: Bug fixes and corrections (loader registration, zimage model handling, USDU crop model patch).
- See [Release Notes v3.1.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.1.1) for details.

## Version 3.1.0

- **Added** two new nodes:
  - **HSWQ FP8 E4M3 UNet Loader** (`HSWQFP8E4M3UNetLoader`) — Standard UNet loader for HSWQ FP8 E4M3 models; extension also installs a Pin Buffer Cache to reduce `cudaHostRegister`/`cudaHostUnregister` overhead under Dynamic VRAM Loading.
  - **HSWQ Batched Detailer (SEGS)** — Detailer (SEGS)–style node that runs VAE encode → UNet sample → VAE decode in three phases (all encodes, then all samples, then all decodes) to minimize model switching and improve performance with Dynamic VRAM Loading.
- See [Release Notes v3.1.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.1.0) for details.

## Version 3.0.2

- **README**: FP8 (fp8e4m3) and torch.compile subsection updated — purpose (use this node with FP8 and torch.compile together) and patches description.
- See [Release Notes v3.0.2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/3.0.2) for details.

## Version 3.0.0

- **Breaking**: Aligned with SDXL SVDQ deprecation (see IMPORTANT NOTICE at top). Node registration reduced to the following three only:
  - **Nunchaku-ussoewwin SDXL Integrated Loader** (Checkpoint Loader style: single checkpoint)
  - **Nunchaku-ussoewwin SDXL DiT Loader (DualCLIP)** (UNet + CLIP from separate files)
  - **Nunchaku Ultimate SD Upscale**
- **Removed** from registration (no longer appear in ComfyUI):
  - Nunchaku-ussoewwin Z-Image-Turbo DiT Loader
  - Nunchaku-ussoewwin SDXL LoRA Stack V3
  - Nunchaku Apply First Block Cache Patch Advanced
- Future SDXL workflows are intended to use fp8e4m3 with standard ComfyUI loaders where applicable.

## Version 2.6.6

- **Fixed**: Fixed `AttributeError: 'Logger' object has no attribute 'mgpu_mm_log'` error that was causing prompt execution to crash. Replaced all instances of `logger.mgpu_mm_log()` with `logger.info()` in `model_management_mgpu.py`, `device_utils.py`, and `wrappers.py`.

## Version 2.6.3

- Added **Checkpoint Loader (SDXL)** node
  - Loads MODEL and CLIP from standard SDXL checkpoints with optional device selection and FP8 precision support
- Nunchaku SDXL SVDQ (4-bit) development discontinued; repository status updated (see IMPORTANT NOTICE at top)
- See [Release Notes v2.6.3](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.6.3) for details

## Version 2.6.2

- Fixed NunchakuUltimateSDUpscale node registration issue with Nunchaku 1.2.0
  - Improved error handling in INPUT_TYPES to prevent node registration failures
  - Node is standalone: uses bundled `usdu_bundle` and does not require ComfyUI_UltimateSDUpscale to be installed
  - See [Issue #2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/issues/2) for details
- See [Release Notes v2.6.2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.6.2) for details

## Version 2.6.1

- Optimized LoRA processing performance for SDXL models
- See [Release Notes v2.6.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.6.1) for details

## Version 2.6

- Fixed ControlNet support for SDXL models (OpenPose, Depth, Canny, etc.)
- See [Release Notes v2.6](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.6) for details

## Version 2.5

- Added SDXL Integrated Loader node for unified checkpoint loading
  - Supports loading both UNet and CLIP from a single checkpoint file
  - Includes Flash Attention 2 support (enabled by default)
  - Automatically detects model configuration from checkpoint keys
- Reorganized node documentation order
- Updated SDXL DiT Loader with advanced user warning
- See [Release Notes v2.5](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.5) for details

## Version 2.4

- Added Flash Attention 2 support for SDXL DiT Loader
  - Optional acceleration feature enabled by default
  - Automatically applies FA2 to all attention layers (typically 140 layers in SDXL models)
  - Requires Flash Attention 2 to be installed in your environment
  - Can be disabled via the `enable_fa2` parameter if needed
- Updated SDXL DiT Loader node image
- See [Release Notes v2.4](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.4) for details

## Version 2.3

- Added Nunchaku Ultimate SD Upscale nodes with improved color normalization
- Improved First Block Cache with residual injection for better quality
- Fixed USDU color normalization for Nunchaku SDXL VAE output
- Fixed module reference separation to prevent data loss
- Optimized cache similarity calculation using fused kernels
- Added Flash Attention 2 support for SDXL DiT Loader (optional, enabled by default)
- See [Release Notes v2.3](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.3) for details

## Version 2.2

- Added First Block Cache feature for Nunchaku SDXL models
- See [Release Notes v2.2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.2) for details

## Version 2.1

- Published LoRA Loader technical documentation
- See [Release Notes v2.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.1) for details

## Version 2.0

- Added SDXL DIT Loader support
- Added SDXL LoRA support
- Added ControlNet support for SDXL models
- See [Release Notes v2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.0) for details

## Version 1.1

- Added Diffsynth ControlNet support for Z-Image-Turbo models
  - Note: Does not work with standard model patch loader. Requires a custom node developed by the author.
- See [Release Notes v1.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/1.1) for details

## 2025-12-25

- Fixed import error for `NunchakuZImageDiTLoader` node by improving alternative import method with better path resolution (see [Issue #1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/issues/1))
