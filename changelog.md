# Changelog

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
