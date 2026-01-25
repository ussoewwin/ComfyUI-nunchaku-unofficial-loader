# ComfyUI-nunchaku-unofficial-loader
<p align="center">
<img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/main/icon.png" width="128">
</p>

<div align="center">

## ⚠️ WARNING

This is an **UNOFFICIAL** test version of this node.  
It may not work correctly depending on your environment.  
**(Unfortunately, no speed advantage at present)**

</div>

## ⚠️ IMPORTANT NOTICE – SDXL SVDQ DEPRECATION

After extensive long-term testing, repeated real-world benchmarking, and significant development effort devoted specifically to improving generation speed and VRAM (VRSAM) efficiency, active development of SDXL SVDQ (4-bit) support in this repository has been discontinued.

Throughout this process, multiple optimization strategies were evaluated, including kernel behavior analysis, runtime integration adjustments, and execution-path tuning. However, despite these efforts, the fundamental limitations of SDXL SVDQ remained unchanged.

For SDXL models, SVDQ / FP4 quantization does **NOT** provide practical advantages over standard fp16 execution:

- No consistent generation speed improvement, even after extensive tuning
- No meaningful VRAM (VRSAM) reduction in real-world usage scenarios
- Additional runtime overhead caused by fp16 conversion, kernel dispatch, and integration costs

While a reduction in model file size was achieved, this factor alone is insufficient to justify continued SDXL SVDQ support, given the lack of runtime and memory efficiency benefits.

As a result:

- SDXL SVDQ models (e.g. Nunchaku-R128-SDXL-Series) are being deprecated
- Related Hugging Face repositories will be removed within approximately 30 days
- SDXL SVDQ should be considered experimental / archival only
- This repository will no longer be updated with new SDXL SVDQ models.

### Future Direction: fp8e4m3

Future SDXL-related development efforts are shifting toward fp8e4m3-based compression and formats.

This decision is based on extensive comparative testing, which demonstrated that fp8e4m3 provides a substantially better balance between performance, memory usage, and image quality:

- Fully compatible with standard ComfyUI loaders
- Image quality effectively equivalent to fp16
- No generation speed penalty
- No increase in VRAM usage
- Model size reduction comparable to 4-bit SVDQ, without its runtime drawbacks

fp8e4m3-based SDXL models, compression scripts, and related technical documentation will continue to be published separately.

### Status of This Repository

This repository remains available strictly for:

- Reference and research purposes
- Advanced experimentation
- Studying Nunchaku integration details, including:
  - forward overrides
  - LoRA mapping behavior
  - ControlNet support
  - First Block Cache
  - SDXL-specific edge cases and limitations

No guarantees are provided regarding future SDXL SVDQ functionality. **Use at your own risk.**

---

These are Nunchaku unofficial loaders quantized with SVDQ (SVDQuant) method, based on [ComfyUI-nunchaku](https://github.com/nunchaku-ai/ComfyUI-nunchaku) with custom additions.

## Installation

### Quick Install

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
# Windows
git clone https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader "%USERPROFILE%\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader"

# Linux/Mac
git clone https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader ~/ComfyUI/custom_nodes/ComfyUI-nunchaku-unofficial-loader
```

Restart ComfyUI to load the nodes.

## Requirements

**Nunchaku library**: You **MUST** have the Nunchaku library version **v1.1.0 or later** installed. This is a hard requirement - other versions will not work.

**Pre-built package**: If you are using Nunchaku with CUDA 13.0 (cu130), use the pre-built package available at [ussoewwin/nunchaku-build-on-cu130-windows](https://huggingface.co/ussoewwin/nunchaku-build-on-cu130-windows).

## Nodes

### Checkpoint Loader (SDXL)

<img src="png/fp8e4m3.png" alt="Checkpoint Loader (SDXL) Node" width="400">

A ComfyUI node that loads **MODEL** and **CLIP** from standard SDXL checkpoints, with optional device selection and FP8 precision support. Use it like the standard Load Checkpoint node; it outputs MODEL and CLIP only (no VAE).

#### Features

- **Checkpoint Loading**: Loads both UNet (MODEL) and CLIP from a single SDXL checkpoint file (same as standard Load Checkpoint)
- **Device Selection**: Optional device parameter to choose GPU (e.g. `cuda:0`, `cuda:1`) or CPU for model loading
- **FP8 Precision**: Supports `default`, `fp8_e4m3fn`, `fp8_e4m3fn_fast`, and `fp8_e5m2` weight dtypes for memory-efficient inference
- **Standard ComfyUI Integration**: Uses `load_checkpoint_guess_config`; compatible with standard ComfyUI workflows

#### Usage Notes

- **Inputs**: `ckpt_name` (checkpoint file), `weight_dtype` (precision), and optionally `device`
- **Outputs**: MODEL and CLIP only; use a separate VAE loader if needed
- **Category**: Loaders (`loaders`)

### Nunchaku Ultimate SD Upscale

<img src="png/upscale.png" alt="Nunchaku Ultimate SD Upscale Node" width="400">

A ComfyUI node for upscaling images using tile-based image-to-image processing, specifically optimized for Nunchaku SDXL models.

#### Features

- **Tile-based Upscaling**: Processes images in tiles to handle high-resolution upscaling efficiently
- **Color Normalization**: Always normalizes Nunchaku SDXL VAE output to full dynamic range (0.0-1.0) before upscaling, fixing pale/washed-out colors
- **Multiple Modes**: Supports Linear, Chess, and None tile modes
- **Seam Fixing**: Includes multiple seam fixing modes (None, Band Pass, Half Tile, Half Tile + Intersections)
- **Module Isolation**: Prevents module reference conflicts with other custom nodes

#### Usage Notes

- **Requires ComfyUI_UltimateSDUpscale**: This node requires the `ComfyUI_UltimateSDUpscale` custom node to be installed
- **Color Range**: Automatically normalizes Nunchaku SDXL VAE's compressed color range (e.g., 0.15-0.85) to full range (0.0-1.0) to restore proper contrast and color saturation
- **Module Safety**: Uses isolated module loading to prevent conflicts with other custom nodes

### Nunchaku-ussoewwin Z-Image-Turbo DiT Loader

⚠️ **WARNING**: This is an **unofficial experimental loader** created as a prototype before the release of ComfyUI-Nunchaku 1.1.0. This is the author's personal testing environment. **Do not use this node.**

A ComfyUI node for loading Nunchaku-quantized Z-Image-Turbo models. This node provides support for loading 4-bit quantized Z-Image-Turbo models that have been processed using SVDQuant quantization.

<img src="png/node.png" alt="Nunchaku-ussoewwin Z-Image-Turbo DiT Loader Node" width="400">

#### Features

- **Model Loading**: Loads Nunchaku-quantized Z-Image-Turbo diffusion transformer models
- **CPU Offloading**: Automatic or manual CPU offloading support to reduce VRAM usage
- **Memory Management**: Configurable GPU memory usage with transformer block offloading options
- **Hardware Compatibility**: Automatic hardware compatibility checks for quantization support
- **Precision Support**: Supports both INT4 and FP4 quantization precisions

## Changelog

### Version 2.6.2

- Fixed NunchakuUltimateSDUpscale node registration issue with Nunchaku 1.2.0
  - Improved error handling in INPUT_TYPES to prevent node registration failures
  - Node now appears in UI even if ComfyUI_UltimateSDUpscale import fails
  - See [Issue #2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/issues/2) for details
- See [Release Notes v2.6.2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.6.2) for details

### Version 2.6.1

- Optimized LoRA processing performance for SDXL models
- See [Release Notes v2.6.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.6.1) for details

### Version 2.6

- Fixed ControlNet support for SDXL models (OpenPose, Depth, Canny, etc.)
- See [Release Notes v2.6](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.6) for details

### Version 2.5

- Added SDXL Integrated Loader node for unified checkpoint loading
  - Supports loading both UNet and CLIP from a single checkpoint file
  - Includes Flash Attention 2 support (enabled by default)
  - Automatically detects model configuration from checkpoint keys
- Reorganized node documentation order
- Updated SDXL DiT Loader with advanced user warning
- See [Release Notes v2.5](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.5) for details

### Version 2.4

- Added Flash Attention 2 support for SDXL DiT Loader
  - Optional acceleration feature enabled by default
  - Automatically applies FA2 to all attention layers (typically 140 layers in SDXL models)
  - Requires Flash Attention 2 to be installed in your environment
  - Can be disabled via the `enable_fa2` parameter if needed
- Updated SDXL DiT Loader node image
- See [Release Notes v2.4](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.4) for details

### Version 2.3

- Added Nunchaku Ultimate SD Upscale nodes with improved color normalization
- Improved First Block Cache with residual injection for better quality
- Fixed USDU color normalization for Nunchaku SDXL VAE output
- Fixed module reference separation to prevent data loss
- Optimized cache similarity calculation using fused kernels
- Added Flash Attention 2 support for SDXL DiT Loader (optional, enabled by default)
- See [Release Notes v2.3](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.3) for details

### Version 2.2

- Added First Block Cache feature for Nunchaku SDXL models
- See [Release Notes v2.2](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.2) for details

### Version 2.1

- Published LoRA Loader technical documentation
- See [Release Notes v2.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.1) for details

### Version 2.0

- Added SDXL DIT Loader support
- Added SDXL LoRA support
- Added ControlNet support for SDXL models
- See [Release Notes v2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.0) for details

### Version 1.1

- Added Diffsynth ControlNet support for Z-Image-Turbo models
  - Note: Does not work with standard model patch loader. Requires a custom node developed by the author.
- See [Release Notes v1.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/1.1) for details

### 2025-12-25

- Fixed import error for `NunchakuZImageDiTLoader` node by improving alternative import method with better path resolution (see [Issue #1](issues/1))

## Safety & License Notice

### Model Distribution & Usage

* **This repository does NOT distribute any model checkpoints, weights, or training data.**
* All model files (including SDXL checkpoints, quantized UNet files, CLIP, VAE, LoRA, and ControlNet models) **must be obtained separately by the user**.
* Users are solely responsible for ensuring that **all downloaded or generated model files comply with their respective licenses** (e.g., CreativeML Open RAIL, Apache-2.0, custom research licenses, etc.).
* The author does **not grant any rights** to redistribute, modify, or use third-party models beyond what is permitted by their original licenses.

### Quantized & Derived Models

* Quantized models (e.g., SVDQ / FP4 / INT4) are considered **derivative works** of the original checkpoints.
* Before sharing or redistributing quantized models, verify that the **original model license explicitly allows redistribution and derivative works**.
* Many SDXL-based models **do not permit redistribution**, even in quantized form.

### Experimental / Unofficial Status

* This project is an **UNOFFICIAL and experimental implementation**.
* It is **not affiliated with, endorsed by, or supported by the Nunchaku or ComfyUI core teams**.
* Behavior, performance, and compatibility may change without notice.
* Use at your own risk.

## License (Apache License 2.0)

This project is licensed under the **Apache License, Version 2.0**.

### Key Points

* Copyright © 2024–2025 ussoewwin
* You are free to **use, modify, and distribute** this software, including for commercial purposes.
* You **must retain**:
  * The original copyright notice
  * A copy of the Apache-2.0 license
  * Any existing NOTICE files (if present)
* If you modify the source code, you **must clearly indicate** that changes were made.
* This software is provided **"AS IS"**, without warranties or conditions of any kind.

See the full license text in [`LICENCE.txt`](./LICENCE.txt).
