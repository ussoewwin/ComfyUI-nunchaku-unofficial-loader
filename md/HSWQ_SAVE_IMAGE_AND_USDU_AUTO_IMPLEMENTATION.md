# HSWQ Save Image Node and Nunchaku Ultimate SD Upscale Auto Mode — Implementation Manual

This document describes the **HSWQ Save Image** output node and the **Auto `upscale_by` / `target_height`** enhancement to **HSWQ&Nunchaku Ultimate SD Upscale** (`NunchakuUltimateSDUpscale`).

Date of record: 2026-06-24  
Repository: `ComfyUI-nunchaku-unofficial-loader`

---

## 1. Summary

| Feature | Purpose |
|--------|---------|
| **HSWQ Save Image** | Output node that saves `IMAGE` tensors to ComfyUI's output folder as **PNG** or **JPG**, with optional PNG workflow metadata. |
| **Auto upscale** | Replaces the fixed `FLOAT upscale_by` slider with a dropdown: **Auto** (scale from `target_height`) or fixed values **0.05–4.00** (step 0.05). |

Typical workflow:

```
Load Image → … → HSWQ&Nunchaku Ultimate SD Upscale (upscale_by=Auto, target_height=4320) → HSWQ Save Image
```

Screenshot reference: `png/usdu_auto_workflow.png`, `png/saveimage.png` (README).

---

## 2. Files Created, Modified, and Related

### 2.1 Created (new source)

| Path | Role |
|------|------|
| `nodes/hswq_save_image.py` | HSWQ Save Image node (`NunchakuSaveImage` class, UI title **HSWQ Save Image**) |

### 2.2 Modified (this feature set)

| Path | Change |
|------|--------|
| `nodes/nunchaku_usdu.py` | Auto `upscale_by`, `target_height`, scale resolution in `upscale()` |
| `__init__.py` | Register `NunchakuSaveImage` from `hswq_save_image` |

### 2.3 Renamed (history)

| Before | After | Commit |
|--------|-------|--------|
| `nodes/nunchaku_save_image.py` | `nodes/hswq_save_image.py` | `13da833` |

### 2.4 Documentation / assets (user-facing, not runtime code)

| Path | Role |
|------|------|
| `README.md` | Sections **Upscale magnification** and **HSWQ Save Image** |
| `png/saveimage.png` | Save node screenshot |
| `png/usdu_auto_workflow.png` | Full workflow screenshot (Auto + target_height) |
| `png/usdu_auto_target.png` | Earlier README image variant |

### 2.5 Stale duplicate (not registered)

| Path | Note |
|------|------|
| `nodes/nunchaku_save_image.py` | Old filename; **not** imported in `__init__.py`. Contains pre-fix filename bug (`_{counter:05}_.{ext}`). **Canonical file is `hswq_save_image.py`.** |

---

## 3. Git Commit Timeline

| Commit | Subject | Files |
|--------|---------|-------|
| `023a1fe` | feat: add HSWQ Save Image node with PNG/JPG format selection | `nodes/nunchaku_save_image.py` (new), `__init__.py` |
| `13da833` | rename: nunchaku_save_image.py → hswq_save_image.py | rename + `__init__.py` import path |
| `b23a1d3` | fix: correct HSWQ Save Image filename so preview matches saved file | `hswq_save_image.py` |
| `c91a419` | fix: clarify quality is for JPG only in HSWQ Save Image | `hswq_save_image.py` (`**kwargs` for renamed input) |
| `2d7ec39` | fix: align HSWQ Save Image quality input label | input key `quality (JPG only)` |
| `d1deda0` | feat: add Auto upscale_by with target height for NunchakuUltimateSDUpscale | `nodes/nunchaku_usdu.py` |

---

## 4. Node Registration (`__init__.py`)

### 4.1 Added block (current)

```python
try:
    from .nodes.hswq_save_image import NunchakuSaveImage

    NODE_CLASS_MAPPINGS["NunchakuSaveImage"] = NunchakuSaveImage
    logger.info("Nunchaku Save Image node registered successfully")
except Exception as e:
    logger.error(f"Failed to register Nunchaku Save Image node: {e}", exc_info=True)
```

### 4.2 Meaning

- **ComfyUI node id**: `NunchakuSaveImage` (workflow JSON `class_type`).
- **Display title**: `HSWQ Save Image` (from class attribute `TITLE`).
- **Import path**: `nodes.hswq_save_image` (HSWQ naming per project convention).
- Wrapped in `try/except` so a save-node failure does not break the whole extension load (same pattern as `NunchakuUltimateSDUpscale` above it).

`NunchakuUltimateSDUpscale` registration is unchanged except it already existed:

```python
NODE_CLASS_MAPPINGS["NunchakuUltimateSDUpscale"] = NunchakuUltimateSDUpscale
```

---

## 5. HSWQ Save Image — Full Source and Explanation

### 5.1 Full file: `nodes/hswq_save_image.py`

```python
import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image, PngImagePlugin


class NunchakuSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "format": (["PNG", "JPG"], {"default": "PNG", "tooltip": "Output image format."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."}),
                "quality (JPG only)": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "JPEG quality (1-100). Only used when format is JPG; ignored for PNG."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory as PNG or JPG."
    TITLE = "HSWQ Save Image"

    def save_images(self, images, format, filename_prefix="ComfyUI", **kwargs):
        quality = kwargs.get("quality (JPG only)", 95)
        prompt = kwargs.get("prompt", None)
        extra_pnginfo = kwargs.get("extra_pnginfo", None)
        format = format.upper()
        if format not in ("PNG", "JPG"):
            raise ValueError(f"Unsupported format: {format}")

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = None
            if format == "PNG":
                metadata = PngImagePlugin.PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            ext = ".png" if format == "PNG" else ".jpg"
            file = f"{filename_with_batch_num}_{counter:05}{ext}"
            full_path = os.path.join(full_output_folder, file)

            if format == "PNG":
                img.save(full_path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                img.save(full_path, quality=quality, optimize=True)

            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}}
```

### 5.2 Design basis

- Modeled on ComfyUI core **Save Image** behavior:
  - `folder_paths.get_save_image_path()` for naming and counter
  - `OUTPUT_NODE = True`, `RETURN_TYPES = ()`
  - Hidden `prompt` / `extra_pnginfo` for PNG embedding
- **Extension**: explicit `format` combo **PNG | JPG** and JPEG `quality`.

### 5.3 Line-by-line / block explanation

| Lines / block | Meaning |
|---------------|---------|
| `import json, os, folder_paths, numpy, torch, PIL` | Standard Comfy save stack: paths, tensor→uint8, PIL encode. |
| `__init__` | `output_dir` from ComfyUI settings; `compress_level=4` for PNG (Pillow default-style). |
| `INPUT_TYPES` → `images` | Batch of `IMAGE` tensors `[B,H,W,C]` in 0–1 float. |
| `format` | User-selectable **PNG** or **JPG** (combo, not free string). |
| `filename_prefix` | Same as built-in Save Image; combined with date/counter by `get_save_image_path`. |
| `quality (JPG only)` | Input name includes scope in the label so users know PNG ignores it. |
| `hidden prompt / extra_pnginfo` | Injected by ComfyUI when running from UI/API; embedded into PNG tEXt chunks only. |
| `OUTPUT_NODE = True` | Terminal node; triggers file write and UI preview. |
| `TITLE = "HSWQ Save Image"` | Menu display name (HSWQ branding). |
| `save_images(..., **kwargs)` | Uses `**kwargs` because ComfyUI passes inputs by **display key name**. When the quality input was renamed to `quality (JPG only)`, a fixed `quality=95` parameter would break; `kwargs.get("quality (JPG only)", 95)` reads the correct socket. |
| `format.upper()` + validation | Normalizes combo value; rejects unknown formats. |
| `get_save_image_path(...)` | Returns folder, filename template, starting counter, subfolder — **width/height** from first image shape `[..., W, H]` → `shape[1]`, `shape[0]`. |
| Loop `enumerate(images)` | Supports batch dimension; `%batch_num%` in template replaced per index. |
| `255.0 * image.cpu().numpy()` | Comfy `IMAGE` is float 0–1; convert to 8-bit RGB. |
| PNG branch `PngInfo` | Writes workflow metadata for PNG-only (JPG has no equivalent here). |
| `file = f"..._{counter:05}{ext}"` | **Fix `b23a1d3`**: removed erroneous underscore before extension (`_00001_.png` → `_00001.png`) so UI preview path matches disk file. |
| `img.save(..., quality=..., optimize=True)` | JPG: Pillow quality 1–100; `optimize=True` for smaller files. |
| `return {"ui": {"images": results}}` | ComfyUI gallery preview list: `{filename, subfolder, type}`. |

### 5.4 Bug fix history (Save Image)

1. **Filename mismatch** (`b23a1d3`): Initial template used `_{counter:05}_.{ext}` (extra `_` before dot). Preview showed one name, disk had another.
2. **Quality input rename** (`c91a419`, `2d7ec39`): Label clarified as JPG-only; handler switched to `**kwargs` for ComfyUI dynamic input names.

---

## 6. Nunchaku Ultimate SD Upscale — Auto Mode Changes

### 6.1 What changed (concept)

**Before** (`upscale_by` as `FLOAT`):

- Single slider: default `2.0`, range `0.05`–`4.0`, step `0.05`.
- Passed directly to `StableDiffusionProcessing(..., upscale_by, ...)`.

**After**:

- Dropdown: **`Auto`** + fixed strings `"0.05"` … `"4.00"`.
- New integer: **`target_height`** (default **4320**).
- **`Auto`**: `scale = target_height / input_image_height`, then clamp to `[0.05, 4.0]`.
- **Numeric selection**: `scale = float(upscale_by)`; **`target_height` ignored**.
- `_to_fp32_image()` runs **before** reading height so PIL dimensions match normalized tensor.

### 6.2 Added function: `_upscale_by_options()`

```python
def _upscale_by_options():
    return ["Auto"] + [f"{round(i * 0.05, 2):.2f}" for i in range(1, 81)]
```

**Meaning**:

- `range(1, 81)` → `i = 1..80` → multipliers `0.05, 0.10, …, 4.00`.
- `round(i * 0.05, 2)` avoids float drift in combo labels.
- First entry **`Auto`** is the special mode token (string compare in `upscale()`).

### 6.3 `USDU_base_inputs()` — input definition changes

**Removed** (old):

```python
("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
```

**Added** (new):

```python
("upscale_by", (_upscale_by_options(), {"default": "Auto", "tooltip": "Choose 'Auto' to calculate the scale from target vertical pixels, or select a fixed magnification."})),
("target_height", ("INT", {"default": 4320, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Target output height in pixels. Used only when upscale_by is 'Auto'."})),
```

**Meaning**:

- ComfyUI combo type: list of strings (not `FLOAT`), so the node shows a dropdown.
- Default **Auto** + **4320** targets ~4K vertical output for common 1080p inputs.
- `MAX_RESOLUTION = 8192` caps `target_height` like other tile size inputs.
- `step: 8` aligns with USDU / latent alignment conventions.

The same pair appears in **`INPUT_TYPES` fallback** block (lines 190–191) so the node still registers if `usdu_bundle` import fails at startup.

### 6.4 `upscale()` — signature and scale resolution

**Added parameter**:

```python
def upscale(
    self,
    image,
    model,
    positive,
    negative,
    vae,
    upscale_by,
    target_height,   # NEW
    seed,
    ...
):
```

**Added logic at start of `upscale()`** (after `_ensure_imports()`):

```python
        # Normalize color range first to get correct input dimensions
        image = _to_fp32_image(image)
        init_img = tensor_to_pil(image, 0)

        # Resolve upscale_by: explicit numeric value takes precedence over Auto
        if upscale_by == "Auto":
            scale = float(target_height) / float(init_img.height)
        else:
            scale = float(upscale_by)
        scale = max(0.05, min(4.0, scale))
```

**Meaning**:

| Step | Why |
|------|-----|
| `_to_fp32_image(image)` first | Nunchaku SDXL VAE may output compressed range; normalization must happen before measuring height and before USDU shared batch. |
| `init_img = tensor_to_pil(image, 0)` | Reads **pixel height** from first batch image after normalization. |
| `Auto` branch | User sets desired **output height**; scale is derived. Example: 1080 → 4320 ⇒ scale 4.0. |
| `else` branch | User picked e.g. `"2.00"` from combo; `target_height` is unused. |
| `clamp 0.05..4.0` | Same effective limits as old FLOAT min/max; prevents extreme USDU / VRAM breakage. |

**Downstream replacements** (use computed `scale`, not raw `upscale_by`):

```python
        self.upscale_by = scale   # was: self.upscale_by = upscale_by

        sdprocessing = StableDiffusionProcessing(
            ...
            scale,                # was: upscale_by
            ...
        )

        _ = script.run(
            ...
            custom_scale=self.upscale_by,   # already used scale via self.upscale_by
        )
```

**Moved** `_to_fp32_image`: previously applied just before `shared.batch` assignment; now applied at the top so **height for Auto** and **batch tensors** share the same normalized image (no double-normalize).

### 6.5 Full diff reference (`git show d1deda0 -- nodes/nunchaku_usdu.py`)

```diff
@@ -68,6 +68,10 @@ def _ensure_imports():
 MAX_RESOLUTION = 8192
 
 
+def _upscale_by_options():
+    return ["Auto"] + [f"{round(i * 0.05, 2):.2f}" for i in range(1, 81)]
+
+
 def _to_fp32_image(image: torch.Tensor) -> torch.Tensor:
@@ -105,7 +109,8 @@ def USDU_base_inputs():
         ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
         ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
         ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
-        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
+        ("upscale_by", (_upscale_by_options(), {"default": "Auto", "tooltip": "Choose 'Auto' to calculate the scale from target vertical pixels, or select a fixed magnification."})),
+        ("target_height", ("INT", {"default": 4320, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Target output height in pixels. Used only when upscale_by is 'Auto'."})),
@@ -182,7 +187,8 @@ class NunchakuUltimateSDUpscale:
-                ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, ...})),
+                ("upscale_by", (_upscale_by_options(), {"default": "Auto", ...})),
+                ("target_height", ("INT", {"default": 4320, ...})),
@@ -223,6 +229,7 @@ class NunchakuUltimateSDUpscale:
         upscale_by,
+        target_height,
@@ -247,6 +254,18 @@ class NunchakuUltimateSDUpscale:
         _ensure_imports()
+
+        image = _to_fp32_image(image)
+        init_img = tensor_to_pil(image, 0)
+
+        if upscale_by == "Auto":
+            scale = float(target_height) / float(init_img.height)
+        else:
+            scale = float(upscale_by)
+        scale = max(0.05, min(4.0, scale))
+
-        self.upscale_by = upscale_by
+        self.upscale_by = scale
@@ -268,9 +287,6 @@ class NunchakuUltimateSDUpscale:
-        image = _to_fp32_image(image)
-
         shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
@@ -289,7 +305,7 @@ class NunchakuUltimateSDUpscale:
-            upscale_by,
+            scale,
```

### 6.6 Behavior matrix

| `upscale_by` | `target_height` | Effective scale | Output height (approx.) |
|--------------|-----------------|-----------------|-------------------------|
| `Auto` | 4320 | `4320 / H_in` (clamped) | 4320 (if scale ≤ 4.0) |
| `Auto` | 2160 | `2160 / H_in` | 2160 |
| `2.00` | (any) | `2.0` | `H_in * 2` |
| `0.50` | (any) | `0.5` | `H_in * 0.5` |

**Clamp example**: Input height 200, `target_height=4320` → raw scale 21.6 → clamped to **4.0** → output height 800.

### 6.7 Unchanged USDU pipeline (context)

After scale is resolved, the node still:

1. Sets `shared.sd_upscalers[0]` and `shared.actual_upscaler` from `upscale_model`.
2. Builds `shared.batch` / `shared.batch_as_tensor` from normalized `image`.
3. Creates `StableDiffusionProcessing` with tile/sampler/seam parameters.
4. Runs `usdu.Script().run(...)` with `custom_scale=self.upscale_by`.
5. Returns stacked `IMAGE` tensor.

Auto mode only changes **how `scale` is chosen** before this pipeline.

---

## 7. ComfyUI Node Identifiers

| UI title | `class_type` (workflow JSON) | Python class | Source file |
|----------|----------------------------|--------------|-------------|
| HSWQ&Nunchaku Ultimate SD Upscale | `NunchakuUltimateSDUpscale` | `NunchakuUltimateSDUpscale` | `nodes/nunchaku_usdu.py` |
| HSWQ Save Image | `NunchakuSaveImage` | `NunchakuSaveImage` | `nodes/hswq_save_image.py` |

---

## 8. Operational Notes

### 8.1 HSWQ Save Image

- Saves under ComfyUI **output** directory (Settings → output path).
- **PNG**: embeds prompt metadata when executed from ComfyUI with hidden inputs populated.
- **JPG**: no workflow metadata; uses `quality (JPG only)`.
- For lossless archival after upscale, use **PNG**; for smaller files, **JPG** at quality 90–95.

### 8.2 Auto upscale

- **Use Auto** when target output resolution is fixed (e.g. always 4320 px tall) regardless of input size.
- **Use fixed combo value** when you want a strict multiplier (e.g. always 2×) regardless of `target_height`.
- If upscale result is shorter than `target_height` after clamp, increase input resolution or lower `target_height`.

### 8.3 Migration from old workflows

- Old workflows with numeric `upscale_by` (float widget) must be re-wired: pick the closest combo string (e.g. `2.00`) or switch to **Auto** + set `target_height`.
- Old workflows using core **Save Image** can swap in **HSWQ Save Image** when JPG output or explicit format control is needed.

---

## 9. Related Existing Documentation

| Document | Topic |
|----------|-------|
| `README.md` | User-facing feature summary |
| `md/USDU_PATCH_FIX_DOCUMENTATION.md` | USDU bundle integration, `_to_fp32_image`, `batch_size` |
| `md/NUNCHAKU_USDU_FIX_REPORT.md` | `INPUT_TYPES` registration robustness |
| `md/reflection_2026_06_24.md` | Development notes (naming, revert history) |

---

## 10. Verification Checklist

- [ ] ComfyUI loads extension; log shows both nodes registered.
- [ ] **HSWQ Save Image**: PNG save includes file on disk and matching UI preview name.
- [ ] **HSWQ Save Image**: JPG save respects `quality (JPG only)`.
- [ ] **Auto**: 1080p input + `target_height=4320` → ~4320 px output height (scale 4.0).
- [ ] **Fixed**: `upscale_by=2.00` doubles height; changing `target_height` has no effect.
- [ ] **Clamp**: very small input + large `target_height` does not exceed 4.0× scale.

---

*End of implementation manual.*
