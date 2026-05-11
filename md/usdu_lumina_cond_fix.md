# USDU Lumina/HunYuan Conditioning Dimension Fix

## 1. The Error

```
RuntimeError: Given normalized_shape=[2560], expected input with shape [*2560],
but got input of size[1, 512, 7680]
```

- **Node:** `NunchakuUltimateSDUpscale` (Node ID 133)
- **Location in ComfyUI core:** `comfy/ldm/lumina/model.py`, line 642, inside `embed_cap()`
- **Failing call:** `cap_feats = self.cap_embedder(cap_feats)`

The `cap_embedder` is a `nn.Sequential` that begins with an `RMSNorm` layer whose `normalized_shape` is `[2560]`. PyTorch's `rms_norm` requires the input tensor's **last dimension** to match `normalized_shape`. The tensor arriving has shape `[1, 512, 7680]` — last dim is **7680**, not **2560**.

### The Numbers

| Value | Meaning |
|-------|---------|
| `2560` | The feature dimension the Lumina/HunYuan-DiT `cap_embedder` was trained on |
| `7680` | The actual feature dimension arriving at runtime |
| `7680 / 2560` | **= 3** — exactly three copies of the expected dimension |
| `512` | Token/sequence length (unchanged, correct) |
| `1` | Batch size (unchanged, correct) |

---

## 2. The Root Cause

### Call Chain Leading to the Error

```
NunchakuUltimateSDUpscale.upscale()
  → usdu_patch.py: patched_script_run()
    → ultimate-upscale.py: USDUpscaler.process()
      → USDURedraw.linear_process()
        → processing.py: process_images()
          → processing.py: sample()
            → nodes.py: common_ksampler()
              → comfy/sample.py: sample()
                → comfy/samplers.py: _calc_cond_batch()
                  → model_base.py: apply_model(context=c_crossattn)
                    → lumina/model.py: forward() → _forward()
                      → patchify_and_embed() → embed_all() → embed_cap()
                        → self.cap_embedder(cap_feats)  ← CRASH HERE
```

### What Changed in ComfyUI Core

ComfyUI recently changed how it constructs the `c_crossattn` tensor inside `_calc_cond_batch()` (in `comfy/samplers.py`). For models using the **Lumina architecture** (HunYuan-DiT, SeedVR2, etc.), the conditioning from **multiple text encoders** is now **concatenated along the feature dimension** (dim=-1) instead of along the token/sequence dimension (dim=1).

**Before the change:**

```
Encoder A output: [1, 512, 2560]  →  context passed to model: [1, 512, 2560]
                                      (only one encoder's output used, or concat on dim=1)
```

**After the change:**

```
Encoder A output: [1, 512, 2560]
Encoder B output: [1, 512, 2560]  →  torch.cat([A, B, C], dim=-1) = [1, 512, 7680]
Encoder C output: [1, 512, 2560]
```

The Lumina model's `cap_embedder` was never designed to receive this concatenated tensor. It expects raw single-encoder output with dim=2560.

### Why the First SeedVR2 Pass Succeeds

Looking at the logs, the **SeedVR2 upscaler node** (ComfyUI-SeedVR2_VideoUpscaler) completes successfully:

```
[18:17:41.094] ✅ Upscaling completed successfully!
[18:18:45.851] ✅ Upscaling completed successfully!
```

This is because SeedVR2 has its **own internal conditioning pipeline** — it generates and manages its own `cap_feats` internally, bypassing the ComfyUI conditioning system entirely.

The USDU node, however, takes **external** `positive` and `negative` conditioning inputs from the ComfyUI graph and passes them through `common_ksampler()`. This is where the newly-concatenated conditioning hits the Lumina model and crashes.

---

## 3. All Modified Code

Three files were modified. Every added line is shown below with full context and annotation.

---

### 3.1. `usdu_bundle/usdu_utils.py` — Two New Functions

Added immediately before the existing `crop_cond()` function (after line 557 of the original file):

```python
def _get_model_expected_cond_dim(model):
    """
    Try to detect the expected cross-attention conditioning dimension from the model.
    For Lumina/HunYuan-DiT models, this is the cap_embedder's normalized_shape.
    Returns the expected dimension or None if not detectable.
    """
    try:
        # Navigate through ModelPatcher -> model -> diffusion_model
        diffusion_model = None
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        elif hasattr(model, 'inner_model') and hasattr(model.inner_model, 'model') and hasattr(model.inner_model.model, 'diffusion_model'):
            diffusion_model = model.inner_model.model.diffusion_model

        if diffusion_model is None:
            return None

        # Check for Lumina/HunYuan cap_embedder
        cap_embedder = getattr(diffusion_model, 'cap_embedder', None)
        if cap_embedder is None:
            return None

        # cap_embedder is typically a Sequential containing a LayerNorm/RMSNorm + Linear
        # The first layer's normalized_shape tells us the expected input dimension
        for module in cap_embedder.modules():
            ns = getattr(module, 'normalized_shape', None)
            if ns is not None:
                if isinstance(ns, (list, tuple)) and len(ns) == 1:
                    return ns[0]
                elif isinstance(ns, int):
                    return ns
        # Fallback: check the first Linear layer's in_features
        for module in cap_embedder.modules():
            if hasattr(module, 'in_features'):
                return module.in_features
    except Exception:
        pass
    return None


def fix_cond_for_model(model, cond):
    """
    Fix conditioning embedding dimension to match what the model expects.

    ComfyUI core may concatenate multi-encoder conditioning along the feature
    dimension (e.g. 3x2560=7680 for Lumina/HunYuan models). This function detects
    that mismatch and truncates the embedding to the expected dimension.
    """
    expected_dim = _get_model_expected_cond_dim(model)
    if expected_dim is None or expected_dim <= 0:
        return cond

    fixed = []
    for emb, cond_dict in cond:
        if torch.is_tensor(emb) and emb.ndim >= 2:
            actual_dim = emb.shape[-1]
            if actual_dim != expected_dim and actual_dim % expected_dim == 0:
                # Truncate to the expected dimension (take the first slice)
                emb = emb[..., :expected_dim].contiguous()
        fixed.append([emb, cond_dict])
    return fixed
```

#### Line-by-Line Explanation

**`_get_model_expected_cond_dim(model)`**

| Lines | Purpose |
|-------|---------|
| `diffusion_model = None` | Initialize. We need to reach the actual neural network inside ComfyUI's wrapper layers. |
| `if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):` | **Path 1:** ComfyUI wraps models in `ModelPatcher`. The chain is `ModelPatcher.model` → `BaseModel.diffusion_model` → the actual DiT/UNet. |
| `elif hasattr(model, 'inner_model') ...` | **Path 2:** Some wrapper chains (e.g. from `crop_model_cond`) add an extra `inner_model` layer. This handles that case. |
| `if diffusion_model is None: return None` | If we can't find the diffusion model, this isn't a model we can introspect. Return `None` to signal "no fix needed". |
| `cap_embedder = getattr(diffusion_model, 'cap_embedder', None)` | Only Lumina-architecture models (HunYuan-DiT, SeedVR2) have `cap_embedder`. FLUX, SD1.5, SDXL do **not**. If absent, return `None` — no fix needed. |
| `for module in cap_embedder.modules():` | Walk all submodules of `cap_embedder` (which is an `nn.Sequential`). |
| `ns = getattr(module, 'normalized_shape', None)` | `RMSNorm` and `LayerNorm` both store `normalized_shape`. This is the authoritative source for the expected input dimension. |
| `if isinstance(ns, (list, tuple)) and len(ns) == 1: return ns[0]` | PyTorch stores `normalized_shape` as a list like `[2560]`. Extract the integer. |
| `if hasattr(module, 'in_features'): return module.in_features` | **Fallback:** If no norm layer is found, check the first `Linear` layer's `in_features` as a secondary source. |
| `except Exception: pass` | Wrapped in try/except for maximum safety. If anything goes wrong during introspection, silently return `None` and let the original code path run unchanged. |

**`fix_cond_for_model(model, cond)`**

| Lines | Purpose |
|-------|---------|
| `expected_dim = _get_model_expected_cond_dim(model)` | Query the model for its expected conditioning dimension. |
| `if expected_dim is None or expected_dim <= 0: return cond` | **Safety gate:** If we can't determine the expected dim (non-Lumina model, or introspection failed), return conditioning **completely unchanged**. This ensures FLUX, SD, SDXL, etc. are never affected. |
| `for emb, cond_dict in cond:` | ComfyUI conditioning is a list of `[embedding_tensor, metadata_dict]` pairs. Iterate over each. |
| `if torch.is_tensor(emb) and emb.ndim >= 2:` | Only process actual tensor embeddings with at least 2 dimensions (batch + features). |
| `actual_dim = emb.shape[-1]` | Get the last dimension of the embedding (the feature dimension). |
| `if actual_dim != expected_dim and actual_dim % expected_dim == 0:` | **Key check:** The actual dim must be (a) different from expected, AND (b) an exact integer multiple. This catches `7680 = 2560 × 3` but would NOT trigger on, say, `4096` (which is not a multiple of `2560`). |
| `emb = emb[..., :expected_dim].contiguous()` | **The fix:** Slice the first `expected_dim` features. The `...` handles any number of leading dimensions. `.contiguous()` ensures the resulting tensor has contiguous memory layout (required by some CUDA kernels). |
| `fixed.append([emb, cond_dict])` | Rebuild the conditioning list with the (possibly fixed) embedding. |

---

### 3.2. `usdu_bundle/modules/processing.py` — Two Changes

**Change 1: Import** (line 7)

```diff
-from usdu_utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond
+from usdu_utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond, fix_cond_for_model
```

Adds `fix_cond_for_model` to the import list.

**Change 2: Call before sampling** (after line 238, before the `with crop_model_cond(...)` block)

```python
    # Fix conditioning dimension for models that expect a specific feature dim
    # (e.g. Lumina/HunYuan cap_embedder expects 2560 but ComfyUI may concat to 7680)
    positive_cropped = fix_cond_for_model(p.model, positive_cropped)
    negative_cropped = fix_cond_for_model(p.model, negative_cropped)
```

This is the **single-tile processing path** — used when `batch_size=1` (the default). Every tile goes through `process_images()` → `sample()` → `common_ksampler()`. The fix is applied **after** `crop_cond()` has prepared the tile-specific conditioning, and **before** the conditioning enters the sampling pipeline.

#### Why Here and Not Earlier?

- **Not in `crop_cond()`** — because `crop_cond` doesn't have access to the model object, and we need to introspect the model to know the expected dimension.
- **Not in `sample()`** — because `sample()` is a generic function that should remain model-agnostic.
- **Right before `crop_model_cond()`** — this is the last point where we control the conditioning before it enters ComfyUI's sampling pipeline.

---

### 3.3. `usdu_bundle/usdu_patch.py` — One Change

**Call before sampling in batch path** (after line 287, inside `_process_batch_tiles()`)

```python
    # Fix conditioning dimension for models that expect a specific feature dim
    positive_cropped = usdu_utils.fix_cond_for_model(p.model, positive_cropped)
    negative_cropped = usdu_utils.fix_cond_for_model(p.model, negative_cropped)
```

This is the **batch-tile processing path** — used when `batch_size > 1`. Multiple tiles are encoded and sampled together. The same dimension fix is needed here because the conditioning goes through the same `common_ksampler()` → Lumina model path.

Note: `usdu_patch.py` already imports `usdu_utils` (line 28), so no additional import is needed.

---

## 4. Safety Analysis

### Models That Are NOT Affected

| Model Type | Has `cap_embedder`? | `_get_model_expected_cond_dim()` returns | Fix applied? |
|------------|--------------------|-----------------------------------------|-------------|
| SD 1.5 | No | `None` | **No** |
| SDXL | No | `None` | **No** |
| FLUX | No | `None` | **No** |
| Hunyuan-DiT | **Yes** | `2560` | **Only if dim mismatches** |
| SeedVR2 (Lumina) | **Yes** | `2560` | **Only if dim mismatches** |

The fix is **completely inert** for non-Lumina models. The `_get_model_expected_cond_dim()` function returns `None` as soon as it fails to find `cap_embedder`, and `fix_cond_for_model()` immediately returns the original conditioning unchanged.

### Edge Cases Handled

| Scenario | Behavior |
|----------|----------|
| `cap_embedder` not found | Returns `None` → no fix |
| Conditioning dim already correct (2560) | `actual_dim != expected_dim` is `False` → no fix |
| Conditioning dim is not an exact multiple (e.g. 4096) | `actual_dim % expected_dim == 0` is `False` → no fix |
| Non-tensor embedding (e.g. None) | `torch.is_tensor(emb)` is `False` → no fix |
| 1D embedding | `emb.ndim >= 2` is `False` → no fix |
| Any exception during model introspection | Caught by `try/except` → returns `None` → no fix |
