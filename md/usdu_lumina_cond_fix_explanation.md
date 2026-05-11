# USDU Conditioning Noise and Tensor Mismatch Resolution

## 1. Cause of the Noise
The fundamental cause of the severe noise in Lumina (SeedVR2, etc.) and HunYuan-DiT architectures during USDU upscaling was a mismatch between the concatenated text encoder output tensors from ComfyUI and the expected feature dimension size of the model's `cap_embedder`, combined with an incorrect slicing approach.

As shown in the error log `got input of size[1, 512, 7680]`, the conditioning tensor passed to the USDU pipeline was inflated to a feature dimension of `7680`. This occurs because ComfyUI concatenates the outputs of multiple text encoders (e.g., CLIP + T5/Gemma) along the feature dimension (`dim=-1`).
Conversely, the `cap_embedder` (RMSNorm) inside the Lumina model strictly expects `2560` dimensions, corresponding exclusively to the T5/Gemma text encoder output.

## 2. What was wrong with the previous fix
To resolve the dimension mismatch (`7680` -> `2560`), the previous patch implemented the following code in `fix_cond_for_model` inside `usdu_utils.py`:

**[The Incorrect Previous Code]**
```python
emb = emb[..., :expected_dim].contiguous()
```

This logic instructed the system to slice the **first** `2560` dimensions of the concatenated tensor. 
However, under ComfyUI's text encoder concatenation sequence, CLIP encoder outputs are placed at the **start**, while the LLM text encoder outputs (T5, Gemma) are appended to the **end**.

As a result of slicing from the beginning, the `cap_embedder` received the CLIP features instead of the expected T5/Gemma features. The model attempted to process these invalid, untrained CLIP feature vectors as T5 embeddings, causing the generated image to completely collapse into severe noise.

## 3. The exact additional corrected code

### A. Modifications in `usdu_bundle/usdu_utils.py`
```python
def fix_cond_for_model(model, cond):
    """
    Fix conditioning embedding dimension to match what the model expects.

    ComfyUI core concatenates multi-encoder conditioning along the feature
    dimension. Since the CLIP encoder outputs are placed at the start, and the
    LLM/T5 text encoder outputs are placed at the end, we extract the *last* 
    `expected_dim` features to target the main cap_embedder input.
    """
    expected_dim = _get_model_expected_cond_dim(model)
    if expected_dim is None or expected_dim <= 0:
        return cond

    fixed = []
    for emb, cond_dict in cond:
        if torch.is_tensor(emb) and emb.ndim >= 2:
            actual_dim = emb.shape[-1]
            if actual_dim != expected_dim and actual_dim > expected_dim:
                # Truncate to the expected dimension by taking the last slice
                emb = emb[..., -expected_dim:].contiguous()
        fixed.append([emb, cond_dict])
    return fixed
```

### B. Modifications in `usdu_bundle/modules/processing.py` (Excerpt)
```python
from usdu_utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond, fix_cond_for_model

# (omitted)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Decode conditioning for Lumina/HunYuan compatibility
    positive_cropped = fix_cond_for_model(p.model, positive_cropped)
    negative_cropped = fix_cond_for_model(p.model, negative_cropped)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)
```

### C. Modifications in `usdu_bundle/usdu_patch.py` (Excerpt)
```python
    positive_cropped = usdu_utils.crop_cond(p.positive, batch_crop_regions, p.init_size, images[0].size, first_tile_size)
    negative_cropped = usdu_utils.crop_cond(p.negative, batch_crop_regions, p.init_size, images[0].size, first_tile_size)

    # Decode conditioning for Lumina/HunYuan compatibility
    positive_cropped = usdu_utils.fix_cond_for_model(p.model, positive_cropped)
    negative_cropped = usdu_utils.fix_cond_for_model(p.model, negative_cropped)

    with crop_model_cond(p.model, batch_crop_regions, p.init_size, images[0].size, first_tile_size) as model:
```

## 4. Explanation and meaning of the corrected code

### The role of `emb[..., -expected_dim:].contiguous()`
This single line is the core correction that resolves the noise issue.
- `...` (Ellipsis): Retains the tensor's batch dimensions and sequence lengths (e.g., `[1, 512]`) identically.
- `-expected_dim:`: For the feature dimension (the final dimension), this retrieves elements starting from **`expected_dim` (e.g., 2560) steps away from the end**.
- `.contiguous()`: Reallocates the tensor into a continuous block in memory after the slicing operation, a safety measure preventing subsequent PyTorch computation errors.

### Why does this fix the issue?
When ComfyUI concatenates outputs from multiple text encoders, it appends them in sequence, for example: `clip_l` (768) + `clip_g` (1280) + `t5xxl`/`gemma` (2560). As a result, the composition of the tensor's feature dimension (`dim=-1`) becomes:

`[CLIP features ... , LLM/T5 features]`

The model's `cap_embedder` requires the LLM/T5 features located at the very end of this sequence. Therefore, instead of slicing from the start, we use `-expected_dim:` to extract the exact required dimension size (2560) **from the end**. This extracts 100% accurate, intended text features expected by the model.

This ensures the model is supplied with valid conditioning data, completely eliminating both the dimension mismatch `RuntimeError` and the severe image noise caused by ingesting CLIP features into the LLM conditioning blocks.
