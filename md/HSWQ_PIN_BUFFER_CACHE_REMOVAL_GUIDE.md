# HSWQ Pin Buffer Cache Removal — Complete Technical Guide

**Date:** 2026-07-16  
**Repository:** `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools`  
**Removal commit:** `5d37ccfb03340c4cd1a84075ab78ce2fad452985`  
**Parent (pre-removal tip):** `9fdfae3`  
**Primary historical release:** [v3.1.0 — Pin Memory Problem and HSWQ Optimization](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.0)  
**ComfyUI tree used for verification:** `D:\USERFILES\ComfyUI\ComfyUI` (`comfyanonymous/ComfyUI`, tag/HEAD around **v0.27.1** / `c2638ce6c`)

This document explains **why Pin Buffer Cache was introduced**, **what ComfyUI changed**, **why the cache is no longer needed**, **which files were modified**, **the full code that changed** (including the deleted module), and **the meaning of each change**.

---

## Table of contents

1. [Circumstances at introduction](#1-circumstances-at-introduction)
2. [What ComfyUI updates changed](#2-what-comfyui-updates-changed)
3. [Why the cache became unnecessary](#3-why-the-cache-became-unnecessary)
4. [Modified / deleted file names](#4-modified--deleted-file-names)
5. [Full text of modified and deleted code](#5-full-text-of-modified-and-deleted-code)
6. [Meaning of each change](#6-meaning-of-each-change)
7. [Appendix A — Deleted `nodes/hswq_pin_cache.py` (full)](#appendix-a--deleted-nodeshswq_pin_cachepy-full)

---

## 1. Circumstances at introduction

### 1.1 Product goal (v3.1.0)

Around **2026-03-14**, release **v3.1.0** documented a dual countermeasure against stalls under ComfyUI’s then-new **Dynamic VRAM Loading**:

| Countermeasure | Role |
|----------------|------|
| **Pin Buffer Cache** | Monkey-patch `comfy.pinned_memory.pin_memory` / `unpin_memory` so pin buffers are **pooled by size** instead of destroyed on every model unload |
| **HSWQ Batched Detailer (SEGS)** | Reorder Detailer work into **one VAE encode phase → one UNet sample phase → one VAE decode phase**, cutting model switches from **O(3n)** to **O(2)** |

ComfyUI core was **not** modified. All logic lived in this extension.

### 1.2 How Dynamic VRAM + pinned memory worked *then*

From the v3.1.0 release body (paraphrased against the old API):

```text
Traditional: entire model on GPU
Dynamic:     each Linear weight streamed CPU → GPU on demand
```

Pinned host RAM enables DMA-style CPU→GPU copies. ComfyUI called `cudaHostRegister` per layer via `model_management.pin_memory(tensor)`.

On unload, `ModelPatcherDynamic.partially_unload_ram` called **`unpin_memory(module)` for every layer**, which:

1. Called `cudaHostUnregister`
2. **`del module._pin`** — the buffer was **gone**

On the next use of that model, ops called `pin_memory(module)` again and **allocated a fresh `torch.empty` of the same size + new Register**.

Cost (v3.1.0 notes):

- Page-table updates, GPU MMU updates, CPU–GPU sync
- ~200 Linear layers (e.g. Lumina2) ⇒ hundreds of Register/Unregister **per model switch**

### 1.3 Why FaceDetailer made it catastrophic

Per face segment, classic Detailer did:

```text
VAE encode → UNet sample → VAE decode
```

**n** faces ⇒ **3n** model switches ⇒ **3n** full pin tear-downs / rebuilds.

Observed symptoms (v3.1.0):

- Extreme slowdown with CUDAGraphs + FaceDetailer
- Logs flooded with pin debug / EVICT traffic
- Bottleneck framed as **pin/unpin thrash**, not “too many CUDA kernels”

### 1.4 What Pin Buffer Cache did (extension design)

Idea: if the same layer sizes return repeatedly, **keep pin tensors in a size-keyed pool** (cap e.g. 16 GB), HIT on re-pin → **skip `cudaHostRegister`**.

Lifecycle of the feature in *this* repo:

| Version | Behavior |
|---------|----------|
| **3.1.0** | Cache installed with the HSWQ UNet loader / extension load; documented as extension-wide |
| **3.1.2** | Cache enabled **only while** `HSWQ Batched Detailer (SEGS)` runs — avoid side effects elsewhere |
| Later (hostbuf era) | Old `pin_memory(module)` / `unpin_memory(module)` wrappers **broke** current ComfyUI (`subset=`, `size=`) and poisoned Nunchaku / Z-Image loads (`0.00 MB usable`) |
| Interim rewrite | `nodes/hswq_pin_cache.py` — Detailer-only scope via `hswq_pin_cache_scope()`, patched hostbuf-era APIs with freestanding pool + soft unload |
| **2026-07-16** | Entire Pin Buffer Cache **removed** (`5d37ccf`) |

**Batched Detailer remains.** Only the pin *cache* was removed.

### 1.5 What was *not* Pin Buffer Cache

Z-Image / ZIT DiT loader UI option `use_pin_memory` is **Nunchaku offload configuration**, unrelated to `hswq_pin_cache`. It was **not** removed.

---

## 2. What ComfyUI updates changed

Verification tree: `D:\USERFILES\ComfyUI\ComfyUI` (upstream ComfyUI **v0.27.1**-class).

### 2.1 Decisive upstream direction

Commit **`5aa5ccc9e`** (*Multi-threaded load… / Aimdo*, PR **#13802**), among other changes:

> `pinned_memory: implement with aimdo growable buffer`  
> `Use a single growable buffer so we can do threaded pre-warming on pinned memory.`  
> Stream host pin buffer … **allocated monolithically (to avoid cudaHostRegister thrash).**  
> **`remove old pin path`**

Later pinned_memory commits refine AIMDO versions and pin pressure (`e154da83b`, `410df2725`, …). The architectural break is the **HostBuffer / remove old pin path** step.

### 2.2 New pin architecture (current `comfy/pinned_memory.py`)

| Old (v3.1.0-era narrative) | Current ComfyUI |
|----------------------------|-----------------|
| Public `pin_memory(module)` | Public `pin_memory(module, subset="weights", size=None)` |
| Public `unpin_memory(module)` | **Removed** from `comfy.pinned_memory` |
| Per-module freestanding `_pin` tensors as the main model | Shared **`comfy_aimdo.host_buffer.HostBuffer`** arena per model Dynamic pin state |
| Unload = unregister + destroy each buffer | Unload = `hostbuf.truncate(offset, do_unregister=…)` (+ registration accounting) |
| No cross-layer steal in extension docs | Core **`_steal_pin`** + priority **buckets** for same-size reuse under budget |
| Extension pools freestanding tensors | Core **`ensure_pin_budget` / `free_registrations` / `unregister_inactive_pins`** (JIT pin pressure) |

`ModelPatcherDynamic` initializes:

```python
pin_state["weights"] = (HostBuffer(...), [], [-1], [0], [0], {})
pin_state["patches"] = (HostBuffer(...), [], [-1], [0], [0], {})
pin_state["hostbufs_initialized"] = True
```

`pin_memory` extends the HostBuffer, registers a **slice** view, and records `(module, offset)` on a stack.

`partially_unload_ram` pops the stack, drops `module._pin`, and **truncates** the HostBuffer to that offset (AIMDO may unregister in that path).

### 2.3 AIMDO `HostBuffer` (runtime package)

Installed module (example path on the verification machine):

`...\python_embeded\Lib\site-packages\comfy_aimdo\host_buffer.py`

Relevant methods:

- `extend(size, reallocate=False, register=True)` — grow arena  
- `truncate(size, do_unregister=True)` — shrink arena  

This is the **growable pinned arena** upstream built so the extension does not need to invent a second pool of freestanding pinned tensors.

### 2.4 What ComfyUI did *not* remove

Frequent **model switching** (VAE ↔ UNet many times) still costs Dynamic VRAM prepare/unload work. That is why **HSWQ Batched Detailer** stays valuable. Core’s HostBuffer solves **Register thrash of the old per-buffer tear-down model**, not “FaceDetailer calls VAE three times per face.”

---

## 3. Why the cache became unnecessary

Map each PinCache job to current core ownership:

| Extension PinCache responsibility (historical) | Now owned by ComfyUI core? | Verdict |
|-----------------------------------------------|----------------------------|---------|
| Avoid destroying pin buffers on every unload | HostBuffer truncate/extend + registration policy | **Yes — redundant** |
| Reuse same nbytes without new `cudaHostRegister` | Growable buffer, steal buckets, monolithic stream pins, explicit “avoid thrash” in #13802 | **Yes — redundant** |
| Monkey-patch `unpin_memory(module)` | API **gone**; unload path is `partially_unload_ram` + HostBuffer | **Patch target invalid / harmful** |
| Detailer-only gating to protect Nunchaku | Native pin path is already Nunchaku-safe if **not** monkey-patched | **Band-aid obsolete** |
| Reduce FaceDetailer **switch count** | Still **not** a HostBuffer job | **Batched Detailer remains; PinCache not a substitute** |

**One-sentence conclusion**

> v3.1.0 Pin Buffer Cache was an external band-aid for “destroy every layer pin then recreate with `cudaHostRegister`.” Current ComfyUI replaces that lifecycle with AIMDO HostBuffers, steal, and pin-pressure management, so the extension cache has **no remaining role** and its patches **fight** the new API.

Hence removal commit `5d37ccf`.

---

## 4. Modified / deleted file names

Commit `5d37ccf` (`9fdfae3..5d37ccf`):

| Path | Action |
|------|--------|
| `nodes/hswq_pin_cache.py` | **Deleted** (494 lines) |
| `nodes/hswq_batched_detailer.py` | Modified — remove `hswq_pin_cache_scope`; keep Retry path |
| `__init__.py` | Modified — drop PinCache / PinDebug comments |
| `README.md` | Modified — remove Pin Buffer Cache product copy; soften Detailer wording |
| `changelog.md` | Modified — **Removed** bullet under Version 3.1.9 |
| `md/HSWQ_INT8_AND_LORA_TECHNICAL_GUIDE.md` | Modified — strip embedded historical PinCache dump; leave “removed” note |

**Unchanged (intentionally):**

- `nodes/models/zimage.py` / `zimage_turbo.py` — `use_pin_memory` (Nunchaku)
- Batched Detailer’s three-phase `do_detail_batched` algorithm

---

## 5. Full text of modified and deleted code

### 5.0 How to read this section

- **§5.1–§5.5** = **post-removal** regions as they exist after `5d37ccf` (the living tree).  
- **Appendix A** = **complete deleted** `nodes/hswq_pin_cache.py` from `9fdfae3` (what was removed).  
- For the short changelog / guide note, the entire relevant block is shown.

### 5.1 `nodes/hswq_batched_detailer.py` — DESCRIPTION (after)

```python
    DESCRIPTION = (
        "Phase-split version of Detailer (SEGS). Groups all VAE encodes, "
        "all KSampler calls, and all VAE decodes into separate phases to "
        "minimize model switching under Dynamic VRAM Loading."
    )
```

### 5.2 `nodes/hswq_batched_detailer.py` — `doit` body (after; full method tail)

```python
    ):
        try:
            enhanced_img, *_ = HSWQBatchedDetailer.do_detail_batched(
                image, segs, model, clip, vae,
                guide_size, guide_size_for, max_size, seed, steps, cfg,
                sampler_name, scheduler, positive, negative, denoise, feather,
                noise_mask, force_inpaint, wildcard, detailer_hook,
                cycle=cycle, inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                scheduler_func_opt=scheduler_func_opt,
                tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                force_fixed_latent_size=False,
            )
        except RuntimeError as e:
            err_msg = str(e)
            if "QuantizedTensor" in err_msg and (
                "copy_" in err_msg or "size mismatch" in err_msg
            ):
                logging.warning(
                    "[HSWQ] BatchedDetailer: QuantizedTensor copy_ mismatch. "
                    "Retrying with fixed latent size for all segments: %s",
                    err_msg[:200],
                )
                enhanced_img, *_ = HSWQBatchedDetailer.do_detail_batched(
                    image, segs, model, clip, vae,
                    guide_size, guide_size_for, max_size, seed, steps, cfg,
                    sampler_name, scheduler, positive, negative, denoise, feather,
                    noise_mask, force_inpaint, wildcard, detailer_hook,
                    cycle=cycle, inpaint_model=inpaint_model,
                    noise_mask_feather=noise_mask_feather,
                    scheduler_func_opt=scheduler_func_opt,
                    tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                    force_fixed_latent_size=True,
                )
            else:
                raise

        return (enhanced_img,)
```

**Before (removed wrapper only):**

```python
        from .hswq_pin_cache import hswq_pin_cache_scope

        with hswq_pin_cache_scope():
            try:
                enhanced_img, *_ = HSWQBatchedDetailer.do_detail_batched(...)
            except RuntimeError as e:
                ...
        return (enhanced_img,)
```

### 5.3 `__init__.py` — comment block (after)

```python
from .utils import get_package_version, get_plugin_version

# INT8 comfy_quant patches are NOT applied at import — only when an
# INT8 HSWQ / SDXL load path calls apply_comfy_quant_int8_patches().

# HSWQ&Nunchaku Ultimate SD Upscale: apply copy_ / FP8 bias / embedder / Lumina compat patches in this extension
```

**Before (removed PinCache / PinDebug warning block):**

```python
# PinCache / PinDebug must NOT install at import (or anywhere globally).
# Old pin_memory(module)/unpin_memory wrappers break current ComfyUI
# pinned_memory.pin_memory(module, subset=, size=) (hostbuf Dynamic VRAM)
# and poisoned Nunchaku / Z-Image / Lumina2 loads (0.00 MB usable → CUDA abort).
# Detailer-only rewrite: nodes/hswq_pin_cache.py (activate inside
# HSWQBatchedDetailer.doit via hswq_pin_cache_scope). Native ComfyUI pin
# elsewhere.
# INT8 comfy_quant patches are NOT applied at import either — only when an
# INT8 HSWQ / SDXL load path calls apply_comfy_quant_int8_patches().
```

### 5.4 `README.md` — UNet + Detailer prose (after, relevant excerpts)

```markdown
### HSWQ FP8 E4M3 UNet Loader

Standard ComfyUI UNet loader wrapper that loads FP8 and INT8 diffusion models (**general FP8 and INT8**, not limited to HSWQ-only weights). Loads the UNet (MODEL) from FP8 / INT8 checkpoints like the standard UNet loader (HSWQ FP8 E4M3, Scaled FP8, and native comfy_quant / `int8_tensorwise` when selected or auto-detected).

### HSWQ Batched Detailer (SEGS)
...
So the pipeline does: VAE load → UNet load → VAE load → UNet load → … With many segments this causes repeated model switches and Dynamic VRAM reloads, leading to long stalls (especially with CUDAGraphs).
```

(Entire “Why the Pin Buffer Cache matters” / “Cache (extension-wide)” subsection deleted.)

### 5.5 `changelog.md` — Version 3.1.9 head (after addition)

```markdown
## Version 3.1.9

- **Removed**: **HSWQ Pin Buffer Cache** (`nodes/hswq_pin_cache.py` and Detailer `hswq_pin_cache_scope`) — no longer needed after ComfyUI Dynamic VRAM / hostbuf updates. Use native ComfyUI pin behavior.
- **Added**: Native **comfy_quant INT8** (`int8_tensorwise`) load path for SDXL checkpoints — **HSWQ FP8/INT8 Loader (VRAM Opt)** auto-detects INT8 vs Scaled FP8; **HSWQ FP8 E4M3 UNet Loader** gains `int8_tensorwise` / auto-detect. Extension-side Conv2d quant support and INT8-safe LoRA bake under Dynamic VRAM.
- See [Release Notes v3.1.9](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.9) for details.
```

Note: Historical entries for **3.1.0** / **3.1.2** still describe the *past* addition and Detailer-scoping of PinCache. Those remain as changelog history.

### 5.6 `md/HSWQ_INT8_AND_LORA_TECHNICAL_GUIDE.md` — replacement note (after)

```markdown
# HSWQ Pin Buffer Cache: removed. Current ComfyUI Dynamic VRAM / hostbuf
# pin path no longer needs this extension's monkey-patch.
```

(Approximately **180 lines** of embedded historical PinCache / PinDebug source in that guide were deleted.)

### 5.7 Deleted module

See **[Appendix A](#appendix-a--deleted-nodeshswq_pin_cachepy-full)** for the **complete** pre-removal source of `nodes/hswq_pin_cache.py`.

---

## 6. Meaning of each change

### 6.1 Delete `nodes/hswq_pin_cache.py`

| Aspect | Meaning |
|--------|---------|
| **What** | Removes freestanding pool, soft `partially_unload_ram`, Detailer `activate`/`deactivate`/`hswq_pin_cache_scope`, and patches to `pinned_memory.pin_memory` / `ModelPatcherDynamic.partially_unload_ram`. |
| **Why** | Upstream HostBuffer + steal + pin pressure already address Register thrash; keeping a second pool doubles lifecycle and fights `truncate`/`extend` semantics. |
| **Risk if kept** | Wrong offsets (`offset < 0` pooled pins), double accounting of `TOTAL_PINNED_MEMORY`, Nunchaku / Dynamic VRAM breakage when signatures diverge again. |

### 6.2 Unwrap `HSWQBatchedDetailer.doit`

| Aspect | Meaning |
|--------|---------|
| **What** | Detailer again runs `do_detail_batched` under **native** ComfyUI pin behavior only. |
| **What stayed** | QuantizedTensor `copy_` mismatch **retry** with `force_fixed_latent_size=True`. |
| **Why** | Pin cache gating had no remaining payoff; retry is an orthogonal INT8/quant correctness fix. |
| **DESCRIPTION** | Drops “pin_memory churn”; states the real remaining goal: **minimize model switching**. |

### 6.3 `__init__.py` comment cleanup

| Aspect | Meaning |
|--------|---------|
| **What** | Removes dead documentation of PinCache / PinDebug install rules. |
| **Why** | No file remains to activate; comments would advertise a lie. |
| **Kept** | INT8 patches still **must not** install at import — that discipline is independent of PinCache. |

### 6.4 `README.md` product copy

| Aspect | Meaning |
|--------|---------|
| **What** | Stops advertising Pin Buffer Cache on UNet Loader; Detailer blurb talks about **model switches**, not `cudaHostRegister` spam. |
| **Why** | Public docs must match runtime: core owns pinning; this extension still helps via **batched phases**. |

### 6.5 `changelog.md` Removed bullet

| Aspect | Meaning |
|--------|---------|
| **What** | Records removal under **3.1.9** for auditors and users who search history. |
| **Why** | Release surfaces need an explicit “feature gone” marker even when tag bodies are edited later. |

### 6.6 Technical guide dump strip

| Aspect | Meaning |
|--------|---------|
| **What** | Deletes a large **pasted outdated** PinCache implementation that could be mistaken for live `__init__.py` policy. |
| **Why** | Docs that embed dead monkey-patches cause the next agent/human to “restore” harmful code. |

### 6.7 What this removal does *not* mean

- It does **not** mean Dynamic VRAM is free.  
- It does **not** deprecate **HSWQ Batched Detailer**.  
- It does **not** change Nunchaku `use_pin_memory`.  
- It does **not** claim FaceDetailer-style pipelines need no care — **phase batching** remains the extension’s answer to switch thrash.

---

## Cross-reference — problem → responsibility matrix

```text
cudaHostRegister thrash on per-buffer destroy/recreate
    THEN: extension Pin Buffer Cache
    NOW:  ComfyUI AIMDO HostBuffer + steal + pin pressure   → REMOVE extension cache

3n VAE/UNet/VAE switches in Detailer
    THEN: Batched Detailer (+ optional PinCache)
    NOW:  Batched Detailer only                              → KEEP Detailer
```

---

## Appendix A — Deleted `nodes/hswq_pin_cache.py` (full)

Source: `git show 9fdfae3:nodes/hswq_pin_cache.py` (exact pre-removal content).

```python
"""HSWQ Batched Detailer pin cache for current ComfyUI (hostbuf Dynamic VRAM).

ComfyUI no longer exposes ``comfy.pinned_memory.unpin_memory``. Module pins
live in per-model hostbufs and are destroyed by
``ModelPatcherDynamic.partially_unload_ram`` (truncate + unregister).

This module restores the old Detailer PinCache *goal* (avoid repeated
``cudaHostRegister`` / hostbuf rebuild when UNet ↔ VAE switches) on the new
API:

  - Freestanding pinned ``uint8`` buffers pooled by nbytes (same idea as
    pre-hostbuf PinCache).
  - During Detailer, ``partially_unload_ram`` moves pins into that pool
    instead of discarding them.
  - During Detailer, ``pinned_memory.pin_memory(module, subset=, size=)``
    prefers a pool HIT before ``hostbuf.extend``.

**Detailer-only:** nothing is patched at extension import. Use
``activate()`` / ``deactivate()`` or ``hswq_pin_cache_scope()`` around
``HSWQBatchedDetailer.doit`` only. Outside Detailer, ComfyUI pin behavior
is untouched (Nunchaku / Z-Image safe).
"""

from __future__ import annotations

import collections
import logging
from contextlib import contextmanager

_logger = logging.getLogger("HSWQ_PinCache")

_MAX_PIN_CACHE_BYTES = 16 * 1024 * 1024 * 1024

_active = False
_depth = 0
_installed = False

_PIN_BUFFER_POOL = collections.defaultdict(list)
_PIN_CACHE_TOTAL = 0
_pin_cache_stats = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
    "swaps": 0,
    "soft_unloads": 0,
}

_orig_dynamic_partially_unload_ram = None
_orig_pm_pin = None
_Dynamic = None
_pm_mod = None


def is_active() -> bool:
    return _active


def _store_pin_in_pool(pin, size: int) -> None:
    """Keep a freestanding pinned buffer for reuse (16GB cap, adaptive swap)."""
    global _PIN_CACHE_TOTAL
    import comfy.model_management as mm

    _pin_cache_stats["stores"] += 1
    u = _pin_cache_stats["stores"]

    if _PIN_CACHE_TOTAL + size <= _MAX_PIN_CACHE_BYTES:
        _PIN_BUFFER_POOL[size].append(pin)
        _PIN_CACHE_TOTAL += size
        if u <= 3 or u % 200 == 0:
            _logger.info(
                "[HSWQ PinCache] STORE size=%d pool_total=%.1f MB keys=%d stores=%d",
                size,
                _PIN_CACHE_TOTAL / (1024 * 1024),
                len(_PIN_BUFFER_POOL),
                u,
            )
        return

    freed = 0
    for other_size in list(_PIN_BUFFER_POOL.keys()):
        if other_size == size:
            continue
        other_pool = _PIN_BUFFER_POOL[other_size]
        while other_pool and _PIN_CACHE_TOTAL + size > _MAX_PIN_CACHE_BYTES:
            old_pin = other_pool.pop()
            mm.unpin_memory(old_pin)
            _PIN_CACHE_TOTAL -= other_size
            freed += other_size
        if not other_pool:
            del _PIN_BUFFER_POOL[other_size]
        if _PIN_CACHE_TOTAL + size <= _MAX_PIN_CACHE_BYTES:
            break

    if _PIN_CACHE_TOTAL + size <= _MAX_PIN_CACHE_BYTES:
        _PIN_BUFFER_POOL[size].append(pin)
        _PIN_CACHE_TOTAL += size
        _pin_cache_stats["swaps"] += 1
        s = _pin_cache_stats["swaps"]
        if s <= 3 or s % 100 == 0:
            _logger.info(
                "[HSWQ PinCache] SWAP size=%d freed=%.1f MB pool_total=%.1f MB swaps=%d",
                size,
                freed / (1024 * 1024),
                _PIN_CACHE_TOTAL / (1024 * 1024),
                s,
            )
        return

    mm.unpin_memory(pin)
    _pin_cache_stats["evictions"] += 1
    e = _pin_cache_stats["evictions"]
    if e <= 3 or e % 100 == 0:
        _logger.info(
            "[HSWQ PinCache] EVICT size=%d pool_total=%.1f MB evictions=%d",
            size,
            _PIN_CACHE_TOTAL / (1024 * 1024),
            e,
        )


def _take_pin_from_pool(size: int):
    global _PIN_CACHE_TOTAL
    pool = _PIN_BUFFER_POOL.get(size)
    if not pool:
        return None
    pin = pool.pop()
    _PIN_CACHE_TOTAL -= size
    if not pool:
        del _PIN_BUFFER_POOL[size]
    return pin


def _force_unregister_pin(pin, size: int, mm) -> None:
    """unpin_memory first; if still HostRegistered, force cudaHostUnregister + PINNED_MEMORY pop."""
    import torch

    try:
        mm.unpin_memory(pin)
    except Exception:
        pass
    try:
        if not bool(getattr(pin, "is_pinned", lambda: False)()):
            return
    except Exception:
        return
    try:
        ptr = int(pin.data_ptr())
    except Exception:
        return
    if ptr == 0:
        return
    try:
        if torch.cuda.cudart().cudaHostUnregister(ptr) != 0:
            try:
                mm.discard_cuda_async_error()
            except Exception:
                pass
    except Exception:
        pass
    pinned = getattr(mm, "PINNED_MEMORY", None)
    if isinstance(pinned, dict) and ptr in pinned:
        try:
            stored = int(pinned.pop(ptr, 0) or 0)
        except Exception:
            stored = 0
            pinned.pop(ptr, None)
        try:
            mm.TOTAL_PINNED_MEMORY = max(
                0, int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0) - (stored or int(size or 0))
            )
        except Exception:
            pass


def _drain_pool() -> None:
    global _PIN_CACHE_TOTAL
    try:
        import comfy.model_management as mm
    except ImportError:
        _PIN_BUFFER_POOL.clear()
        _PIN_CACHE_TOTAL = 0
        return
    for size, pool in list(_PIN_BUFFER_POOL.items()):
        while pool:
            pin = pool.pop()
            _force_unregister_pin(pin, size, mm)
            _PIN_CACHE_TOTAL = max(0, _PIN_CACHE_TOTAL - size)
        del _PIN_BUFFER_POOL[size]
    _PIN_CACHE_TOTAL = 0


def _cached_pin_memory(module, subset="weights", size=None):
    """Detailer-gated pin_memory with freestanding pool HIT before hostbuf.extend."""
    global _PIN_CACHE_TOTAL

    if not _active:
        return _orig_pm_pin(module, subset=subset, size=size)

    try:
        import comfy.model_management as mm
        import comfy.memory_management as mem
        import comfy.pinned_memory as pm
        from comfy.cli_args import args
    except ImportError:
        return _orig_pm_pin(module, subset=subset, size=size)

    if args.disable_pinned_memory:
        return

    # Warm re-register of an existing _pin (hostbuf view or pooled).
    pin = pm.get_pin(module, subset)
    if pin is not None:
        return

    pin_state = getattr(module, "_pin_state", None)
    if pin_state is None:
        return _orig_pm_pin(module, subset=subset, size=size)

    hostbuf, stack, stack_split, pinned_size, counter, buckets = pin_state[subset]
    if size is None:
        size = mem.vram_aligned_size([module.weight, module.bias])

    pooled = _take_pin_from_pool(size)
    if pooled is not None:
        # Pool entries stay cudaHostRegistered (old PinCache semantics).
        # TOTAL_PINNED_MEMORY already includes them via mm.pin_memory at STORE.
        if not getattr(pooled, "is_pinned", lambda: False)():
            if not mm.pin_memory(pooled):
                _pin_cache_stats["misses"] += 1
                return _orig_pm_pin(module, subset=subset, size=size)

        module._pin = pooled
        module._pin_registered = True
        module._hswq_pooled_pin = True
        # Sentinel offset: not a hostbuf slice — unload must not truncate.
        stack.append((module, -1))
        module._pin_stack_index = len(stack) - 1
        stack_split[0] = max(stack_split[0], module._pin_stack_index)
        pinned_size[0] += size
        try:
            priority = getattr(module, "_pin_balancer_priority", None)
            if priority is None:
                import comfy.utils as comfy_utils

                priority = comfy_utils.bit_reverse_range(counter[0], 16)
                counter[0] += 1
                module._pin_balancer_priority = priority
            pm._add_to_bucket(module, buckets, size, priority)
        except Exception:
            pass

        _pin_cache_stats["hits"] += 1
        h = _pin_cache_stats["hits"]
        if h <= 3 or h % 200 == 0:
            _logger.info(
                "[HSWQ PinCache] HIT size=%d pool_total=%.1f MB hits=%d misses=%d",
                size,
                _PIN_CACHE_TOTAL / (1024 * 1024),
                h,
                _pin_cache_stats["misses"],
            )
        return True

    _pin_cache_stats["misses"] += 1
    return _orig_pm_pin(module, subset=subset, size=size)


def _soft_partially_unload_ram(self, ram_to_unload, subsets=None):
    """Move pins into the freestanding pool instead of discarding them."""
    if subsets is None:
        subsets = ["weights", "patches"]
    if not _active:
        return _orig_dynamic_partially_unload_ram(self, ram_to_unload, subsets=subsets)

    import torch
    import comfy.model_management as mm

    freed = 0
    pin_state = self.model.dynamic_pins[self.load_device]
    for subset in subsets:
        hostbuf, stack, stack_split, pinned_size, *_rest = pin_state[subset]
        buckets = _rest[-1] if _rest else None
        while len(stack) > 0 and ram_to_unload > 0:
            module, offset = stack.pop()
            pin = getattr(module, "_pin", None)
            if pin is None:
                stack_split[0] = min(stack_split[0], len(stack) - 1)
                continue

            size = pin.numel() * pin.element_size()
            registered = bool(getattr(module, "_pin_registered", False))
            is_pooled = bool(getattr(module, "_hswq_pooled_pin", False)) or offset < 0

            if hasattr(module, "_pin_balancer_entry"):
                try:
                    module._pin_balancer_entry[-1] = None
                except Exception:
                    pass
                try:
                    del module._pin_balancer_entry
                except Exception:
                    pass

            del module._pin
            if hasattr(module, "_pin_stack_index"):
                del module._pin_stack_index
            if hasattr(module, "_hswq_pooled_pin"):
                del module._hswq_pooled_pin
            module._pin_registered = False

            if is_pooled:
                # Keep cudaHostRegister (old PinCache). Only drop per-model pinned_size.
                if registered:
                    pinned_size[0] = max(0, pinned_size[0] - size)
                if not getattr(pin, "is_pinned", lambda: False)():
                    mm.pin_memory(pin)
                _store_pin_in_pool(pin, size)
            else:
                # Hostbuf view → freestanding pool entry.
                if registered:
                    try:
                        ptr = int(pin.data_ptr())
                    except Exception:
                        ptr = 0
                    if ptr and torch.cuda.cudart().cudaHostUnregister(ptr) != 0:
                        mm.discard_cuda_async_error()
                    else:
                        pinned = getattr(mm, "PINNED_MEMORY", None)
                        if isinstance(pinned, dict) and ptr in pinned:
                            try:
                                pinned.pop(ptr, None)
                            except Exception:
                                pass
                        mm.TOTAL_PINNED_MEMORY = max(0, mm.TOTAL_PINNED_MEMORY - size)
                        pinned_size[0] = max(0, pinned_size[0] - size)
                try:
                    hostbuf.truncate(offset, do_unregister=False)
                except TypeError:
                    try:
                        hostbuf.truncate(offset)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    freestanding = torch.empty((size,), dtype=torch.uint8)
                    if mm.pin_memory(freestanding):
                        _store_pin_in_pool(freestanding, size)
                    else:
                        del freestanding
                except Exception:
                    pass

            stack_split[0] = min(stack_split[0], len(stack) - 1)
            if buckets is not None:
                # Leave buckets; cleanup pass filters None entries.
                pass

            freed += size
            ram_to_unload -= size

    _pin_cache_stats["soft_unloads"] += 1
    n = _pin_cache_stats["soft_unloads"]
    if n <= 3 or n % 50 == 0:
        _logger.info(
            "[HSWQ PinCache] soft-unload #%d moved=%.1f MB pool_total=%.1f MB",
            n,
            freed / (1024 * 1024),
            _PIN_CACHE_TOTAL / (1024 * 1024),
        )
    return freed


def _install_patches() -> bool:
    global _installed, _orig_dynamic_partially_unload_ram, _orig_pm_pin
    global _Dynamic, _pm_mod

    if _installed:
        return True

    try:
        import comfy.model_patcher as mp
        import comfy.pinned_memory as pm
    except ImportError as e:
        _logger.warning("[HSWQ PinCache] import failed: %s", e)
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None or getattr(Dynamic, "partially_unload_ram", None) is None:
        _logger.warning("[HSWQ PinCache] ModelPatcherDynamic.partially_unload_ram missing")
        return False

    _Dynamic = Dynamic
    _orig_dynamic_partially_unload_ram = Dynamic.partially_unload_ram
    Dynamic.partially_unload_ram = _soft_partially_unload_ram

    _pm_mod = pm
    _orig_pm_pin = pm.pin_memory
    pm.pin_memory = _cached_pin_memory
    pm._hswq_pin_cache_active = False

    _installed = True
    _logger.info(
        "[HSWQ PinCache] Detailer hostbuf+pool ready (max %.1f GB)",
        _MAX_PIN_CACHE_BYTES / (1024 ** 3),
    )
    return True


def _uninstall_patches() -> None:
    global _installed, _orig_dynamic_partially_unload_ram, _orig_pm_pin
    global _Dynamic, _pm_mod

    if not _installed:
        return

    if _Dynamic is not None and _orig_dynamic_partially_unload_ram is not None:
        _Dynamic.partially_unload_ram = _orig_dynamic_partially_unload_ram
    if _pm_mod is not None and _orig_pm_pin is not None:
        _pm_mod.pin_memory = _orig_pm_pin
        _pm_mod._hswq_pin_cache_active = False

    _drain_pool()

    _orig_dynamic_partially_unload_ram = None
    _orig_pm_pin = None
    _Dynamic = None
    _pm_mod = None
    _installed = False


def activate() -> bool:
    """Enable Detailer pin cache (nested-safe)."""
    global _active, _depth
    if not _install_patches():
        return False
    _depth += 1
    _active = True
    if _pm_mod is not None:
        _pm_mod._hswq_pin_cache_active = True
    if _depth == 1:
        _logger.info("[HSWQ PinCache] ACTIVE (Batched Detailer)")
    return True


def deactivate() -> None:
    """Disable Detailer pin cache; uninstall when nesting depth hits 0."""
    global _active, _depth
    if _depth > 0:
        _depth -= 1
    if _depth > 0:
        return
    _active = False
    if _pm_mod is not None:
        _pm_mod._hswq_pin_cache_active = False
    _uninstall_patches()
    _logger.info(
        "[HSWQ PinCache] OFF hits=%d misses=%d stores=%d soft_unloads=%d",
        _pin_cache_stats["hits"],
        _pin_cache_stats["misses"],
        _pin_cache_stats["stores"],
        _pin_cache_stats["soft_unloads"],
    )


def purge_pin_cache() -> int:
    """Force-drain every pooled pin buffer. Returns bytes held before drain."""
    global _PIN_CACHE_TOTAL, _active, _depth
    before = int(_PIN_CACHE_TOTAL)
    _active = False
    _depth = 0
    if _pm_mod is not None:
        try:
            _pm_mod._hswq_pin_cache_active = False
        except Exception:
            pass
    try:
        _uninstall_patches()
    except Exception:
        pass
    _drain_pool()
    _logger.info("[HSWQ PinCache] PURGE drained %.1f MB", before / (1024 * 1024))
    return before


@contextmanager
def hswq_pin_cache_scope():
    """Context manager: PinCache on only for Batched Detailer."""
    ok = activate()
    try:
        yield ok
    finally:
        deactivate()
```

---

**End of guide.**
