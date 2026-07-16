# HSWQ INT8 and Nunchaku Coexistence — Complete Technical Guide

**Date of record:** 2026-07-16  
**Repository:** `ussoewwin/ComfyUI-nunchaku-unofficial-loader`  
**Primary file:** `patches/comfy_quant_int8.py`  
**Status of fix:** Unidirectional VRAM handoff **INT8 Dynamic → Nunchaku SVDQ** (working).  
**Rejected approach:** Bidirectional handoff with pin/VBAR reset + Nunchaku detach before INT8 reload (**broke INT8 reload**; rolled back).

This document explains the **INT8 HSWQ UNet (Dynamic VRAM) → Nunchaku Z-Image (SVDQ)** crash, the **root cause**, the **working countermeasure**, the **files and full code** involved, and the **meaning** of each piece. Section numbering matches the requested outline (①–⑥).

---

## Table of contents

1. [Error content (①)](#1-error-content-)
2. [Root cause (②)](#2-root-cause-)
3. [Countermeasure overview (③)](#3-countermeasure-overview-)
4. [Modified file names (④)](#4-modified-file-names-)
5. [Full text of added / modified code (⑤)](#5-full-text-of-added--modified-code-)
6. [Meaning of the code (⑥)](#6-meaning-of-the-code-)
7. [Appendix A — Rejected bidirectional handoff (v2)](#appendix-a--rejected-bidirectional-handoff-v2)
8. [Appendix B — Related false-positive INT8 bake on Nunchaku](#appendix-b--related-false-positive-int8-bake-on-nunchaku)

---

## 1. Error content (①)

### 1.1 Reproduction workflow

Typical failing sequence (same ComfyUI process, no restart between steps):

1. Load **HSWQ INT8** UNet (e.g. `*_int8` / `int8_tensorwise`) under **Dynamic VRAM**.
2. Run KSampler — INT8 path completes (often with INT8 LoRA bake `requant` OK).
3. Load / sample **Nunchaku Z-Image** (SVDQ, EasyCache, etc.) in the same session.

### 1.2 Observed log pattern

After INT8 succeeds, Nunchaku load reports essentially **zero GPU budget**:

```text
loaded partially; 0.00 MB usable, 0.00 MB loaded, ~4007 MB offloaded, ...
0 models unloaded.
```

Then the process dies during model initialization / first forward:

```text
Fatal Python error: Aborted
```

Stack / site of failure (typical):

- ComfyUI: `comfy/ldm/lumina/model.py` → `patchify_and_embed`
- Nunchaku path uses classic patcher (`ModelPatcher` / `ZImageModelPatcher`), **not** `ModelPatcherDynamic`

### 1.3 What the numbers mean

| Log fragment | Meaning |
|--------------|---------|
| `0.00 MB usable` | Comfy decided almost **no** free GPU memory is available for this load |
| `0.00 MB loaded` / `~4007 MB offloaded` | Weights stay off GPU (CPU / pin / offload side); GPU gets nothing useful |
| `0 models unloaded` | `free_memory` did not fully remove the resident that still occupies VRAM |
| `Fatal Python error: Aborted` | Native / CUDA path aborts when running with a non-loaded / empty GPU model |

INT8 alone is fine. The failure appears at the **handoff into Nunchaku** while INT8 Dynamic residency is still present.

---

## 2. Root cause (②)

### 2.1 Two loaders, two memory models

| Path | Patcher | VRAM style |
|------|---------|------------|
| HSWQ INT8 UNet | `ModelPatcherDynamic` | Dynamic VRAM: **VBAR** + **HostBuffer** / pin staging |
| Nunchaku Z-Image SVDQ | Classic `ModelPatcher` / `ZImageModelPatcher` | Conventional load / partial load; **not** the same Dynamic accounting |

They share one GPU and Comfy’s `current_loaded_models` / `free_memory` machinery, but they do **not** free each other’s reservations the same way.

### 2.2 Why Comfy’s normal unload is not enough

When Nunchaku requests GPU memory, Comfy calls `free_memory` (and related unload). For Dynamic models this often stops after **`ModelPatcherDynamic.partially_unload`**: a little VBAR is freed and the call returns **without a full `detach`**.

What remains:

- HostBuffers / staging / Dynamic pin state still occupy host/GPU-adjacent resources
- Accounting still makes Comfy believe usable VRAM for the **next** classic load is **~0**
- Nunchaku therefore starts with `0.00 MB usable` → Abort in Lumina / SVDQ forward

### 2.3 What is *not* the root cause (for this Abort)

- INT8 LoRA bake succeeding on INT8 is expected; that alone does not explain Nunchaku Abort.
- Nunchaku “being broken by itself” is not required; the same Nunchaku path can work after a clean restart (no leftover INT8 Dynamic).
- A later **bidirectional** attempt (also detach Nunchaku before INT8 reload + reset pins) is a **different** problem: it broke INT8 **reload**. That approach was rejected; see [Appendix A](#appendix-a--rejected-bidirectional-handoff-v2).

### 2.4 One-sentence root cause

**INT8 Dynamic VRAM is only partially unloaded; residual HostBuffer/VBAR occupancy leaves Nunchaku with zero usable VRAM, so SVDQ forward Aborts.**

---

## 3. Countermeasure overview (③)

### 3.1 Working policy (current)

**Unidirectional handoff only:** when `load_models_gpu` is about to load a **Nunchaku SVDQ** model, **force-detach** any **INT8 Dynamic** patchers on the same device (full `detach`, drop from `current_loaded_models`), then `free_memory(1e30)` + `soft_empty_cache`, then call the original `load_models_gpu`.

- Trigger: `_model_is_nunchaku_svdq` on the models being loaded.
- Target of detach: Dynamic patchers whose base has **comfy_quant** `QuantizedTensor` INT8 weights (`_model_has_int8_quantized_weights`).
- Keep list: the Nunchaku patchers about to load (and their `model_patches_models`).

### 3.2 Explicit non-goals (current)

| Action | Status |
|--------|--------|
| Force detach INT8 Dynamic before Nunchaku | **In scope (required)** |
| Reset Dynamic pin/VBAR for later INT8 reload | **Out of scope** (v2; broke reload) |
| Detach Nunchaku before INT8 reload | **Out of scope** (v2; broke reload) |
| Modify ComfyUI core | **Not done** (extension monkey-patch only) |

### 3.3 Success signals

After ComfyUI restart with the patch applied (INT8 patches install on INT8 load path):

```text
[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load (forced INT8 Dynamic unload=…)
```

Nunchaku load must **not** show `0.00 MB usable` / `0.00 MB loaded` with multi-GB offloaded in the same failure pattern.

### 3.4 Supporting coexistence guards (related)

Separately, Dynamic INT8 LoRA bake must **never** run on Nunchaku (Comfy class name is often `Lumina2`). That uses the same `_model_is_nunchaku_svdq` / `_model_has_int8_quantized_weights` helpers. See [Appendix B](#appendix-b--related-false-positive-int8-bake-on-nunchaku).

---

## 4. Modified file names (④)

| Path | Role |
|------|------|
| `patches/comfy_quant_int8.py` | **Only file** for this coexistence handoff: SVDQ/INT8 detectors, force-detach helper, `load_models_gpu` wrapper, wiring inside `apply_comfy_quant_int8_patches()` |

No ComfyUI core files. No separate new module. Install copy (when synced):  
`ComfyUI/custom_nodes/ComfyUI-nunchaku-unofficial-loader/patches/comfy_quant_int8.py`.

---

## 5. Full text of added / modified code (⑤)

Source of truth: working tree `patches/comfy_quant_int8.py` as of this guide.  
Handoff version marker in tree: `_VER = 3` (behavior = original unidirectional handoff; version bumped so a prior bidirectional `_VER = 2` wrapper is replaced on re-apply).

### 5.1 SVDQ / INT8 detection (helpers used by handoff and bake)

```python
def _model_is_nunchaku_svdq(model) -> bool:
    """True when BaseModel / NextDiT carries Nunchaku SVDQ modules.

    ComfyUI registers Z-Image as ``Lumina2`` — classname checks for
    ``Nunchaku`` / ``ZImage`` miss that. Any SVDQ / ComfyNunchaku module means
    never run comfy_quant INT8 Dynamic LoRA bake.
    """
    if model is None:
        return False
    roots = [model]
    dm = getattr(model, "diffusion_model", None)
    if dm is not None:
        roots.append(dm)
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        roots.append(inner)
        dm2 = getattr(inner, "diffusion_model", None)
        if dm2 is not None:
            roots.append(dm2)
    seen = set()
    for root in roots:
        rid = id(root)
        if rid in seen:
            continue
        seen.add(rid)
        try:
            named = root.named_modules()
        except Exception:
            continue
        for _, module in named:
            cls_name = type(module).__name__
            if (
                "SVDQ" in cls_name
                or "Nunchaku" in cls_name
                or cls_name.startswith("ComfyNunchaku")
            ):
                return True
            mod = getattr(type(module), "__module__", "") or ""
            if "nunchaku" in mod.lower():
                return True
    return False


def _model_has_int8_quantized_weights(model) -> bool:
    """True only for native comfy_quant INT8 (comfy.quant_ops.QuantizedTensor).

    Must NOT treat bare ``torch.int8`` weights as comfy_quant INT8.
    Nunchaku SVDQ / Z-Image / Lumina2 modules often use int8 storage; a false
    positive here arms Dynamic.load INT8 LoRA bake and can Abort those paths.
    """
    if _model_is_nunchaku_svdq(model):
        return False
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "SVDQ" in cls_name or "Nunchaku" in cls_name:
            continue
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if isinstance(w, QuantizedTensor):
            return True
    return False
```

### 5.2 Force detach INT8 Dynamic + `load_models_gpu` handoff (core fix)

```python
def _force_detach_int8_dynamic_models(device=None, keep_patchers=None) -> int:
    """Fully detach INT8 Dynamic VRAM models (VBAR + hostbufs).

    Comfy's free_memory can stop after ModelPatcherDynamic.partially_unload
    frees a little VBAR and return without detach. HostBuffers / staging then
    leave Nunchaku (classic ModelPatcher / ZImageModelPatcher) with
    ``0.00 MB usable`` → Aborted in lumina patchify_and_embed.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return 0

    keep_ids = {id(p) for p in (keep_patchers or []) if p is not None}
    unloaded = 0
    i = 0
    while i < len(mm.current_loaded_models):
        lm = mm.current_loaded_models[i]
        patcher = lm.model
        if patcher is None:
            i += 1
            continue
        if id(patcher) in keep_ids:
            i += 1
            continue
        if device is not None and getattr(lm, "device", None) is not None:
            try:
                if str(lm.device) != str(device):
                    i += 1
                    continue
            except Exception:
                pass
        is_dyn = False
        try:
            is_dyn = bool(patcher.is_dynamic())
        except Exception:
            is_dyn = False
        if not is_dyn:
            i += 1
            continue
        base = getattr(patcher, "model", None)
        if base is None or not _model_has_int8_quantized_weights(base):
            i += 1
            continue
        try:
            patcher.detach(unpatch_weights=True)
        except Exception as exc:
            _console(f"[HSWQ INT8→Nunchaku] detach failed: {exc!r}")
        try:
            fin = getattr(lm, "model_finalizer", None)
            if fin is not None:
                fin.detach()
        except Exception:
            pass
        try:
            lm.model_finalizer = None
            lm.real_model = None
        except Exception:
            pass
        mm.current_loaded_models.pop(i)
        unloaded += 1
    if unloaded > 0:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    return unloaded


def _patch_load_models_gpu_int8_nunchaku_handoff() -> bool:
    """Before Nunchaku SVDQ load, force-release INT8 Dynamic VRAM occupancy."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return False

    original = getattr(mm, "load_models_gpu", None)
    if original is None:
        return False
    # v3 = rollback of bidirectional v2; same behavior as original unidirectional handoff.
    _VER = 3
    if getattr(original, "_hswq_int8_nunchaku_handoff_ver", 0) >= _VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_models_gpu", original)

    def load_models_gpu(
        models,
        memory_required=0,
        force_patch_weights=False,
        minimum_memory_required=None,
        force_full_load=False,
    ):
        keep = []
        need_handoff = False
        device = None
        for m in models or []:
            keep.append(m)
            for mm_extra in getattr(m, "model_patches_models", lambda: [])() or []:
                keep.append(mm_extra)
            base = getattr(m, "model", None)
            if _model_is_nunchaku_svdq(base) or _model_is_nunchaku_svdq(m):
                need_handoff = True
                if device is None:
                    device = getattr(m, "load_device", None)
        if need_handoff:
            n = _force_detach_int8_dynamic_models(device=device, keep_patchers=keep)
            try:
                if device is not None:
                    mm.free_memory(1e30, device, keep_loaded=[], for_dynamic=False)
                mm.soft_empty_cache()
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] free_memory handoff failed: {exc!r}")
            _console(
                f"[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load "
                f"(forced INT8 Dynamic unload={n})"
            )
        return true_orig(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            force_full_load=force_full_load,
        )

    load_models_gpu._hswq_int8_nunchaku_handoff = True
    load_models_gpu._hswq_int8_nunchaku_handoff_ver = _VER
    load_models_gpu._hswq_orig_load_models_gpu = true_orig
    mm.load_models_gpu = load_models_gpu
    return True
```

### 5.3 Wiring inside `apply_comfy_quant_int8_patches`

Relevant lines (same file):

```python
    ok_dyn_bake = _patch_model_patcher_dynamic_int8_lora_bake()
    ok_handoff = _patch_load_models_gpu_int8_nunchaku_handoff()
    if _PATCHES_APPLIED:
        return True
    # ...
            f"{' + Dynamic INT8 LoRA bake' if ok_dyn_bake else ''}"
            f"{' + INT8→Nunchaku VRAM handoff' if ok_handoff else ''})"
```

Handoff installs whenever `apply_comfy_quant_int8_patches()` runs (normally when an INT8 HSWQ load path applies patches). After that, every subsequent `comfy.model_management.load_models_gpu` in the process goes through the wrapper.

### 5.4 Dynamic bake skip on SVDQ (coexistence guard; same file)

Inside `_patch_model_patcher_dynamic_int8_lora_bake` → wrapped `load`:

```python
        # INT8 LoRA bake only — never touch Nunchaku SVDQ (class is often Lumina2).
        if _model_is_nunchaku_svdq(self.model):
            return result
        if not _model_has_int8_quantized_weights(self.model) and not getattr(
            self.model, "_hswq_int8_baked_keys", None
        ):
            return result
```

---

## 6. Meaning of the code (⑥)

### 6.1 `_model_is_nunchaku_svdq`

| Concern | Meaning |
|---------|---------|
| Why scan modules? | Comfy registers Z-Image as **`Lumina2`**; class-name checks for `Nunchaku` / `ZImage` miss it |
| What counts? | Module class contains `SVDQ` / `Nunchaku`, or `ComfyNunchaku*`, or `__module__` contains `nunchaku` |
| Role in handoff | **Only** when the models being loaded are SVDQ does handoff run |

### 6.2 `_model_has_int8_quantized_weights`

| Concern | Meaning |
|---------|---------|
| True only for | `comfy.quant_ops.QuantizedTensor` on non-SVDQ modules |
| False for | Bare `torch.int8` (common inside Nunchaku) |
| Role in detach | Only **HSWQ / comfy_quant INT8 Dynamic** patchers are force-detached; other Dynamic models are left alone |

### 6.3 `_force_detach_int8_dynamic_models`

| Step | Meaning |
|------|---------|
| Walk `current_loaded_models` | Find live LoadedModel entries |
| Skip keep / wrong device | Do not detach the Nunchaku about to load |
| Require `is_dynamic()` + INT8 QuantizedTensor | Target INT8 Dynamic only |
| `patcher.detach(unpatch_weights=True)` | Full release of Dynamic VBAR / hostbufs (stronger than `partially_unload`) |
| Clear finalizer / real_model / pop list | Remove Comfy’s LoadedModel bookkeeping so free_memory sees them gone |
| `soft_empty_cache` if any unloaded | Encourage CUDA to reclaim |

### 6.4 `_patch_load_models_gpu_int8_nunchaku_handoff`

| Step | Meaning |
|------|---------|
| Wrap `mm.load_models_gpu` | Intercept every GPU model load after INT8 patches apply |
| `_VER = 3` | Re-apply over older bidirectional `_VER = 2`; **behavior** is unidirectional (same as original `_VER = 1`) |
| Build `keep` | Incoming models + their patch models stay protected |
| `need_handoff` | Set only if any incoming model is Nunchaku SVDQ |
| Force detach + `free_memory(1e30)` + soft cache | Make room **before** original loader runs |
| Log line | Auditable proof that handoff ran |

### 6.5 Why unidirectional only (owner-validated)

- **INT8 → Nunchaku:** force detach INT8 Dynamic → Nunchaku gets usable VRAM → Abort gone. **This works.**
- **Nunchaku → INT8 “park + pin/VBAR reset + detach SVDQ” (v2):** intended to make INT8 reload clean; **in practice INT8 reload broke.** Owner ordered rollback; current tree must not reintroduce that path.

---

## Appendix A — Rejected bidirectional handoff (v2)

### A.1 What v2 added (do not reintroduce)

1. After parking INT8 Dynamic, **reset** `hostbufs_initialized` / HostBuffer / VBAR so a later `Dynamic.load` could recreate pins.
2. When loading INT8 Dynamic again, **detach Nunchaku SVDQ** first.

Planned logs (v2 only):

```text
[HSWQ INT8→Nunchaku] park INT8 Dynamic for SVDQ (… pin/VBAR reset for reload)
[HSWQ Nunchaku→INT8] detach SVDQ before INT8 Dynamic reload
```

### A.2 Why it was cancelled

Owner finding: **reload path broke**; unidirectional handoff is the state that **actually works**. v2 was reverted to INT8→Nunchaku-only force detach (current §5.2).

### A.3 Lesson

Do not “complete” coexistence by symmetrically unloading Nunchaku for INT8 until a reload-safe design is proven. Prefer the minimal fix that clears INT8 Dynamic before SVDQ load.

---

## Appendix B — Related false-positive INT8 bake on Nunchaku

A **separate** coexistence bug: treating Nunchaku int8 storage as comfy_quant INT8 armed **Dynamic INT8 LoRA bake** on Lumina2/SVDQ and could Abort Nunchaku even without the VRAM handoff issue.

Guards (same helpers as §5.1):

- `_model_is_nunchaku_svdq` → skip bake
- `_model_has_int8_quantized_weights` → require `QuantizedTensor`, not bare int8

This guide’s primary Abort (`0.00 MB usable` after INT8 Dynamic) is the **VRAM handoff** issue in §1–§3. Appendix B is listed so both INT8↔Nunchaku failure modes stay documented in one place.

---

## Document control

| Item | Value |
|------|-------|
| Guide path | `md/HSWQ_INT8_NUNCHAKU_COEXISTENCE_GUIDE.md` |
| Code path | `patches/comfy_quant_int8.py` |
| Working handoff | Unidirectional INT8 Dynamic → Nunchaku |
| Rejected handoff | Bidirectional park/reset + Nunchaku→INT8 detach |

End of guide.
