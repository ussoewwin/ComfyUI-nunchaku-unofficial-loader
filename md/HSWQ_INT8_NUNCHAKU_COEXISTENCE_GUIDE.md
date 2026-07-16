# HSWQ INT8 and Nunchaku Coexistence — Complete Technical Guide

**Date of record:** 2026-07-16 (rewritten after PinCache / Abort causation audit)  
**Repository:** `ussoewwin/ComfyUI-nunchaku-unofficial-loader`  
**Primary file:** `patches/comfy_quant_int8.py`  
**Authoritative tree tip for this rewrite:** `4e51074` (`fix: remove Detailer Pin Buffer Cache again`) on top of `df9ba74` / `747b64b`  
**PinCache status:** **Removed** (again). Not part of the INT8→Nunchaku Abort fix. See `md/HSWQ_PIN_BUFFER_CACHE_REMOVAL_GUIDE.md` and [v3.2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.2.0).

This document explains **INT8 HSWQ (Dynamic VRAM) → Nunchaku Z-Image (SVDQ)** failures in one ComfyUI process: **what Aborts look like**, **verified root causes**, **working countermeasures in `comfy_quant_int8.py`**, **what is explicitly not a countermeasure (Pin Buffer Cache / HostBuffer theater)**, and **how to read the code**.

---

## Table of contents

1. [Error content](#1-error-content)
2. [Root cause (verified)](#2-root-cause-verified)
3. [Countermeasure overview](#3-countermeasure-overview)
4. [Modified file names](#4-modified-file-names)
5. [Full text of added / modified code](#5-full-text-of-added--modified-code)
6. [Meaning of the code](#6-meaning-of-the-code)
7. [Appendix A — Rejected bidirectional handoff (v2)](#appendix-a--rejected-bidirectional-handoff-v2)
8. [Appendix B — Pin Buffer Cache is not the Abort fix](#appendix-b--pin-buffer-cache-is-not-the-abort-fix)
9. [Appendix C — Audit timeline (correlation ≠ causation)](#appendix-c--audit-timeline-correlation--causation)

---

## 1. Error content

### 1.1 Reproduction workflow

Typical failing sequence (same ComfyUI process, no restart between steps):

1. Load **HSWQ INT8** UNet (`*_int8` / `int8_tensorwise`) under **Dynamic VRAM**.
2. Run KSampler — INT8 path completes (often with INT8 LoRA bake `requant` OK).
3. Load / sample **Nunchaku Z-Image** (SVDQ) in the same session.

### 1.2 Two Abort signatures (do not conflate)

| Signature | Typical logs / site | What it usually means |
|-----------|---------------------|------------------------|
| **A — Zero usable VRAM handoff** | `loaded partially; 0.00 MB usable, 0.00 MB loaded, ~N GB offloaded` then `Fatal Python error: Aborted` (often Lumina `patchify_and_embed`) | INT8 **Dynamic** residency not fully released; classic Nunchaku patcher starts with empty GPU budget |
| **B — Fused CUDA Abort with VRAM OK** | Handoff log may show unload; GPU memory looks fine; Abort in Nunchaku fused path such as `_forward_silu_gating` | Extension patches **grabbed bare `torch.int8`** (SVDQ storage) as if it were `comfy.quant_ops.QuantizedTensor` — LowVram / bake corruption |

Signature **A** is the original “INT8 Dynamic left occupying accounting” problem.  
Signature **B** is a **separate** coexistence bug: **QuantizedTensor-only** discipline broken on Nunchaku weights.

### 1.3 What the zero-VRAM numbers mean (signature A)

| Log fragment | Meaning |
|--------------|---------|
| `0.00 MB usable` | Comfy decided almost **no** free GPU memory is available for this load |
| `0.00 MB loaded` / multi-GB offloaded | Weights stay off GPU; GPU gets nothing useful |
| `0 models unloaded` | `free_memory` did not fully remove the resident that still occupies VRAM |
| `Fatal Python error: Aborted` | Native / CUDA path aborts when running with a non-loaded / empty GPU model |

INT8 alone is fine. Signature A appears at the **handoff into Nunchaku** while INT8 Dynamic residency is still present.

---

## 2. Root cause (verified)

### 2.1 Two loaders, two memory models (signature A)

| Path | Patcher | VRAM style |
|------|---------|------------|
| HSWQ INT8 UNet | `ModelPatcherDynamic` | Dynamic VRAM: **VBAR** + **HostBuffer** / pin staging |
| Nunchaku Z-Image SVDQ | Classic `ModelPatcher` / `ZImageModelPatcher` | Conventional load / partial load; **not** the same Dynamic accounting |

They share one GPU and Comfy’s `current_loaded_models` / `free_memory` machinery, but they do **not** free each other’s reservations the same way.

When Nunchaku requests GPU memory, Comfy often stops after **`ModelPatcherDynamic.partially_unload`**: a little VBAR is freed and the call returns **without a full `detach`**. Residual Dynamic occupancy leaves usable VRAM for the next classic load at **~0** → signature A.

### 2.2 QuantizedTensor vs bare int8 (signature B — critical)

| Weight kind | Who owns it | Extension must |
|-------------|-------------|----------------|
| `comfy.quant_ops.QuantizedTensor` | HSWQ / comfy_quant INT8 | May LowVram-fix / Dynamic bake / force-detach on handoff |
| Bare `torch.int8` tensor | Common **inside Nunchaku SVDQ** | **Must leave upstream alone** |

If LowVramPatch or Dynamic bake treats bare int8 as comfy_quant INT8:

- LoRA / dtype path dequantizes or rebakes the wrong tensors
- Fused Nunchaku CUDA (`_forward_silu_gating`, etc.) Aborts **even when VRAM handoff already freed GPU memory**

Comfy registers Z-Image as **`Lumina2`**. Class-name checks for `Nunchaku` / `ZImage` alone are insufficient; module scan for SVDQ / `nunchaku` is required (`_model_is_nunchaku_svdq`).

### 2.3 Broken handoff API (signature A amplifier)

ComfyUI `ModelPatcher.detach` accepts **`unpatch_all=`**, not `unpatch_weights=`.

A handoff that called `detach(unpatch_weights=True)` raised:

```text
detach failed: TypeError(... unpatch_weights ...)
```

and left INT8 Dynamic still resident → signature A returned even when “handoff code” was present.

**Current correct call:** `patcher.detach(unpatch_all=True)` with `TypeError` fallback to `detach()`.

### 2.4 What is *not* the root cause of INT8→Nunchaku Abort

| Claim | Verdict |
|-------|---------|
| Pin Buffer Cache is required to stop INT8→Nunchaku Abort | **False.** PinCache is Detailer-scoped pin pooling for the **old** per-layer `cudaHostRegister` lifecycle. INT8→Nunchaku load does **not** activate `[HSWQ PinCache] ACTIVE`. See [Appendix B](#appendix-b--pin-buffer-cache-is-not-the-abort-fix). |
| FaceDetailer pin thrash is why PinCache “fixed” Abort | **False framing.** FaceDetailer amplified **old pin Register thrash**; Batched Detailer cuts **switch count**. That is orthogonal to SVDQ Abort after INT8 Dynamic. |
| HostBuffer / VBAR “reset theater” before Nunchaku | **Rejected.** Bidirectional park/reset broke INT8 reload ([Appendix A](#appendix-a--rejected-bidirectional-handoff-v2)). |
| “Deleting PinCache caused Abort after `d6c87ff`” | **Correlation, not causation.** Same retest showed `detach failed: TypeError(unpatch_weights)` — handoff bug already in `e3edf77`. `d6c87ff` removed PinCache only; it did not introduce that TypeError. See [Appendix C](#appendix-c--audit-timeline-correlation--causation). |

### 2.5 One-sentence root causes

1. **Signature A:** INT8 Dynamic is only partially unloaded; residual occupancy leaves Nunchaku with zero usable VRAM → Abort. Fix = force-detach INT8 Dynamic with a **working** `detach` API before SVDQ load.  
2. **Signature B:** Extension LowVram / bake paths must touch **`QuantizedTensor` only**; grabbing bare Nunchaku int8 corrupts fused CUDA → Abort with VRAM OK.

---

## 3. Countermeasure overview

### 3.1 Working policy (current — three parts, one file)

All live in `patches/comfy_quant_int8.py`:

| Part | Marker | Role |
|------|--------|------|
| **LowVramPatch float intermediate** | `_LV_VER = 3` | Custom path **only** when `weight` is `QuantizedTensor`; bare int8 → upstream unchanged |
| **Dynamic INT8 LoRA bake** | `_DYN_VER = 5` | Skip SVDQ; bake only `QuantizedTensor` keys |
| **Unidirectional VRAM handoff** | `_VER = 4` | Before Nunchaku SVDQ `load_models_gpu`, force-detach INT8 Dynamic via `detach(unpatch_all=True)`, then `free_memory(1e30)` + `soft_empty_cache` |

### 3.2 Explicit non-goals (current)

| Action | Status |
|--------|--------|
| Force detach INT8 Dynamic before Nunchaku | **In scope (required)** |
| LowVram / bake **QuantizedTensor-only** | **In scope (required)** |
| Reintroduce Pin Buffer Cache for Abort | **Forbidden** — removed (`4e51074`); not an Abort countermeasure |
| Reset Dynamic pin/VBAR for later INT8 reload | **Out of scope** (v2; broke reload) |
| Detach Nunchaku before INT8 reload | **Out of scope** (v2; broke reload) |
| Modify ComfyUI core | **Not done** (extension monkey-patch only) |

### 3.3 Success signals

After **full ComfyUI restart** with patches applied (INT8 load path installs patches):

```text
[HSWQ INT8] comfy_quant patches applied (... + LowVramPatch float dtype + Dynamic INT8 LoRA bake + INT8→Nunchaku VRAM handoff)
```

On INT8 → Nunchaku:

```text
[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load (forced INT8 Dynamic unload=…)
```

Must **not** appear:

- Signature A: `0.00 MB usable` / `0.00 MB loaded` with multi-GB offloaded in the same failure pattern  
- `detach failed: TypeError(... unpatch_weights ...)`  
- `[HSWQ PinCache] ACTIVE` as a supposed Abort dependency (module is gone)

### 3.4 Supporting coexistence guards

Same helpers for handoff and bake:

- `_model_is_nunchaku_svdq` — never bake / never treat as HSWQ INT8 target when SVDQ modules present  
- `_model_has_int8_quantized_weights` — `QuantizedTensor` only; bare int8 is not comfy_quant INT8  

---

## 4. Modified file names

| Path | Role |
|------|------|
| `patches/comfy_quant_int8.py` | **Only file** for INT8↔Nunchaku coexistence: detectors, LowVram QT-only, Dynamic bake, force-detach, `load_models_gpu` wrapper |

**Not** part of this Abort fix (and must stay absent for PinCache):

| Path | Role |
|------|------|
| `nodes/hswq_pin_cache.py` | **Deleted** (`4e51074` / historically `5d37ccf`, `d6c87ff`) |
| `nodes/hswq_batched_detailer.py` | No `hswq_pin_cache_scope` wrapper after PinCache removal |

Install copy (when synced):  
`ComfyUI/custom_nodes/ComfyUI-nunchaku-unofficial-loader/patches/comfy_quant_int8.py`.

Cross-doc: PinCache history and HostBuffer supersession → `md/HSWQ_PIN_BUFFER_CACHE_REMOVAL_GUIDE.md`.

---

## 5. Full text of added / modified code

Source of truth: `patches/comfy_quant_int8.py` at tip **`4e51074`** (behavior from `df9ba74` + `747b64b`).  
Version markers: LowVram `_LV_VER = 3`, Dynamic bake `_DYN_VER = 5`, handoff `_VER = 4`.

### 5.1 SVDQ / INT8 detection

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

### 5.2 LowVramPatch — QuantizedTensor only (`_LV_VER = 3`)

```python
def _patch_lowvram_patch_float_intermediate() -> bool:
    """Fix LowVramPatch intermediate_dtype for comfy_quant QuantizedTensor only.

    Must NOT divert bare ``torch.int8`` tensors. Nunchaku SVDQ / Lumina2 use
    int8 storage; grabbing them here corrupts fused CUDA (Abort in
    ``_forward_silu_gating``) even when VRAM handoff already freed GPU memory.
    """
    # ... imports ...
    _LV_VER = 3

    def __call__(self, weight):
        # QuantizedTensor only. Bare int8 / float / None → upstream unchanged.
        if weight is None or not isinstance(weight, QuantizedTensor):
            return true_orig(self, weight)
        patches = (
            self.prepared_patches
            if self.prepared_patches is not None
            else self.patches[self.key]
        )
        w = weight.dequantize()
        dtype = getattr(w, "dtype", None)
        if dtype is not None and hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
            idtype = dtype
        else:
            idtype = torch.float32
        return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype)

    LowVramPatch.__call__ = __call__
    return True
```

### 5.3 Dynamic bake — skip SVDQ; bake QuantizedTensor only (`_DYN_VER = 5`)

Core guards inside bake / wrapped `Dynamic.load`:

```python
    if _model_is_nunchaku_svdq(getattr(patcher, "model", None)):
        return 0
    # ...
            # Bake only comfy_quant QuantizedTensor — never bare int8 (Nunchaku).
            if not isinstance(weight, QuantizedTensor):
                continue
```

```python
        # INT8 LoRA bake only — never touch Nunchaku SVDQ (class is often Lumina2).
        if _model_is_nunchaku_svdq(self.model):
            return result
        if not _model_has_int8_quantized_weights(self.model) and not getattr(
            self.model, "_hswq_int8_baked_keys", None
        ):
            return result
```

### 5.4 Force detach + `load_models_gpu` handoff (`_VER = 4`)

```python
def _force_detach_int8_dynamic_models(device=None, keep_patchers=None) -> int:
    """Fully detach INT8 Dynamic VRAM models (VBAR + hostbufs).

    Same as e3edf77 unidirectional handoff. Only difference from that commit:
    ``detach(unpatch_all=True)`` — ComfyUI ModelPatcher has no ``unpatch_weights``.
    """
    # ... walk current_loaded_models; require is_dynamic() + INT8 QuantizedTensor ...
        try:
            patcher.detach(unpatch_all=True)
        except TypeError:
            try:
                patcher.detach()
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] detach failed: {exc!r}")
        except Exception as exc:
            _console(f"[HSWQ INT8→Nunchaku] detach failed: {exc!r}")
    # ... clear finalizer / pop list / soft_empty_cache ...
    return unloaded


def _patch_load_models_gpu_int8_nunchaku_handoff() -> bool:
    """Before Nunchaku SVDQ load, force-release INT8 Dynamic VRAM occupancy."""
    # v4 = e3edf77 handoff + detach(unpatch_all=True) only. No HostBuffer extras.
    _VER = 4
    # ... wrap mm.load_models_gpu ...
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
        return true_orig(...)
```

### 5.5 Wiring inside `apply_comfy_quant_int8_patches`

```python
    ok_lowvram = _patch_lowvram_patch_float_intermediate()
    ok_dyn_bake = _patch_model_patcher_dynamic_int8_lora_bake()
    ok_handoff = _patch_load_models_gpu_int8_nunchaku_handoff()
    # ...
            f"{' + LowVramPatch float dtype' if ok_lowvram else ''}"
            f"{' + Dynamic INT8 LoRA bake' if ok_dyn_bake else ''}"
            f"{' + INT8→Nunchaku VRAM handoff' if ok_handoff else ''})"
```

Handoff / LowVram / bake install when `apply_comfy_quant_int8_patches()` runs (normally on INT8 HSWQ load). After that, subsequent `load_models_gpu` calls in the process go through the handoff wrapper.

---

## 6. Meaning of the code

### 6.1 `_model_is_nunchaku_svdq`

| Concern | Meaning |
|---------|---------|
| Why scan modules? | Comfy registers Z-Image as **`Lumina2`**; class-name checks for `Nunchaku` / `ZImage` miss it |
| What counts? | Module class contains `SVDQ` / `Nunchaku`, or `ComfyNunchaku*`, or `__module__` contains `nunchaku` |
| Role | Gate handoff trigger, skip Dynamic bake, refuse `_model_has_int8_quantized_weights` on SVDQ graphs |

### 6.2 `_model_has_int8_quantized_weights`

| Concern | Meaning |
|---------|---------|
| True only for | `comfy.quant_ops.QuantizedTensor` on non-SVDQ modules |
| False for | Bare `torch.int8` (common inside Nunchaku) |
| Role | Only **HSWQ / comfy_quant INT8 Dynamic** patchers are force-detached; bake only those weights |

### 6.3 LowVramPatch QT-only

| Step | Meaning |
|------|---------|
| `isinstance(weight, QuantizedTensor)` | Enter custom dequant + float `intermediate_dtype` path |
| Else | Call **upstream** `__call__` unchanged — protects SVDQ bare int8 |

Without this gate, signature **B** Aborts appear even after a successful VRAM handoff.

### 6.4 Dynamic bake QT-only

| Step | Meaning |
|------|---------|
| Skip if SVDQ | Never run comfy_quant bake on Nunchaku |
| Skip non-`QuantizedTensor` | Never bake bare int8 storage |
| Clear LowVram, keep `_v` | Bake via `set_weight`; do not delete VBAR bump slots (FaceDetailer 2nd-load OOM rule — bake hygiene, not PinCache) |

### 6.5 `_force_detach_int8_dynamic_models` + handoff `_VER = 4`

| Step | Meaning |
|------|---------|
| Walk `current_loaded_models` | Find live LoadedModel entries |
| Skip keep / wrong device | Do not detach the Nunchaku about to load |
| Require `is_dynamic()` + INT8 QuantizedTensor | Target INT8 Dynamic only |
| `detach(unpatch_all=True)` | Full release; **correct Comfy API** (not `unpatch_weights`) |
| Clear finalizer / pop list | Remove bookkeeping so `free_memory` sees them gone |
| `free_memory(1e30)` + soft cache | Make room **before** original loader runs |

### 6.6 Why unidirectional only (owner-validated)

- **INT8 → Nunchaku:** force detach INT8 Dynamic + QT-only LowVram/bake → Abort gone. **This works.**  
- **Nunchaku → INT8 “park + pin/VBAR reset + detach SVDQ” (v2):** broke INT8 **reload**. Rolled back; must not reintroduce.

---

## Appendix A — Rejected bidirectional handoff (v2)

### A.1 What v2 added (do not reintroduce)

1. After parking INT8 Dynamic, **reset** `hostbufs_initialized` / HostBuffer / VBAR so a later `Dynamic.load` could recreate pins.  
2. When loading INT8 Dynamic again, **detach Nunchaku SVDQ** first.

### A.2 Why it was cancelled

Owner finding: **reload path broke**; unidirectional handoff is the state that **actually works**.

### A.3 Lesson

Do not “complete” coexistence by symmetrically unloading Nunchaku for INT8 until a reload-safe design is proven. Prefer the minimal fix: clear INT8 Dynamic before SVDQ load + never touch bare Nunchaku int8 in LowVram/bake.

---

## Appendix B — Pin Buffer Cache is not the Abort fix

### B.1 What PinCache was for (v3.1.0 / removal guide)

Historical **Pin Buffer Cache** pooled pin tensors to avoid **destroy `_pin` → re-`cudaHostRegister`** on every Dynamic unload. It was a dual countermeasure with **HSWQ Batched Detailer** against **old pin lifecycle thrash**.

After ComfyUI AIMDO **HostBuffer**, PinCache is **redundant and harmful**. Authoritative removal narrative: `md/HSWQ_PIN_BUFFER_CACHE_REMOVAL_GUIDE.md`, release [v3.2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/v3.2.0). Tree tip after Abort audit: **`4e51074`** removes PinCache again while **keeping** QT-only LowVram + `unpatch_all` handoff.

### B.2 Scope mismatch with INT8→Nunchaku

| Fact | Implication |
|------|-------------|
| PinCache was Detailer-scoped (`hswq_pin_cache_scope`) | INT8→Nunchaku load path does not turn it on |
| No `[HSWQ PinCache] ACTIVE` on Abort workflows that matter here | Cannot be the mechanism that “saved” SVDQ |
| Batched Detailer reduces **model switch count** | Still valuable; **not** a substitute for handoff / QT-only patches |

### B.3 Forbidden framing

Do **not** document PinCache as:

- required for INT8→Nunchaku Abort safety, or  
- primarily “because FaceDetailer thrash” as the reason Abort returned after PinCache deletion.

FaceDetailer was an **amplifier of old pin Register thrash**. Switch count is Batched Detailer’s job. Abort after INT8 Dynamic is **`comfy_quant_int8.py`** (handoff + QT-only).

---

## Appendix C — Audit timeline (correlation ≠ causation)

| Commit | What changed | Abort-relevant truth |
|--------|--------------|----------------------|
| `e3edf77` | INT8→Nunchaku handoff **plus** unauthorized PinCache restore | Real handoff work **bundled** with PinCache → looked like PinCache “fixed” Abort |
| `d6c87ff` | Remove PinCache only | Abort **returned** in retest; logs showed **`detach failed: TypeError(unpatch_weights)`** — broken API already in `e3edf77`; PinCache delete did not invent that TypeError |
| `747b64b` | Handoff uses `detach(unpatch_all=True)` | Fixes signature A detach failure |
| `df9ba74` | LowVram **QuantizedTensor-only** | Fixes signature B fused CUDA Abort |
| `4e51074` | Remove PinCache **again**; keep Abort fixes | Confirms PinCache is not required for coexistence |

**Rule for readers:** if PinCache deletion coincided with Abort, open the log for `detach failed` / LowVram bare-int8 symptoms before blaming pin pooling.

---

## Document control

| Item | Value |
|------|-------|
| Guide path | `md/HSWQ_INT8_NUNCHAKU_COEXISTENCE_GUIDE.md` |
| Code path | `patches/comfy_quant_int8.py` |
| Working fixes | Unidirectional handoff `_VER=4` + LowVram `_LV_VER=3` + Dynamic bake `_DYN_VER=5` |
| Rejected | Bidirectional HostBuffer theater; PinCache as Abort countermeasure |
| Related | `md/HSWQ_PIN_BUFFER_CACHE_REMOVAL_GUIDE.md`, v3.2.0, tip `4e51074` |

End of guide.
