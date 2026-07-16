# SDXL INT8 Normal Generation Break — Handoff False-Positive Fix (Complete Guide)

**Date of record:** 2026-07-16  
**Repository:** `ussoewwin/ComfyUI-nunchaku-unofficial-loader`  
**Primary file:** `patches/comfy_quant_int8.py`  
**Handoff marker after this fix:** `_VER = 10` (`_hswq_int8_nunchaku_handoff_ver`)  
**Live Comfy sync target (owner install):**  
`D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader\patches\comfy_quant_int8.py`

This guide documents **only** the SDXL / native comfy_quant INT8 **normal generation** breakage caused by a **false-positive Nunchaku SVDQ detector** that armed the **INT8→Nunchaku VRAM handoff** on non-SVDQ loads. It is separate from the coexistence Abort guide (`md/HSWQ_INT8_NUNCHAKU_COEXISTENCE_GUIDE.md`), which covers the legitimate INT8 Dynamic → real SVDQ handoff path.

---

## Table of contents

1. [What problem occurred](#1-what-problem-occurred)
2. [Root cause](#2-root-cause)
3. [Countermeasure overview](#3-countermeasure-overview)
4. [Modified file names](#4-modified-file-names)
5. [Full text of modified code](#5-full-text-of-modified-code)
6. [Meaning of the code](#6-meaning-of-the-code)

---

## 1. What problem occurred

### 1.1 Symptom

- **Workflow:** Load an **SDXL HSWQ / comfy_quant INT8** checkpoint and run **normal generation** (e.g. KSampler). No Nunchaku SVDQ / Z-Image model is required for the failure.
- **Output:** Broken images — typically **black** and/or **salt-and-pepper noise** (and related unusable samples).
- **Scope note:** The bug is not “SDXL-only” in mechanism. Any architecture that uses this extension’s INT8 Conv2d (module path contains the substring `nunchaku` from the custom-node folder name) could false-trigger. SDXL INT8 normal gen was the owner’s concrete broken case.

### 1.2 Misleading console signal

When the false positive armed handoff, the console could show the coexistence banner even though the load was **not** a real SVDQ load:

```text
[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load (INT8 Dynamic offload keep-weights=…, no free_memory unpatch)
```

**Interpretation:** That line on a pure SDXL INT8 normal-gen session means the handoff wrapper treated the graph as Nunchaku SVDQ. After the fix, that line must **not** appear on SDXL / native INT8 / non-SVDQ loads.

### 1.3 What was *not* the problem definition

- This incident is about **ordinary generation** with SDXL INT8.
- Do not reframe it as a separate diagnostic topic using owner-forbidden wording.
- Coexistence Abort (`0.00 MB usable` when switching **into** real Nunchaku) is a **different** failure mode; the handoff exists for that path. This guide is about handoff running when it **must not**.

### 1.4 Related damage path (same detector)

The same `_model_is_nunchaku_svdq` gate is used elsewhere (e.g. Dynamic LoRA bake skip for real SVDQ). A false positive can also skew those branches. The **primary** owner-facing break for normal gen was **handoff + `force_full_load` mis-arm**.

---

## 2. Root cause

### 2.1 Intended design (legitimate)

`_patch_load_models_gpu_int8_nunchaku_handoff` wraps `comfy.model_management.load_models_gpu` so that **only when loading real Nunchaku SVDQ** it will:

1. Force-offload resident **INT8 Dynamic** models (`_force_detach_int8_dynamic_models`, keep QT weights / avoid destructive unpatch).
2. `soft_empty_cache`.
3. Call the original loader with **`force_full_load=True`** (avoids Nunchaku starting with ~0 usable VRAM).

That path is correct for **INT8 Dynamic → real SVDQ** coexistence.

### 2.2 Broken gate (direct cause)

Detection used a **bare substring** on each module’s `type(module).__module__`:

```text
"nunchaku" in __module__
```

This extension lives under a custom-node directory whose name contains **`nunchaku`** (e.g. `ComfyUI-nunchaku-unofficial-loader`). INT8 `Conv2d` classes defined in `patches/comfy_quant_int8.py` therefore carry a `__module__` path that also contains **`nunchaku`**, even though they are **not** the real `nunchaku` Python package and **not** SVDQ kernels.

**Chain:**

1. SDXL INT8 loads → graph contains this extension’s INT8 Conv2d.  
2. `_model_is_nunchaku_svdq` → **True** (false positive).  
3. Handoff wrapper sets `need_handoff = True`.  
4. INT8 Dynamic force-detach + `force_full_load=True` run on a **non-SVDQ** load.  
5. Normal SDXL INT8 generation is corrupted (black / noise). Older handoff variants that leaned on `free_memory` / `unpatch_weights=True` were especially lethal to non-SVDQ INT8 (see comments in-tree).

### 2.3 Why “string contains nunchaku” is illegal here

| Path kind | Example | Must count as SVDQ? |
|---|---|---|
| Real Nunchaku package | `nunchaku.…` / `….nunchaku.…` | Yes |
| This unofficial loader / INT8 patch | `…nunchaku-unofficial-loader…` / `…comfy_quant_int8…` | **No** |

Substring match cannot tell these apart.

### 2.4 Human / process cause (recorded)

The coexistence handoff was attached to a **sloppy global detector** and allowed to fire on every `load_models_gpu` without a hard “native INT8 → never handoff” branch. Regression on **SDXL INT8 normal generation** was not treated as the primary definition of the bug until the owner forced that definition.

---

## 3. Countermeasure overview

### 3.1 Detection fix

Add **`_module_path_is_real_nunchaku_package(mod)`**:

- Reject paths containing `unofficial`, `comfy_quant_int8`, `nunchaku-unofficial`, `nunchaku_unofficial`.
- Accept only real package forms: `nunchaku`, `nunchaku.*`, or `*.nunchaku.*`.
- **Forbid** bare `"nunchaku" in path`.

Wire `_model_is_nunchaku_svdq` to use this helper for `__module__` checks. Class-name hits (`SVDQ` / `Nunchaku` / `ComfyNunchaku*`) remain.

### 3.2 Handoff arming fix (`_VER = 10`)

Inside the `load_models_gpu` wrapper, for each ModelPatcher’s **BaseModel** (`m.model`):

| Branch | Condition | Action |
|---|---|---|
| **A** | `_model_has_int8_quantized_weights(base)` (native comfy_quant INT8, any arch) | **Never** arm handoff (`continue`) |
| **B** | `_model_is_nunchaku_svdq(base)` (real SVDQ only) | Arm handoff |
| else | FP / other | Pass through; no handoff |

Do **not** probe the ModelPatcher object alone for SVDQ (avoids extra false positives).

### 3.3 Owner boundary

- Not “exclude SDXL only.”
- **No handoff on anything that is not real Nunchaku SVDQ.**
- After Comfy restart: SDXL INT8 normal gen must **not** log `[HSWQ INT8→Nunchaku] VRAM handoff…`.

### 3.4 Install sync

Repo file and the live Comfy custom_nodes copy must match. Only valid sync target for this owner:

`D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader\`

---

## 4. Modified file names

| Path | Role |
|---|---|
| `patches/comfy_quant_int8.py` | **Only code file** for this repair (detector + handoff `_VER = 10`) |
| Live mirror (must match) | `D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader\patches\comfy_quant_int8.py` |
| This guide | `md/HSWQ_SDXL_INT8_HANDOFF_FALSE_POSITIVE_FIX_GUIDE.md` |

No other production modules were required for this specific false-positive fix.

---

## 5. Full text of modified code

The following blocks are the **repair surface** as present in `patches/comfy_quant_int8.py` after the fix (line numbers may shift; content is authoritative). Included:

1. `_module_path_is_real_nunchaku_package` (new gate)
2. `_model_is_nunchaku_svdq` (uses the gate)
3. `_model_has_int8_quantized_weights` (Branch A dependency; unchanged contract, listed because handoff Branch A calls it)
4. `_force_detach_int8_dynamic_models` (executed only when handoff arms; keep-weights / no destructive unpatch)
5. `_patch_load_models_gpu_int8_nunchaku_handoff` (`_VER = 10`, Branches A/B)

```python
def _module_path_is_real_nunchaku_package(mod: str) -> bool:
    """True only for real Nunchaku package modules — never this unofficial-loader.

    INT8 Conv2d from this extension lives under a path containing ``nunchaku``;
    a bare ``\"nunchaku\" in path`` false-positive armed VRAM handoff on
    non-SVDQ loads (SDXL INT8 and any other architecture using those Conv2d)
    and destroyed normal generation. Substring match is forbidden.
    """
    mod_l = (mod or "").lower().replace("\\", "/")
    if not mod_l:
        return False
    # This extension / INT8 patch path must never count as SVDQ.
    if (
        "unofficial" in mod_l
        or "comfy_quant_int8" in mod_l
        or "nunchaku-unofficial" in mod_l
        or "nunchaku_unofficial" in mod_l
    ):
        return False
    if mod_l == "nunchaku" or mod_l.startswith("nunchaku."):
        return True
    if ".nunchaku." in mod_l:
        return True
    return False


def _model_is_nunchaku_svdq(model) -> bool:
    """True only when the graph carries real Nunchaku SVDQ modules.

    ComfyUI registers Z-Image as ``Lumina2`` — classname checks for
    ``Nunchaku`` / ``ZImage`` miss that. Any SVDQ / ComfyNunchaku module means
    never run comfy_quant INT8 Dynamic LoRA bake.

    Branch: everything that is not real SVDQ (SDXL, Flux, ZIT, native INT8,
    FP, …) returns False. Module-path checks must not match this
    unofficial-loader package (see ``_module_path_is_real_nunchaku_package``).
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
            if _module_path_is_real_nunchaku_package(mod):
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


def _force_detach_int8_dynamic_models(device=None, keep_patchers=None) -> int:
    """Offload INT8 Dynamic VRAM (VBAR + hostbufs) without destroying QT weights.

    free_memory often stops after partially_unload and leaves HostBuffers, so
    Nunchaku sees ``0.00 MB usable`` and Aborts. We must fully offload INT8
    Dynamic models before SVDQ load.

    Critical: use ``unpatch_weights=False`` / ``detach(unpatch_all=False)``.
    ``unpatch_all=True`` unpatches INT8 QuantizedTensor + baked LoRA and causes
    black / noise on the next normal SDXL INT8 KSampler.
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
        # Preserve QT + baked LoRA; only free GPU / VBAR occupancy.
        try:
            lm.model_unload(unpatch_weights=False)
        except TypeError:
            try:
                patcher.detach(unpatch_all=False)
            except TypeError:
                try:
                    patcher.detach(False)
                except Exception as exc:
                    _console(f"[HSWQ INT8→Nunchaku] detach(False) failed: {exc!r}")
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] detach(unpatch_all=False) failed: {exc!r}")
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
        except Exception as exc:
            _console(f"[HSWQ INT8→Nunchaku] model_unload(False) failed: {exc!r}")
            i += 1
            continue
        mm.current_loaded_models.pop(i)
        unloaded += 1
    if unloaded > 0:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    return unloaded


def _patch_load_models_gpu_int8_nunchaku_handoff() -> bool:
    """Before Nunchaku SVDQ load, offload INT8 Dynamic VRAM without unpatch."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return False

    original = getattr(mm, "load_models_gpu", None)
    if original is None:
        return False
    # v10 = handoff arms ONLY for real Nunchaku SVDQ. All other loads
    # (SDXL / Flux / ZIT / native INT8 / FP / …) pass through untouched.
    # free_memory → model_unload(unpatch_weights=True) kills non-SVDQ INT8.
    _VER = 10
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
            # Branch A: native comfy_quant INT8 (any architecture) — never handoff.
            if base is not None and _model_has_int8_quantized_weights(base):
                continue
            # Branch B: only real Nunchaku SVDQ on the BaseModel arms handoff.
            # Do not probe the ModelPatcher itself (false positives).
            if base is not None and _model_is_nunchaku_svdq(base):
                need_handoff = True
                if device is None:
                    device = getattr(m, "load_device", None)
        if need_handoff:
            n = _force_detach_int8_dynamic_models(device=device, keep_patchers=keep)
            # Second pass: any INT8 Dynamic still listed (missed first pass) —
            # never leave them for free_memory(unpatch=True).
            n2 = _force_detach_int8_dynamic_models(device=None, keep_patchers=keep)
            try:
                mm.soft_empty_cache()
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] soft_empty_cache failed: {exc!r}")
            _console(
                f"[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load "
                f"(INT8 Dynamic offload keep-weights={n + n2}, no free_memory unpatch)"
            )
        return true_orig(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            # Full load after handoff avoids Nunchaku 0.00 MB usable Abort.
            # Non-SVDQ loads never set need_handoff (branches A/B above).
            force_full_load=True if need_handoff else force_full_load,
        )

    load_models_gpu._hswq_int8_nunchaku_handoff = True
    load_models_gpu._hswq_int8_nunchaku_handoff_ver = _VER
    load_models_gpu._hswq_orig_load_models_gpu = true_orig
    mm.load_models_gpu = load_models_gpu
    return True
```

---

## 6. Meaning of the code

### 6.1 `_module_path_is_real_nunchaku_package`

| Piece | Meaning |
|---|---|
| Reject `unofficial` / `comfy_quant_int8` / `nunchaku-unofficial` | This custom node’s INT8 Conv2d must **never** look like SVDQ. |
| Accept `nunchaku` / `nunchaku.*` / `*.nunchaku.*` | Only real installed Nunchaku package modules. |
| No bare substring `"nunchaku" in` | Folder name pollution is not evidence of SVDQ. |

### 6.2 `_model_is_nunchaku_svdq`

Walks BaseModel / `diffusion_model` / nested `model` and returns True only if:

- a module **class name** looks like SVDQ / Nunchaku / ComfyNunchaku, **or**
- `__module__` passes `_module_path_is_real_nunchaku_package`.

Everything else (SDXL INT8, Flux, ZIT, native INT8, FP, …) returns **False**.

### 6.3 `_model_has_int8_quantized_weights`

True only for native **`comfy.quant_ops.QuantizedTensor`** weights. Used as **Branch A**: “this load is native INT8 → do not run Nunchaku handoff.” Also refuses to treat real SVDQ graphs as comfy_quant INT8 (avoids baking the wrong path).

### 6.4 `_force_detach_int8_dynamic_models`

Legitimate coexistence tool: remove INT8 **Dynamic** residency from GPU **without** destroying QuantizedTensor + baked LoRA (`unpatch_weights=False` / `detach(unpatch_all=False)`).  

**Must run only when Branch B armed handoff.** If it runs because of a false SVDQ detection, it participates in breaking the next / current INT8 normal gen. Docstring explicitly records that `unpatch_all=True` causes black/noise on the next SDXL INT8 KSampler.

### 6.5 `_patch_load_models_gpu_int8_nunchaku_handoff` (`_VER = 10`)

| Piece | Meaning |
|---|---|
| `_VER = 10` | Idempotent re-patch marker; ensures the Branch A/B logic is installed once per process. |
| Branch A `continue` | Native INT8 load → **no** detach, **no** handoff log, **no** forced `force_full_load`. |
| Branch B `need_handoff = True` | Real SVDQ on BaseModel only → detach INT8 Dynamic, then load with `force_full_load=True`. |
| Probe `m.model` only | Avoids ModelPatcher-level false positives. |
| Console line | Should appear **only** on real SVDQ loads after INT8 coexistence handoff. |

### 6.6 Verification checklist (owner)

1. Restart ComfyUI so `_VER = 10` is applied.  
2. Load SDXL INT8 and run **normal** generation.  
3. Confirm **no** `[HSWQ INT8→Nunchaku] VRAM handoff` line.  
4. Confirm sample is not black / salt-and-pepper from this failure mode.  
5. Separately (coexistence): INT8 Dynamic then real Nunchaku SVDQ may still show the handoff line — that is expected.

---

## Related documents

- Coexistence Abort / legitimate handoff narrative: `md/HSWQ_INT8_NUNCHAKU_COEXISTENCE_GUIDE.md`  
- Pin Buffer Cache removal (not this bug): `md/HSWQ_PIN_BUFFER_CACHE_REMOVAL_GUIDE.md`  
- Incident / speech / “past maximum” record (owner): `反省文/` (not a substitute for this technical guide)

---

**End of guide.**
