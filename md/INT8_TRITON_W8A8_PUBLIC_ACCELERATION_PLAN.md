# INT8 Triton W8A8 Public Acceleration — Implementation Plan

This document is the implementation plan for delivering **INT8 Linear (W8A8) Triton speed** through this custom node for **public / worldwide** ComfyUI users.

Date: 2026-07-18  
Repository: `ComfyUI-nunchaku-unofficial-loader`

Primary strategy: **Plan B** (ship Triton INT8 GEMM kernels inside this extension and call them from the `QuantizedTensor` / `int8_tensorwise` Linear path).  
Rejected as the speed guarantee: **Plan A alone** (relying on ComfyUI `comfy_kitchen` Triton + `--enable-triton-backend`).

---

## 1. Goal

| Requirement | Meaning |
|-------------|---------|
| Speed for public users | Users who install this node via Manager / `install.py` / `requirements.txt` must be able to hit the **fused Triton INT8 Linear** path without depending on Comfy launch flags or a special local fork. |
| Same kernel class as INT8-Fast / forge-classic-neo | Row-wise activation quant + INT8 matmul + dequant (scalar weight scale and per-row scale variants), with autotune. |
| Safe coexistence | Must not break HSWQ FP8, Nunchaku SVDQ, Dynamic VRAM handoff, LoRA bake, or bare-int8 false-positive gates. |
| Installable environment | `install.py` must install the **Triton runtime** appropriate for the OS so the kernels can actually compile and run. |

Success criterion: on a CUDA GPU with SM ≥ 8.0 (Ampere+), after a normal node install, INT8 SDXL / Z-Image Linear layers use the extension’s Triton path and match INT8-Fast / kitchen fused-kernel latency class (same algorithm). Eager / `_int_mm` remain fallbacks only when Triton is unavailable or for tiny `M`.

---

## 2. Why Plan A alone is not the public speed guarantee

| Fact | Consequence for worldwide users |
|------|----------------------------------|
| ComfyUI keeps kitchen Triton **off by default** unless `--enable-triton-backend` | Most Manager / portable users never pass that flag → kitchen stays eager → **no speed**. |
| Kitchen CUDA ops often need a matching torch / CUDA kitchen build (e.g. cu130) | Many installs cannot enable kitchen even with the flag. |
| Triton may be missing on Windows | Without a Windows Triton wheel, kitchen and any Triton kernel fail open to eager. |
| Speed when both hit the same fused kernels is ≈ equal | A is not “slower by design”; A is **unreliable to reach** for the public. |

Plan A (optional kitchen enable) may remain a **bonus** if the user’s Comfy already has kitchen Triton on. It is **not** the delivery mechanism this plan relies on.

---

## 3. Plan B — runtime architecture

### 3.1 Reference kernels (read-only sources)

| Source | Role |
|--------|------|
| `ComfyUI-INT8-Fast` `int8_fused_kernel.py` | Primary kernel recipe: `_quantize_rowwise_kernel`, `_int8_matmul_dequant_kernel`, `_int8_matmul_dequant_per_row_kernel`, Python wrappers + autotune configs. |
| forge-classic-neo `comfy_kitchen` INT8 linear | Same kernel family; confirms recipe used in production INT8 paths. |

### 3.2 Wiring inside this repo

| Piece | Responsibility |
|-------|----------------|
| New module e.g. `patches/int8_triton_kernels.py` (or `triton_ops/`) | Own the Triton kernels and `triton_int8_linear` / `triton_int8_linear_per_row` wrappers. Import `triton` lazily; expose `is_triton_int8_available()`. |
| `patches/comfy_quant_int8.py` | On the **comfy_quant / int8_tensorwise QuantizedTensor Linear.forward** path only: call Triton wrappers when available; else existing eager / `torch._int_mm` behavior. |
| Loaders / MixedPrecisionOps | No change to weight packing format; only accelerate the Linear forward that already receives INT8 weights + scales. |
| SVDQ / handoff / bare-int8 gates | Unchanged. Triton must **never** arm on Nunchaku SVDQ modules or false-positive `"nunchaku" in __module__` paths. |

### 3.3 Forward decision order (Linear)

1. Not `int8_tensorwise` QuantizedTensor Linear → do not touch.  
2. Triton import / compile / capability check fails → eager or `_int_mm` (current hardening path).  
3. Very small `M` (e.g. `M ≤ 16`, same class as INT8-Fast) → dequant + `F.linear` (avoid Triton launch overhead).  
4. Otherwise → fused Triton INT8 Linear (scalar scale or per-row scale as weight metadata dictates).

### 3.4 Non-goals

- No global `F.linear` monkeypatch.  
- No Triton attention rewrite in this phase.  
- No requirement that users pass `--enable-triton-backend`.  
- No specialization to a single developer machine’s Comfy tree.

---

## 4. `install.py` — environment that can deliver speed

Speed is not only kernels in git. Without a working **Triton** package in the **same** Python as ComfyUI, Plan B falls back and users see no acceleration. This plan therefore makes **`install.py` responsible for installing Triton** after the existing requirements path.

### 4.1 Current `install.py` (keep)

Existing behavior must remain:

1. Upgrade `pip` / `setuptools` / `wheel` (fixes Python 3.12 `pkgutil.ImpImporter` / `filterpy` sdist failures — see `md/REQUIREMENTS_INSTALL_PKGUTIL_IMPIMPORTER_FIX_GUIDE.md`).  
2. `pip install -r requirements.txt`.

### 4.2 New stage: Triton for INT8 acceleration

After requirements succeed, `install.py` runs a dedicated Triton install stage:

| Platform | Action |
|----------|--------|
| **Windows** | Uninstall stock `triton` if present (often unusable on native Windows). Install **`triton-windows`** from PyPI (public wheels; pin with an upper bound compatible with common torch builds, e.g. `triton-windows<3.7` unless a tighter pin is required by measured torch version). |
| **Linux** | Ensure `triton` is importable. If missing, `pip install triton` (many torch CUDA wheels already pull it; install only when needed). |
| **macOS / no CUDA** | Skip Triton install; print a clear message that INT8 Triton acceleration requires NVIDIA CUDA. |

### 4.3 Capability probe (same `sys.executable`)

After install attempts:

1. `import triton` must succeed in a subprocess or in-process check using **this** interpreter.  
2. Optionally probe `torch.cuda.is_available()` and log GPU / CUDA availability (do not fail the whole custom-node install solely because the machine is CPU-only).  
3. Print an explicit banner:

   - `INT8 Triton speed path: READY` when `import triton` works.  
   - `INT8 Triton speed path: UNAVAILABLE — falling back to eager/_int_mm` when it does not.

### 4.4 Failure policy

| Case | Exit / behavior |
|------|-----------------|
| `requirements.txt` fails | Non-zero exit (unchanged). |
| Triton install fails on CUDA Windows/Linux | Log loud error + remediation commands; **do not** undo successful requirements. Prefer exit `0` for Manager compatibility, with unmistakable “no speed until Triton is fixed” text. |
| Triton install succeeds but import fails | Same loud remediation (VC++ redistributable / include+libs for embedded Python on Windows, etc.). |

### 4.5 What `install.py` must not do

- Must not call ComfyUI with `--enable-triton-backend` (Plan A is not the guarantee).  
- Must not `pip install` a second global Python; always `sys.executable -m pip`.  
- Must not add Triton to `requirements.txt` as a blind cross-platform pin (Windows needs `triton-windows`, not stock `triton`). OS branching belongs in **`install.py`**.

### 4.6 Optional docs / Manager notes

README or install notes may point users to:

- Windows: `python.exe -m pip install -U "triton-windows<3.7"`  
- Linux: `python -m pip install triton`  
- Embedded portable: always use that tree’s `python_embeded\python.exe -m pip …`

Kernel code still ships in the node; install only supplies the compiler/runtime.

---

## 5. Implementation phases

| Phase | Work | Done when |
|-------|------|-----------|
| **P0** | Extend `install.py` with OS-aware Triton install + import probe + clear READY/UNAVAILABLE logs | Fresh Manager / portable install on Windows and Linux either READY or documents why not |
| **P1** | Port INT8-Fast fused kernels into this repo module; unit-smoke import without Comfy | Module imports; no Triton → soft disable |
| **P2** | Wire into `patches/comfy_quant_int8.py` Linear path for `int8_tensorwise` only | INT8 workflow uses Triton when READY; SVDQ / FP8 unchanged |
| **P3** | Fallbacks: no Triton, `M` small, compile error → eager / `_int_mm` | Never hard-crash the graph for missing Triton |
| **P4** | Optional: if kitchen Triton already enabled, leave as-is (no conflict); do not depend on it | Public path still works with kitchen off |
| **P5** | Public md / README short “INT8 speed requires Triton; install.py installs it” | Users know READY vs UNAVAILABLE |

---

## 5.1 UI — Triton accelerate on/off toggle (exactly two nodes)

| Node class | Title |
|------------|--------|
| `HSWQFP8E4M3UNetLoader` | HSWQ FP8 E4M3/INT8 UNet Loader |
| `NunchakuUssoewwinCheckpointLoaderSDXL` | HSWQ Checkpoint Loader (SDXL) |

UI shape (same pattern as SDXL LoRA Dynamic `enabled_*` / `debug`):

1. Backend value channel: `triton_accelerate` as `BOOLEAN` (serialization / graph JSON).
2. Frontend: `js/hswq_triton_accelerate_toggle.js` sets `widget.type = "toggle"` and label **Triton accelerate** on both nodes (`WEB_DIRECTORY = "js"`).

**ON** (default): when loading INT8 (`int8_tensorwise` or auto-detected INT8 checkpoint), prefer Triton fused INT8 Linear if Triton is READY.  
**OFF**: force eager / `_int_mm` (no Triton Linear), even if Triton is installed.

Do **not** use combo / button / socket-only UI. Do **not** add this toggle to any other node.

Stamp: `model.model_options["hswq_triton_accelerate"]` (and load helpers in `patches/comfy_quant_int8.py`).

---

## 6. Files to touch (expected)

| Path | Change |
|------|--------|
| `install.py` | Triton stage (this plan’s environment guarantee). |
| `patches/int8_triton_kernels.py` (new) | Kernels + availability helpers. |
| `patches/comfy_quant_int8.py` | Call Triton on INT8 Linear forward only; honor `hswq_triton_accelerate`. |
| `hswq/zimage_fp8_e4m3_unet.py` / `__init__.py` | `triton_accelerate` BOOLEAN on the two loaders only. |
| `js/hswq_triton_accelerate_toggle.js` | Force on/off **toggle** widget on those two nodes. |
| `patches/__init__.py` / node package init | Ensure patch module loads once. |
| `requirements.txt` / `pyproject.toml` | **Do not** put Windows-incompatible `triton` as a flat dep; keep OS logic in `install.py`. |
| README / short install note | Point to Triton READY message. |

---

## 7. Safety checklist (must pass before release)

```
□ Triton path only for real comfy_quant int8_tensorwise QuantizedTensor Linear
□ No handoff arming on unofficial-loader INT8 / non-SVDQ
□ No global F.linear patch
□ FP8 / SVDQ / LoRA bake / Dynamic VRAM paths unchanged in behavior
□ install.py still upgrades setuptools before requirements (ImpImporter fix)
□ Windows uses triton-windows; Linux uses triton
□ Missing Triton → fallback, not crash
□ Speed claim tied to READY probe, not to kitchen flag
```

---

## 8. Verification (public-user oriented)

1. Clean portable / venv ComfyUI (no prior Triton).  
2. Install this custom node so Manager runs `install.py`.  
3. Confirm log: `INT8 Triton speed path: READY`.  
4. Run SDXL or Z-Image HSWQ INT8 workflow; confirm console / debug that Triton Linear is used (not only `_int_mm`).  
5. Uninstall Triton and re-run: must fall back without breaking generation.  
6. Confirm Nunchaku SVDQ workflow still loads and handoff gates still correct.

---

## 9. Summary

| Decision | Choice |
|----------|--------|
| Public speed strategy | **Plan B** — extension-owned Triton INT8 Linear kernels |
| Plan A (kitchen + flag) | Optional bonus only; **not** the guarantee |
| Environment for speed | **`install.py` installs Triton** (`triton-windows` on Windows, `triton` on Linux) and probes `import triton` |
| Fallback | Eager / `_int_mm` / small-`M` dequant linear |

This plan is complete only when both the **kernels** and the **`install.py` Triton stage** ship together. Kernels without Triton install do not deliver public speed; Triton install without kernels does not accelerate this node’s INT8 path.
