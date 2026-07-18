# Krea2 INT8 — Implementation Plan

Date: 2026-07-19  
Repository: `ussoewwin/ComfyUI-nunchaku-unofficial-loader`  
Primary goal: **end-to-end Krea2 INT8 (including `*Int8Convrot*`) in ComfyUI via this custom node**, before Forge-Nunchaku follow-up.

**Authoritative reference (read-only):** `D:\ComfyUI-INT8-Fast` / [BobJohnson24/ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)

Related in-repo docs:

- `md/INT8_TRITON_W8A8_PUBLIC_ACCELERATION_PLAN.md` — Triton W8A8 for non-ConvRot Linear  
- `md/HSWQ_INT8_AND_LORA_TECHNICAL_GUIDE.md` — existing SDXL / Z-Image INT8 path  

---

## 1. Goal

| Requirement | Meaning |
|-------------|---------|
| ComfyUI first | Complete load + sample of Krea2 DiT INT8 in ComfyUI using this repo. Forge Issue #3 is secondary until this works. |
| Pre-quantized ConvRot | Load checkpoints such as `*Int8Convrot*.safetensors` (`int8_tensorwise` + `comfy_quant.convrot=true`) **without** exploding system RAM. |
| Keep SDXL / Z-Image | Existing `int8_tensorwise` + Triton path for SDXL / Z-Image must stay green. Krea2 work is **a separate branch**, never a rewrite of the old path. |
| Match INT8-Fast architecture class | Own Linear ops + own ConvRot (Hadamard) + Triton / `_int_mm` — **not** dependency on kitchen `TensorCoreConvRotW4A4Layout`. |
| Hard isolation | New Krea2 / ConvRot / W8A8 Ops path must **not** alter SDXL Checkpoint / Z-Image UNet / MixedPrecision dispatch behavior. See **§5.1**. |

**Success criterion:** On ~16 GB VRAM / ~64 GB system RAM class hardware, select a real Krea2 INT8 ConvRot UNet + Wan VAE + Qwen3-VL-4B (`CLIPLoader` type `krea2`), run KSampler to completion, without kitchen ConvRot import failure forcing full float dequant of the DiT.

---

## 2. Current state (this repo)

| Area | Status |
|------|--------|
| SDXL INT8 | Working (`NunchakuUssoewwinCheckpointLoaderSDXL` + `install_int8_option_dispatch`) |
| Z-Image INT8 | Working (`HSWQFP8E4M3UNetLoader` + MixedPrecision / `comfy_quant_int8` patches) |
| Triton W8A8 | Present for **non-ConvRot** `TensorWiseINT8Layout` Linear (`patches/comfy_quant_int8.py` skips `params.convrot=True`) |
| Krea2 name / nodes / exclusions | **Absent** (repo-wide `krea` / `Krea` / `SingleStreamDiT` = zero hits) |
| Own ConvRot module | **Absent** |
| Own `Int8TensorwiseOps` as `custom_operations` | **Absent** (uses Comfy MixedPrecision + kitchen / QuantizedTensor) |
| Layout registry fallback for missing kitchen | **Absent** (Forge has `patch_comfy_quant_layout_fallback.py`; this repo does not) |

ComfyUI **core** already defines Krea2 (`SingleStreamDiT`, `CLIPType.KREA2`, Wan VAE). This plan does **not** reimplement the DiT architecture. It adds the **INT8 + ConvRot load / forward** path that SDXL / Z-Image never required.

---

## 3. Why kitchen-only path fails for Krea2 INT8 ConvRot

Issue reports on Forge (Krea2 INT8) show:

1. `Failed to import comfy_kitchen` — missing `TensorCoreConvRotW4A4Layout`  
2. Quant layouts unavailable → crash or float dequant fallback  
3. Full dequant + `--pin-shared-memory` → **system RAM blow-up** (~64 GB)

INT8-Fast already solved the product problem differently:

- Register a minimal `Int8TensorwiseLayout` in-process  
- Load with `model_options={"custom_operations": Int8TensorwiseOps}`  
- Store INT8 weights as plain `nn.Parameter` + `weight_scale` buffers  
- Apply ConvRot in **forward** via local `convrot.py` (activation rotate), not kitchen ConvRot tensor layouts  

This plan adopts that class of solution for Krea2 (and optionally any ConvRot INT8 UNet loaded through the new path).

---

## 4. Reference map (INT8-Fast → this repo)

| INT8-Fast file | Role | Planned destination in this repo |
|----------------|------|----------------------------------|
| `__init__.py` `_register_layouts()` | Register `Int8TensorwiseLayout` + `QUANT_ALGOS["int8_tensorwise"]` | New: `patches/int8_fast_ops.py` (or `patches/krea2_int8/`) init hook called from `__init__.py` |
| `convrot.py` | Hadamard build, `rotate_weight`, `rotate_activation` | New: `patches/convrot.py` (port; keep MIT attribution) |
| `int8_quant.py` `Int8TensorwiseOps` | Custom Linear load + forward + ConvRot + LoRA bake hooks | New: `patches/int8_tensorwise_ops.py` (port / adapt; share Triton with existing `int8_triton_kernels.py` where possible) |
| `int8_unet_loader.py` `UNetLoaderINTW8A8` | Loader node; `model_type` including **`krea2`** exclusions | **New node only** (e.g. `HSWQINT8W8A8UNetLoader`). Do **not** extend `HSWQFP8E4M3UNetLoader` or SDXL Checkpoint loaders |
| `int8_fused_kernel.py` | Triton W8A8 (scalar + **per_row**) | Prefer existing `patches/int8_triton_kernels.py` (`triton_int8_linear` + `triton_int8_linear_per_row`); wire Ops forward by `weight_scale` shape |
| `int8_lora.py` / Pre-Lora / `INT8ModelPatcher` | LoRA bake + patcher wrap | Phase 3 — do not block Krea2 DiT smoke |
| `int8_save.py` `INT8ModelSave` | Export INT8 + `comfy_quant` (incl. convrot flags) | Phase 3+ optional product node |
| `convert_to_comfy.py` | Old `.comfy_quant` JSON → native `format: int8_tensorwise` | Phase 0/2: document or ship helper for circulating INT8-Fast-era files |

### 4.1 Krea2 sensitive-layer exclusions (copy from INT8-Fast)

On-the-fly quant and / or keep-FP16 for:

```text
first, last, tmlp, tproj, txtfusion, txtmlp
```

Source: `int8_unet_loader.py` `model_type == "krea2"`.

Pre-quantized ConvRot checkpoints still need the same exclusion list only when OTF is enabled; for prequant load, exclusions mainly protect accidental re-quant of sensitive prefixes if mixed state_dicts appear.

---

## 5. Architecture (target)

```text
[User] Load Diffusion Model (INT8 / W8A8) — model_type=krea2, enable_convrot=ON
          │
          ▼
  Register Int8TensorwiseLayout (idempotent)
  Set Int8TensorwiseOps flags (excluded_names, enable_convrot, Triton)
          │
          ▼
  comfy.sd.load_diffusion_model[_state_dict](
        path, model_options={"custom_operations": Int8TensorwiseOps}
  )
          │
          ▼
  Linear._load_from_state_dict:
    - parse layer `*.comfy_quant` JSON bytes (convrot / convrot_groupsize / format / optional per_row)
    - int8 weight + weight_scale → keep INT8 Parameter (no float bake)
    - weight_scale numel==1 → scalar path; shape (out,1) → per_row path
    - pop/ignore `input_scale` if present (cleanup only)
    - comfy_quant.convrot → _use_convrot + groupsize (default groupsize **256**)
          │
          ▼
  Linear.forward:
    - if _use_convrot: rotate_activation(x) with stamped groupsize (in_features % groupsize == 0)
    - Triton / _int_mm: scalar vs per_row by weight_scale shape
    - else: F.linear dequant fallback (tiny M / non-quant layers only)
          │
          ▼
  [Comfy] CLIPLoader type=krea2 (Qwen3-VL-4B) + Wan VAE + KSampler
```

**Hard rule:** Do not route Krea2 ConvRot through “missing kitchen → dequant entire UNet to bf16” as the default success path. That is the RAM failure mode. Dequant-per-layer remains only for tiny-batch or non-quant layers.

### 5.1 Hard isolation — two INT8 branches (do not merge)

This repository already has a working INT8 stack for SDXL / Z-Image. Krea2 ConvRot must be a **second branch**, not a patch of the first.

```text
                    ┌─────────────────────────────────────────────┐
  EXISTING (keep)   │ SDXL Checkpoint / HSWQFP8E4M3UNetLoader     │
                    │   → install_int8_option_dispatch            │
                    │   → load_unet_hswq_weight_dtype             │
                    │   → apply_comfy_quant_int8_patches()        │
                    │   → MixedPrecisionOps + TensorWiseINT8Layout│
                    │   → Triton on non-ConvRot only              │
                    │   (comfy_quant_int8.py skips params.convrot)│
                    └─────────────────────────────────────────────┘
                                      ▲
                                      │  FORBIDDEN: auto-detect ConvRot /
                                      │  Krea2 into this path; FORBIDDEN:
                                      │  rewrite load_unet_hswq_* for Krea2
                                      │
                    ┌─────────────────────────────────────────────┐
  NEW (additive)    │ HSWQINT8W8A8UNetLoader (new node)           │
                    │   → Int8TensorwiseOps as custom_operations  │
                    │   → Int8TensorwiseLayout (INT8-Fast name)   │
                    │   → patches/convrot.py in Linear.forward    │
                    │   → Triton import-only from existing kernels│
                    │   → model_type=krea2 exclusions             │
                    └─────────────────────────────────────────────┘
```

| Rule | Existing route (SDXL / Z-Image) | New route (Krea2 / ConvRot W8A8) |
|------|----------------------------------|-----------------------------------|
| Entry node | `NunchakuUssoewwinCheckpointLoaderSDXL`, `HSWQFP8E4M3UNetLoader` | **New** `HSWQINT8W8A8UNetLoader` (name TBD; INT8-Fast twin) |
| Ops | MixedPrecision via `comfy_quant_int8` patches | `Int8TensorwiseOps` via `model_options["custom_operations"]` only |
| Layout string | Kitchen / Comfy `TensorWiseINT8Layout` (unchanged) | Separate `Int8TensorwiseLayout` (`setdefault` only; never overwrite kitchen) |
| ConvRot | Skipped (`params.convrot=True` → no Triton in old path) | Handled in new Ops forward |
| Shared code allowed | — | **Read-only import** of `patches/int8_triton_kernels.py` (`triton_int8_linear` / `_per_row`) |
| Shared code forbidden | — | Editing `comfy_quant_int8.py` / `install_int8_option_dispatch` / `load_unet_hswq_weight_dtype` to “also do Krea2” |
| Auto-detect | INT8 probe stays for SDXL/ZI only | ConvRot / Krea2 files must **not** be redirected into `HSWQFP8E4M3UNetLoader` by filename heuristics |
| Handoff / SVDQ gates | Existing false-positive rules unchanged | New Ops must still be non-SVDQ for handoff probes |

**Phase 1–2 forbidden edits (unless a later order names the exact file):**

- `patches/comfy_quant_int8.py` — no Krea2 / ConvRot / custom_operations merge  
- `install_int8_option_dispatch` / `load_unet_hswq_weight_dtype` — no new branches for Krea2  
- `HSWQFP8E4M3UNetLoader.load_unet` signature / UI — do not add `model_type` / `enable_convrot` here  
- `NunchakuUssoewwinCheckpointLoaderSDXL` / ZIT Checkpoint loaders — untouched  
- Overwriting `QUANT_ALGOS["int8_tensorwise"]` or kitchen layout classes already registered for the old path  

**Phase 1–2 allowed touch list:**

- New: `patches/convrot.py`, `patches/int8_tensorwise_ops.py`, new loader module under `hswq/` or `nodes/`  
- `__init__.py` — **register the new node only** (+ idempotent `setdefault` layout register for the new layout name)  
- Optional: import Triton helpers from existing `int8_triton_kernels.py` without changing their public API for SDXL/ZI  

**Product decision (locked):** Prefer **A only** — dedicated W8A8 loader node. Option **B** (extend `HSWQFP8E4M3UNetLoader`) is **rejected** for Phase 1–2 because it risks regressing FP8 / Z-Image / SDXL INT8 UI and dispatch.

---

## 6. Phased work

### Phase 0 — Preconditions (verify, no product code yet)

1. Confirm host ComfyUI build has `comfy.ldm.krea2`, `CLIPType.KREA2`, Wan VAE.  
2. Confirm a real `*Int8Convrot*` (or equivalent `int8_tensorwise` + `convrot`) safetensors is available under `diffusion_models`.  
3. Confirm Triton path already works for SDXL / Z-Image INT8 on the same venv (baseline).

### Phase 1 — Port ConvRot + Ops core (minimum for prequant load)

| Step | Action | Files |
|------|--------|-------|
| 1.1 | Port `convrot.py` | `patches/convrot.py` |
| 1.2 | Port / slim `Int8TensorwiseOps` Linear load + forward (ConvRot + Triton scalar **and** per_row) | `patches/int8_tensorwise_ops.py` |
| 1.3 | Idempotent layout registration as **`Int8TensorwiseLayout`** (INT8-Fast name — not kitchen `TensorWiseINT8Layout`) | called from Ops module or `__init__.py` |
| 1.4 | Reuse `int8_triton_kernels` (`triton_int8_linear` + `triton_int8_linear_per_row`) | import from `patches/int8_triton_kernels.py` |
| 1.5 | Parse `*.comfy_quant` JSON; stamp `_use_convrot` / `_convrot_groupsize` (default 256) | inside Linear `_load_from_state_dict` |
| 1.6 | Isolation: new modules only; **do not** edit `comfy_quant_int8.py` / `load_unet_hswq_weight_dtype` / existing loader UI | §5.1 |

**Exit:** Unit-style load of one Linear from a ConvRot INT8 state_dict key keeps `dtype=torch.int8` and `_use_convrot=True`; scalar and `(out,1)` `weight_scale` both select the correct forward. SDXL/ZI loaders unchanged at git diff.

### Phase 2 — Dedicated loader node + Krea2 exclusions (separate branch)

| Step | Action |
|------|--------|
| 2.1 | **New** loader node inputs: `model_type` (include `krea2`), `enable_convrot` (OTF; prequant follows metadata), `weight_dtype` compute override, OTF **default off** |
| 2.2 | Wire `custom_operations=Int8TensorwiseOps` into `load_diffusion_model_state_dict` **only inside the new node** |
| 2.3 | Apply Krea2 `excluded_names` when `model_type=="krea2"` (OTF / mixed only) |
| 2.4 | Register new node in `__init__.py`; do **not** change `HSWQFP8E4M3UNetLoader` mapping |
| 2.5 | Document circulating formats: native `format:int8_tensorwise`+`convrot` vs old INT8-Fast JSON (`convert_to_comfy.py` if needed) |
| 2.6 | Regression check: open SDXL INT8 + Z-Image INT8 graphs; confirm still use old dispatch (no new Ops) |

**Product shape (locked):** **A only** — new node `HSWQ Load Diffusion Model INT8 (W8A8)` (INT8-Fast `OTUNetLoaderW8A8` twin).  
**Rejected:** **B** — extending `HSWQFP8E4M3UNetLoader` / auto-routing ConvRot into MixedPrecision.

**Exit:** Graph loads Krea2 INT8 ConvRot UNet via the **new** node; VRAM stays compressed-INT8 class; no 64 GB RAM climb on load; existing SDXL/ZI nodes behave as before.

### Phase 3 — TE / coexistence / LoRA (after DiT smoke)

| Step | Action |
|------|--------|
| 3.1 | Quantized Qwen3-VL TE: if stock `CLIPLoader` fails on layout None, add TE-side layout fallback **or** document BF16 TE requirement for v1 |
| 3.2 | INT8 LoRA bake for ConvRot layers (INT8-Fast `INT8ModelPatcher` class) — Phase 3, not smoke-blocker |
| 3.3 | Confirm Nunchaku handoff / false-positive gates still ignore this Ops path |
| 3.4 | Optional: Forge-Nunchaku port of the same Ops path (out of scope until Comfy smoke green) |

### Phase 4 — Validation matrix

| Case | Expected |
|------|----------|
| Krea2 INT8 ConvRot prequant, ConvRot ON | Sample completes; RAM stable |
| Krea2 INT8 ConvRot, kitchen broken / absent | Still completes via own Ops (no kitchen ConvRot required) |
| Krea2 INT8 **non-ConvRot** prequant (plain `*_int8`) | Completes without rotation; Triton scalar/per_row as scale shape requires |
| Z-Image INT8 tensorwise (existing loader) | Unchanged |
| SDXL INT8 (existing loader) | Unchanged |
| Non-INT8 Krea2 BF16 (stock UNET loader) | Unchanged (stock Comfy path) |

---

## 7. Explicit non-goals (this plan)

- Editing `ComfyUI-master/` inside Forge-Nunchaku (forbidden unless that exact path is named in the same order).  
- Making kitchen `TensorCoreConvRotW4A4Layout` the only success path.  
- Replacing the entire SDXL / Z-Image MixedPrecision stack with INT8-Fast Ops in one commit.  
- Shipping Forge UI / Issue #3 reply as part of Phase 1–2.  
- Public release notes / version bump until smoke is confirmed.

---

## 8. Risk register

| Risk | Mitigation |
|------|------------|
| Dual INT8 stacks confuse loaders | §5.1: separate node; document “ConvRot / Krea2 → W8A8 loader only” |
| Accidental merge into MixedPrecision | Forbidden edit list in §5.1; Phase 2.6 regression on SDXL/ZI |
| Layout name clash (`Int8TensorwiseLayout` vs kitchen `TensorWiseINT8Layout`) | `setdefault` register INT8-Fast name only; never overwrite kitchen / MixedPrecision layout |
| Porting full `int8_quant.py` too large | Phase 1: Linear-only subset required for prequant forward |
| OTF quant RAM spike (INT8-Fast known) | Default OTF **off** for Krea2 smoke; prequant only |
| Circulating old `.comfy_quant` without `format` key | Document `convert_to_comfy.py` or accept both JSON shapes in parser |
| `weight_scale` per_row vs scalar mismatch | Branch forward like INT8-Fast; reuse existing per_row Triton in this repo |
| TE still OOMs | Phase 3: BF16 TE first; quantized TE second |
| `scipy` / Hadamard size constraints | Port Regular Hadamard (power-of-4 groupsize); fail soft if `in_features % groupsize != 0` |

---

## 9. Suggested file touch list (implementation order)

**Allowed (new / register-only):**

1. `patches/convrot.py` — new  
2. `patches/int8_tensorwise_ops.py` — new (Linear + flags + `setdefault` layout register)  
3. `hswq/int8_w8a8_unet.py` (or `nodes/…`) — **new** loader node only  
4. `__init__.py` — append new `NODE_CLASS_MAPPINGS` entry + one-time layout `setdefault`  
5. `md/KREA2_INT8_IMPLEMENTATION_PLAN.md` — status checkboxes when phases land  
6. Later: LoRA / TE docs under `md/` only if behavior is shipped  

**Forbidden for Phase 1–2 (existing routes):**

- `patches/comfy_quant_int8.py`  
- `hswq/zimage_fp8_e4m3_unet.py` (`HSWQFP8E4M3UNetLoader` body)  
- SDXL Checkpoint loader class in `__init__.py`  
- Changing Triton public API used by MixedPrecision (import-only OK)

Do **not** put plan-only notes under ignored `/docs/` for Forge; this plan lives in **this** repo’s tracked `md/`.

---

## 10. Phase checklist (execution)

```
Phase 0
□ Comfy core has Krea2 + CLIPType.KREA2
□ Real Int8Convrot (or equivalent) checkpoint on disk
□ SDXL/Z-Image INT8 baseline still OK

Phase 1
□ convrot.py ported (groupsize default 256, power-of-4 Regular Hadamard)
□ Int8TensorwiseOps Linear load keeps int8 + parses comfy_quant → _use_convrot
□ Forward rotates activations when _use_convrot
□ Triton / _int_mm hooked for **scalar and per_row**; tiny-M fallback only
□ Int8TensorwiseLayout registered via setdefault (kitchen / MixedPrecision untouched)
□ git diff: no changes under comfy_quant_int8 / HSWQFP8E4M3UNetLoader / SDXL loader

Phase 2
□ **New** W8A8 loader node only (HSWQFP8E4M3UNetLoader not extended)
□ custom_operations wired only inside that node
□ End-to-end KSampler smoke on ~16 GB VRAM class
□ System RAM does not climb to full machine on load
□ Old vs native comfy_quant format documented (convert helper if needed)
□ SDXL INT8 + Z-Image INT8 still load via existing nodes (regression)

Phase 3
□ TE policy decided (BF16 vs quantized + fallback)
□ INT8ModelPatcher + LoRA bake / Pre-Lora (optional)
□ INT8ModelSave (optional)
□ Handoff / SVDQ gates still clean
□ compute_dtype / SM75 notes if Windows Triton path matters

Phase 4
□ Matrix above signed off
□ Only then: Forge port / Issue #3 follow-up
```

---

## 11. Decision summary

| Decision | Choice |
|----------|--------|
| Reference implementation | `ComfyUI-INT8-Fast` |
| ConvRot provider | In-repo Hadamard (`convrot.py`), not kitchen ConvRot layout |
| First product surface | **Dedicated** INT8 W8A8 UNet loader (`krea2` exclusions) — **not** an extension of `HSWQFP8E4M3UNetLoader` |
| Existing SDXL/Z-Image path | **Separate branch**; MixedPrecision / `comfy_quant_int8` untouched in Phase 1–2 (§5.1) |
| Shared Triton | Import-only from existing kernels; no API change for old path |
| Forge | After Comfy smoke |

---

## 12. Next action after this plan

**Default (after §14):** Prefer stock ComfyUI + `comfy_kitchen` for Krea2 / ConvRot. Do **not** start Phase 1 INT8-Fast port unless the owner orders a fallback for kitchen-absent hosts.

If fallback is ordered: implement Phase 1 → Phase 2 (§5.1 isolation) until smoke passes. Do not start Forge-Nunchaku changes until that smoke is green unless a separate order names Forge paths explicitly.

---

## 13. Gap review vs `D:\ComfyUI-INT8-Fast` (2026-07-19)

Reviewed against: `__init__.py`, `convrot.py`, `int8_quant.py`, `int8_unet_loader.py`, `int8_fused_kernel.py`, `int8_lora.py`, `int8_save.py`, `convert_to_comfy.py`, `README.md`.

### 13.1 Already covered (no change to goal)

| Item | Plan status |
|------|-------------|
| Own Ops + `custom_operations` (not kitchen ConvRot layout) | Covered |
| `convrot.py` Hadamard + activation rotate in forward | Covered |
| Dedicated loader + `model_type=krea2` exclusions | Covered |
| OTF default off; prequant first | Covered |
| Keep SDXL / Z-Image MixedPrecision untouched | Covered |
| Triton reuse from this repo | Covered (kernels already include per_row) |

### 13.2 Gaps found in the previous plan draft (now patched above)

| Gap | Why it matters for circulating Krea2 INT8 | Severity for smoke |
|-----|-------------------------------------------|--------------------|
| **`*.comfy_quant` JSON parse** (`convrot`, `convrot_groupsize`, optional `per_row`, `format`) | Prequant ConvRot is stamped per-layer in metadata bytes, not only by filename | **Smoke-blocker** |
| **Default `CONVROT_GROUP_SIZE = 256`** + divisibility check | INT8-Fast hardcodes 256; wrong groupsize breaks rotate | **Smoke-blocker** |
| **per_row vs scalar `weight_scale`** | OTF and many ConvRot dumps use axiswise/`(out,1)` scales; scalar-only forward is wrong | **Smoke-blocker** |
| **Layout class name `Int8TensorwiseLayout`** vs kitchen `TensorWiseINT8Layout` | INT8-Fast registers a **different** layout string; clash risk was under-specified | High (regression) |
| **Old vs native comfy_quant JSON** (`convert_to_comfy.py`) | Older INT8-Fast dumps omit `format: int8_tensorwise` | High (compat) |
| **`input_scale` pop/ignore** | Leftover keys can confuse strict load | Medium |
| **`weight_dtype` / compute_dtype (incl. SM75)** | Loader widget in INT8-Fast; needed for Ampere/Turing edge cases | Medium (post-smoke OK) |
| **`INT8ModelPatcher.clone` after load** | Required for normal LoRA nodes on INT8 layers | Phase 3 |
| **Pre-Lora + `INT8GroupedLora` + `INT8ModelSave`** | Full product parity with INT8-Fast | Phase 3+ |
| **Aimdo / `dynamic_load_device` defer bake** | RAM path for OTF+LoRA; not needed for prequant smoke | Later |
| **Phase 4 missing plain Krea2 INT8 (non-ConvRot)** | HF also ships non-ConvRot `*_int8` | Validation gap (now added) |
| **Reference table said LoRA “Phase 2” while body said Phase 3** | Inconsistency | Fixed → Phase 3 |

### 13.3 Explicit non-gaps (safe to defer)

- Re-quant / Save node for producing new ConvRot packs — not required to **load** circulating weights.
- Full multi-`model_type` exclusion lists (flux2, wan, ltx2, …) — only **`krea2`** is required for this plan’s primary goal.
- `WEB_DIRECTORY` / js from INT8-Fast — irrelevant.
- Replacing kitchen for all models — still a non-goal.

### 13.4 Smoke-minimum port subset (from INT8-Fast)

For Phase 1–2 only, port/adapt:

1. `convrot.py` (full)  
2. `Int8TensorwiseOps.Linear` load (int8 + scale + comfy_quant) + forward (rotate + Triton/ `_int_mm` scalar+per_row)  
3. `_register_layouts()` → `Int8TensorwiseLayout` + `QUANT_ALGOS["int8_tensorwise"]` setdefault  
4. Loader: flags + `krea2` exclusions + `load_diffusion_model_state_dict` + `custom_operations`  

## 14. Stock ComfyUI + `comfy_kitchen` ConvRot (2026-07-19 discovery)

Verified on host tree `D:\USERFILES\ComfyUI` (latest ComfyUI + installed `comfy_kitchen`).

### 14.1 What stock already does

| Piece | Location | Role |
|-------|----------|------|
| ConvRot metadata on `int8_tensorwise` | `ComfyUI/comfy/ops.py` (~1094–1106, ~1165–1167) | Reads `comfy_quant.convrot` / `convrot_groupsize` into layout Params |
| Layout + Hadamard / rotate | `comfy_kitchen.tensor.int8.TensorWiseINT8Layout` | Weight-side ConvRot + forward via `torch.ops.comfy_kitchen.int8_linear(..., convrot=...)` |
| Unit tests | `ComfyUI/tests-unit/comfy_quant/test_mixed_precision.py` | `test_int8_convrot_metadata_loads_into_params` |
| Official Krea2 INT8 template | `comfyui_workflow_templates_json` / `image_krea2_turbo_t2i_int8.json` | Stock workflow surface |

**Implication:** Circulating `*Int8Convrot*` / `int8_tensorwise`+`convrot` can load on **stock UNETLoader + MixedPrecision** when kitchen is present. Re-porting INT8-Fast `Int8TensorwiseOps` + `convrot.py` for *load correctness* is **redundant** on this host class.

### 14.2 Triton / acceleration — kitchen already has it

| Backend | `int8_linear` | ConvRot in forward |
|---------|---------------|--------------------|
| **Triton** | `comfy_kitchen.backends.triton.quantization.int8_linear` | Yes (rotate activation → `triton_quantize_rowwise` → fused INT8 GEMM). Constraint: **SM ≥ 8.0** for Triton INT8 dot |
| **CUDA** | `comfy_kitchen.backends.cuda` | Yes (native / Turing paths) |
| **Eager** | `torch._int_mm` / cuBLASLt IMMA | Yes (fallback) |

So: **“ComfyUI has no Triton INT8 acceleration” is false** for current kitchen. Acceleration is inside `comfy_kitchen`, selected by the kitchen registry (cuda / triton / eager), not missing.

This repo’s own `patches/int8_triton_kernels.py` remains useful for:

- Environments **without** kitchen / broken kitchen import  
- Non-ConvRot SDXL / Z-Image path already wired in `comfy_quant_int8.py`  
- Note: that patch **still returns early when `params.convrot=True`** — ConvRot forward is intentionally left to kitchen’s `int8_linear` (which can use kitchen Triton)

### 14.3 Efficiency strategy (revised) — import / detect, do not clone INT8-Fast

```text
IF comfy_kitchen TensorWiseINT8Layout supports Params.convrot
   AND stock load path wires comfy_quant.convrot
THEN
   Krea2 / ConvRot: use stock Comfy load (UNETLoader / MixedPrecision)
   DO NOT port Int8TensorwiseOps / local convrot.py as the primary path
   Optional: thin detect + docs / node tip only
ELSE
   Fall back to §5–§6 INT8-Fast-class port (older Comfy / no kitchen)
```

| Do | Don’t |
|----|-------|
| `import comfy_kitchen` / `comfy.quant_ops` / `comfy.ops` at **runtime** from the **same Python** as ComfyUI | Hardcode `sys.path` to `D:\USERFILES\ComfyUI\...` as the product path |
| Prefer kitchen `int8_linear` (cuda→triton→eager) for ConvRot | Duplicate Hadamard + Ops from INT8-Fast when kitchen works |
| Keep this repo’s Triton for **non-ConvRot** / kitchen-absent hosts | Replace kitchen ConvRot with a second Ops stack on hosts that already load fine |
| Document: “latest Comfy + kitchen = stock ConvRot OK” | Assume all users lack Triton |

### 14.4 What remains useful in *this* custom node

1. **SDXL / Z-Image INT8** MixedPrecision compatibility patches only (Conv2d / LoRA / ControlLora / handoff). Linear acceleration is **Comfy kitchen**, not this repo.  
2. **Kitchen-absent / old Comfy** fallback (only if we still need to support those hosts).  
3. Optional UX: detect ConvRot checkpoint and tell the user to use stock loader when kitchen ConvRot is available.  
4. Do **not** start Phase 1 INT8-Fast port as default work while stock ConvRot already samples.

### 14.6 Product decision — own Triton removed (v3.2.7)

**Fact (owner):** The in-node fused INT8 Triton path (v3.2.6 Plan B) is **discarded**. Acceleration is delegated to **ComfyUI + `comfy_kitchen`**. This extension stays light: INT8 load compatibility only.

| Former concern (v3.2.6) | Current answer (v3.2.7+) |
|---------|------------------------------|
| Unclear when stock Comfy uses Triton vs `_int_mm` | Accept kitchen dispatch (`cuda` → `triton` → `eager`) |
| Triton missing on Windows / Manager installs | Not this extension’s job; host Comfy / kitchen install |
| Operator switch `triton_accelerate` | **Removed** |
| Own kernels / `install.py` Triton stage | **Removed** |

**Product implication:** Prefer kitchen for INT8 Linear speed (including ConvRot). Do **not** reintroduce a second Triton stack in this node unless the owner explicitly orders it again.
