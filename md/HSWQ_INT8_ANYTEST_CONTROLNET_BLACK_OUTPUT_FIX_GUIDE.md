# HSWQ INT8 — anytest (LoRA-type ControlNet) Black-Output Fix

This document explains the black-image problem that occurs when an SDXL
**LoRA-type ControlNet** (e.g. `CN-anytest_v4-marged_am_dim256.safetensors`) is
combined with an **INT8 (`comfy_quant`)** quantized base UNet, why it happens,
how it is fixed, and the exact code that implements the fix.

Ordinary full-weight ControlNets (Canny, Depth, etc.) are **not** affected. The
problem is specific to LoRA-type ControlNets that borrow the base UNet's own
weights. The same setup works under FP8, so the failure is INT8-specific.

---

## 1. Why the output turned completely black

### 1.1 How a LoRA-type ControlNet loads its weights

A LoRA-type ControlNet is marked with `lora_controlnet` and is handled by
`comfy.controlnet.ControlLora`. Unlike a full ControlNet, it does **not** carry a
complete copy of the UNet. Instead, at sampling time, `ControlLora.pre_run`
**borrows the base UNet's own weights** and injects them into the control model:

```python
# comfy/controlnet.py  (ControlLora.pre_run, abridged for context)
sd = diffusion_model.state_dict()
for k in sd:
    weight = sd[k]
    ...
    comfy.utils.set_attr_param(self.control_model, k, weight)
```

The control model itself uses `ControlLoraOps` — **plain float linear/conv ops**.
Its forward calls `comfy.ops.cast_bias_weight`, which simply casts the stored
weight to the compute dtype. It has **no ability to reconstruct a quantized
weight**, because it never receives a quantization scale.

### 1.2 What INT8 `state_dict()` actually returns

Under INT8, the base UNet's linear/conv modules are wrapped by
`MixedPrecisionOps`. Its `state_dict` implementation
(`_quantized_weight_state_dict`) does **not** emit `QuantizedTensor` objects.
It **flattens** each quantized `weight` into several separate raw tensors:

| state_dict key        | dtype           | content                          |
|-----------------------|-----------------|----------------------------------|
| `X.weight`            | `torch.int8`    | raw quantized data (qdata)       |
| `X.weight_scale`      | `torch.float32` | per-tensor dequantization scale  |
| `X.comfy_quant`       | `torch.uint8`   | JSON quantization metadata       |
| `X.input_scale` / `X.weight_scale_2` | varies | extra params (fp8/nvfp4) |

So when `ControlLora.pre_run` reads `diffusion_model.state_dict()`, the value it
receives for `X.weight` is the **raw int8 qdata with no scale attached**.

### 1.3 The failure

`ControlLora.pre_run` injects that raw `int8` tensor into the **float** control
model. On forward, the control model feeds the int8 values directly into
`F.linear` / `conv2d` without ever multiplying by `weight_scale`. The result is
numerically meaningless — magnitudes explode or collapse, producing `NaN` / `Inf`
and therefore a **completely black image**.

FP8 escapes this because the fp8 `weight` dtype differs from the compute dtype in
a way that does not detonate the same code path; INT8 does not, so INT8 is the
only variant that goes black.

**Root cause, stated precisely:** the control model receives the base UNet's
`int8` qdata **without its scale**, because `state_dict()` flattens the
`QuantizedTensor` and `ControlLoraOps` cannot rebuild it.

---

## 2. Countermeasure overview

The fix wraps `diffusion_model.state_dict` **only for the duration of
`ControlLora.pre_run`**, so that the borrow returns a **dequantized** state dict:

- For every module whose `.weight` is a `QuantizedTensor`, `X.weight` is replaced
  with `weight.dequantize()` — a real float tensor at full precision.
- The now-meaningless sidecar keys (`X.weight_scale`, `X.weight_scale_2`,
  `X.comfy_quant`, `X.input_scale`) are dropped so they are not injected as bogus
  parameters into the float control model.
- All non-quantized weights, biases and buffers pass through unchanged.
- After `pre_run` returns, the wrapper is removed and the model's normal
  `state_dict` is restored.

Design points:

- The wrapper only intercepts the **argument-less** borrow call that
  `ControlLora` makes. Any `state_dict(destination=...)` / keyword usage falls
  back to the original method, so unrelated code paths are untouched.
- Dequantization uses the **original bound `state_dict` method captured before
  replacement** (`orig_sd`). Calling `diffusion_model.state_dict()` from inside
  the wrapper would re-enter the wrapper and cause
  `RecursionError: maximum recursion depth exceeded`; using `orig_sd()` breaks
  that loop.
- Full-weight ControlNets never enter `ControlLora.pre_run`, so they are
  completely unaffected. The real anytest LoRA weights (`.up` / `.down`) are
  plain fp16 and are never touched — only the **borrowed base UNet weights** are
  dequantized.

The patch is versioned (`_CL_VER = 2`) and idempotent: re-applying it is a no-op
once the current version is installed.

---

## 3. Added / modified files

| File | Change |
|------|--------|
| `patches/comfy_quant_int8.py` | Added `_patch_controllora_int8_dequant()` and registered it in `apply_comfy_quant_int8_patches()`. |

No other files were changed. No changes were made to ComfyUI core
(`comfy/controlnet.py`, `comfy/ops.py`, `comfy/quant_ops.py`); the behavior is
altered at runtime by wrapping `ControlLora.pre_run`.

---

## 4. Full added / modified code (no omissions)

### 4.1 `patches/comfy_quant_int8.py` — `_patch_controllora_int8_dequant()`

```python
def _patch_controllora_int8_dequant() -> bool:
    """Dequantize borrowed base-UNet quantized weights in ControlLora.pre_run.

    LoRA-type ControlNets (``lora_controlnet`` marker, e.g. anytest) build a
    control_model that BORROWS the base UNet's own weights via
    ``diffusion_model.state_dict()`` and injects them with ``set_attr_param``.
    The control_model uses ``ControlLoraOps`` (plain float ops); its forward
    calls ``comfy.ops.cast_bias_weight``, which cannot reconstruct a quantized
    weight without its scale.

    Root cause (confirmed from logs + comfy/ops.py):
    ``MixedPrecisionOps.state_dict`` (``_quantized_weight_state_dict``) does NOT
    emit ``QuantizedTensor`` objects. It FLATTENS each quantized ``weight`` into
    separate tensors:
      * ``X.weight``        -> raw int8 qdata      (torch.int8)
      * ``X.weight_scale``  -> per-tensor scale    (torch.float32)
      * ``X.comfy_quant``   -> JSON metadata       (torch.uint8)
      * ``X.input_scale`` / ``X.weight_scale_2`` -> extra params (fp8/nvfp4)
    So ``ControlLora.pre_run`` injects the RAW int8 ``X.weight`` (no scale) into
    the float control_model, and forward feeds int8 straight into
    ``F.linear`` / ``conv2d`` -> NaN / black output. FP8 avoids this only
    because its dtype differs from the compute dtype.

    Fix: wrap ``diffusion_model.state_dict`` during ``ControlLora.pre_run`` and
    return a DEQUANTIZED state dict: for every module whose ``.weight`` is a
    ``QuantizedTensor``, replace ``X.weight`` with ``weight.dequantize()`` (a
    real float tensor) and drop the now-meaningless sidecar keys
    (``X.weight_scale``, ``X.weight_scale_2``, ``X.comfy_quant``,
    ``X.input_scale``). All non-quant weights, biases and buffers pass through
    unchanged. Full-weight ControlNets (Canny) never enter
    ``ControlLora.pre_run`` and are unaffected; the real anytest LoRA weights
    (``.up`` / ``.down``) are plain fp16 and are not touched.
    """
    try:
        import comfy.controlnet as cn
        import comfy.utils
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    ControlLora = getattr(cn, "ControlLora", None)
    if ControlLora is None:
        return False
    original = getattr(ControlLora, "pre_run", None)
    _CL_VER = 2
    if original is None or getattr(original, "_hswq_int8_controllora_ver", 0) >= _CL_VER:
        return getattr(original, "_hswq_int8_controllora", False)
    true_orig = getattr(original, "_hswq_orig_controllora_pre_run", original)

    def _dequantized_state_dict(diffusion_model, orig_sd):
        """Return diffusion_model.state_dict() with quantized weights turned
        back into float tensors and their scale/metadata sidecars removed.

        ``orig_sd`` is the ORIGINAL bound ``state_dict`` method captured before
        we replaced ``diffusion_model.state_dict``. It MUST be used here instead
        of ``diffusion_model.state_dict()`` to avoid re-entering our wrapper
        (which caused ``RecursionError: maximum recursion depth exceeded``)."""
        full = orig_sd()

        # Collect the state-dict prefix of every quantized weight.
        quant_weight_keys = {}
        for name, module in diffusion_model.named_modules():
            w = getattr(module, "weight", None)
            if isinstance(w, QuantizedTensor):
                key = (name + "." if name else "") + "weight"
                quant_weight_keys[key] = w

        out = {}
        n_dequant = 0
        n_drop = 0
        for k, v in full.items():
            replaced = False
            dropped = False
            for wk, qt in quant_weight_keys.items():
                if k == wk:
                    # raw int8 qdata -> real float weight
                    try:
                        out[k] = qt.dequantize()
                        n_dequant += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[HSWQ INT8] ControlLora: dequantize failed for %s: %s",
                            k, e,
                        )
                        out[k] = v
                    replaced = True
                    break
                base = wk[: -len("weight")]  # "X."
                if (
                    (k.startswith(wk) and k != wk)      # X.weight_scale / weight_scale_2
                    or k == base + "comfy_quant"        # uint8 JSON metadata
                    or k == base + "input_scale"        # fp8 extra param
                ):
                    dropped = True
                    break
            if replaced:
                continue
            if dropped:
                n_drop += 1
                continue
            out[k] = v

        print(
            f"[HSWQ INT8][ControlLora] dequantized state_dict: "
            f"weights dequantized(int8->float)={n_dequant}, "
            f"sidecar keys dropped(scale/comfy_quant/input_scale)={n_drop}, "
            f"total keys out={len(out)}",
            flush=True,
        )
        return out

    def pre_run(self, model, percent_to_timestep_function):
        diffusion_model = getattr(model, "diffusion_model", None)
        patched = False
        orig_sd = None
        if diffusion_model is not None:
            orig_sd = diffusion_model.state_dict

            def dequant_state_dict(*a, **kw):
                # Only intercept the argument-less borrow call ControlLora makes;
                # fall back to the original for any keyword/destination usage.
                if a or kw:
                    return orig_sd(*a, **kw)
                return _dequantized_state_dict(diffusion_model, orig_sd)

            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER "
                "(LoRA-type ControlNet / lora_controlnet path) "
                "-> wrapping diffusion_model.state_dict for INT8 base-weight dequant",
                flush=True,
            )
            diffusion_model.state_dict = dequant_state_dict
            patched = True
        else:
            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER but model has no "
                "diffusion_model; running unpatched",
                flush=True,
            )

        try:
            result = true_orig(self, model, percent_to_timestep_function)
        finally:
            if patched:
                # Remove the instance-level override so the class method is used again.
                try:
                    del diffusion_model.state_dict
                except AttributeError:
                    diffusion_model.state_dict = orig_sd

        print(
            "[HSWQ INT8][ControlLora] pre_run EXIT (base weights injected as float)",
            flush=True,
        )
        logger.info(
            "[HSWQ INT8] ControlLora: injected dequantized base UNet weights "
            "(anytest / lora_controlnet black-output fix)"
        )
        return result

    pre_run._hswq_int8_controllora = True
    pre_run._hswq_int8_controllora_ver = _CL_VER
    pre_run._hswq_orig_controllora_pre_run = true_orig
    ControlLora.pre_run = pre_run
    print(
        "[HSWQ INT8][ControlLora] pre_run patch INSTALLED "
        "(v%d): borrowed INT8 base weights dequantized via state_dict wrap "
        "for LoRA-type ControlNet (anytest fix)" % _CL_VER,
        flush=True,
    )
    return True
```

### 4.2 `patches/comfy_quant_int8.py` — registration in `apply_comfy_quant_int8_patches()`

```python
def apply_comfy_quant_int8_patches() -> bool:
    """Install INT8 comfy_quant patches once. Returns True if applied (or already applied)."""
    global _PATCHES_APPLIED
    ok_keys = _patch_load_lora_key_counts()
    ok_name = _patch_lora_loader_name_context()
    ok_path = _patch_loras_folder_path_name()
    ok_torch = _patch_load_torch_file_lora_name()
    ok_lowvram = _patch_lowvram_patch_float_intermediate()
    ok_dyn_bake = _patch_model_patcher_dynamic_int8_lora_bake()
    ok_handoff = _patch_load_models_gpu_int8_nunchaku_handoff()
    ok_controllora = _patch_controllora_int8_dequant()
```

---

## 5. What the code means (line-by-line intent)

### 5.1 Guard, imports and version gate

```python
    try:
        import comfy.controlnet as cn
        import comfy.utils
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
```

The patch only makes sense when ComfyUI's ControlNet and quantization modules
exist. If they cannot be imported, the patch does nothing and reports failure.

```python
    ControlLora = getattr(cn, "ControlLora", None)
    if ControlLora is None:
        return False
    original = getattr(ControlLora, "pre_run", None)
    _CL_VER = 2
    if original is None or getattr(original, "_hswq_int8_controllora_ver", 0) >= _CL_VER:
        return getattr(original, "_hswq_int8_controllora", False)
    true_orig = getattr(original, "_hswq_orig_controllora_pre_run", original)
```

`_CL_VER = 2` is the patch version. If `pre_run` is already patched at this
version or newer, the function returns early — installation is **idempotent** and
safe to call on every startup. `true_orig` recovers the genuine, unwrapped
`pre_run` even if an earlier patch version had already wrapped it, so wrappers
never stack.

### 5.2 Dequantizing helper — `_dequantized_state_dict`

```python
        full = orig_sd()
```

`orig_sd` is the **original** bound `state_dict` captured before replacement.
Calling it (instead of `diffusion_model.state_dict()`) is what prevents infinite
recursion: the wrapper never calls itself. This is the exact line that fixes the
`RecursionError`.

```python
        quant_weight_keys = {}
        for name, module in diffusion_model.named_modules():
            w = getattr(module, "weight", None)
            if isinstance(w, QuantizedTensor):
                key = (name + "." if name else "") + "weight"
                quant_weight_keys[key] = w
```

This walks the live module tree and records, for every module whose `.weight` is
still a live `QuantizedTensor`, the state-dict key of that weight (`X.weight`)
together with the `QuantizedTensor` object itself. This gives access to the real
`dequantize()` on the live object, which the flattened `state_dict` no longer
exposes.

```python
        for k, v in full.items():
            ...
            for wk, qt in quant_weight_keys.items():
                if k == wk:
                    out[k] = qt.dequantize()
                    ...
```

For a key that is a quantized weight, the raw int8 value in the state dict is
replaced with `qt.dequantize()` — a proper float tensor reconstructed from qdata
× scale. This is the value the float control model actually needs.

```python
                base = wk[: -len("weight")]  # "X."
                if (
                    (k.startswith(wk) and k != wk)      # X.weight_scale / weight_scale_2
                    or k == base + "comfy_quant"        # uint8 JSON metadata
                    or k == base + "input_scale"        # fp8 extra param
                ):
                    dropped = True
                    break
```

The sidecar keys that only exist to describe the quantization
(`X.weight_scale`, `X.weight_scale_2`, `X.comfy_quant`, `X.input_scale`) are
dropped. Once `X.weight` is a plain float tensor, these carry no meaning and must
not be injected into the float control model as spurious parameters.

```python
            if replaced:
                continue
            if dropped:
                n_drop += 1
                continue
            out[k] = v
```

Everything else — non-quantized weights, biases, norm buffers — passes through
unchanged. The counters `n_dequant` / `n_drop` are printed so the console log
shows exactly how many weights were converted and how many sidecars were removed.

### 5.3 The `pre_run` wrapper

```python
        if diffusion_model is not None:
            orig_sd = diffusion_model.state_dict

            def dequant_state_dict(*a, **kw):
                if a or kw:
                    return orig_sd(*a, **kw)
                return _dequantized_state_dict(diffusion_model, orig_sd)

            diffusion_model.state_dict = dequant_state_dict
            patched = True
```

Just before the original `pre_run` runs, `diffusion_model.state_dict` is
temporarily replaced. The replacement only rewrites the **argument-less** borrow
call that `ControlLora.pre_run` makes; any keyword or `destination=` usage falls
straight through to the original, so nothing else in ComfyUI is disturbed.

```python
        try:
            result = true_orig(self, model, percent_to_timestep_function)
        finally:
            if patched:
                try:
                    del diffusion_model.state_dict
                except AttributeError:
                    diffusion_model.state_dict = orig_sd
```

The original `pre_run` runs with the wrapper active, so the weights it borrows are
already dequantized. The `finally` block guarantees the override is removed even
if `pre_run` raises, restoring the model's normal `state_dict` behavior. Deleting
the instance attribute lets the class method take over again; if there is no
instance attribute to delete, the captured original is reassigned as a fallback.

### 5.4 Version stamping and installation

```python
    pre_run._hswq_int8_controllora = True
    pre_run._hswq_int8_controllora_ver = _CL_VER
    pre_run._hswq_orig_controllora_pre_run = true_orig
    ControlLora.pre_run = pre_run
```

The new `pre_run` is stamped with the patch flag, its version, and a handle to
the genuine original. `ControlLora.pre_run` is then replaced class-wide. The
stamps are what make the version gate in §5.1 work on subsequent startups.

### 5.5 Registration

```python
    ok_controllora = _patch_controllora_int8_dequant()
```

`apply_comfy_quant_int8_patches()` calls the installer during INT8 patch setup,
so the ControlLora dequant wrap is active before any sampling with a LoRA-type
ControlNet begins.

---

## 6. Result

With the patch active, a LoRA-type ControlNet under INT8 borrows the base UNet
weights as **float** tensors (dequantized on the fly) instead of raw int8 qdata.
The console log during `pre_run` reports the number of weights dequantized
(int8 → float) and the number of sidecar keys dropped, and generation produces a
correct image instead of a black frame. Full-weight ControlNets and the FP8 path
are unchanged.
