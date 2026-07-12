# HSWQ INT8（comfy_quant）対応・LoRA 対応 — 完全技術解説書

記録日: 2026-07-12  
リポジトリ: `ussoewwin/ComfyUI-nunchaku-unofficial-loader`  
反映コミット: `516ac28`（`feat: native comfy_quant INT8 load, LoRA bake Status, fix Dynamic VBAR OOM`）  
バージョン表記: `3.1.8`（本変更では bump / tag / GitHub Release なし）

本文書は、ご主人様のご命令どおり次の四点を満たす。

1. **実施した概要**
2. **追加・修正したファイル名**
3. **追加・修正したコードの全文**
4. **その意味**

---

## ① 実施した概要

### 1.1 目的

ComfyUI 本体を改変せず、拡張 `ComfyUI-nunchaku-unofficial-loader` 側のモンキーパッチだけで次を実現した。

| 柱 | 内容 |
|----|------|
| **A. Native comfy_quant INT8 ロード** | `int8_tensorwise` / `.comfy_quant` 付き SD UNet チェックポイントを、本体 `MixedPrecisionOps` 経路で読めるようにする（特に **Conv2d**）。 |
| **B. INT8 上での LoRA** | 任意の LoRA ローダが通る共通経路にフックし、**lora名 / 適用キー数 / スキップキー数** を Status ログで出し、Dynamic VRAM でも **dequant → bake → requant** で焼き込む。 |
| **C. FaceDetailer 二回目の VBAR OOM 修正** | bake 後に `module._v` を消すと bump allocator が再 `alloc` して OOM する。`_v` を残し、baked キーの LowVramPatch を剥がす。 |

### 1.2 なぜ本体パッチでは足りないか

ComfyUI 本体の `MixedPrecisionOps` は主に **Linear / Embedding / MoE** の量子化ロードを想定している。SD UNet の INT8（comfy_quant）は **Conv2d の weight が int8 + `.comfy_quant` マーカー**として入る。ここを本体だけに任せると次が起きる。

- `Only Tensors of floating point and complex dtype can require gradients`（int8 Parameter 扱い失敗）
- Conv 層に `set_weight` / `convert_weight` が無く、LoRA が **int8 への丸め**に落ちてデルタが消える

よって拡張側で次を注入する。

1. `convert_old_quants` — comfy_quant JSON の正規化（二重エンコード・裸文字列）
2. `MixedPrecisionOps.Conv2d` — Linear と同型の量子化 Conv2d（`set_weight` / `convert_weight` 付き）
3. LoRA 名キャプチャ（folder_paths / load_torch_file / LoraLoader / stack）
4. `load_lora` / `load_lora_for_models` — 適用・スキップ集計
5. `ModelPatcherDynamic.load` — Dynamic VRAM 後の INT8 bake + VBAR 安全策
6. `LowVramPatch.__call__` — int8 中間 dtype の回避

### 1.3 ロード入口（ユーザーが触るノード）

| ノード / API | 挙動 |
|--------------|------|
| **HSWQLoader**（`nodes/hswq_loader_node.py`） | パスを `checkpoint_looks_like_comfy_quant_int8` で判定。INT8 なら `load_checkpoint_guess_config`（本体 MixedPrecisionOps）。FP8 Scaled なら従来の `.scale` 逆量子化経路。 |
| **HSWQFP8E4M3UNetLoader**（`hswq/zimage_fp8_e4m3_unet.py`） | `weight_dtype` に `int8_tensorwise` を追加。自動検出または明示指定で INT8。INT8 weight に float8 強制をしない。 |

INT8 ロード直後: `reset_int8_lora_log_counters()` → `summarize_int8_lora_capability(model)`（Linear/Conv2d の set/convert 準備状況をコンソールへ）。

### 1.4 LoRA Status の仕様（ご要求どおり）

1 スロットあたりの必須表示:

- `lora名=` … 実ファイル名（アダプタ型名 `"lora"` ではない）
- `適用キー数=` … unet + clip（内訳付き）
- `スキップキー数=` … `not_mapped` + `mapped_but_not_attached`

スタイルは `ComfyUI-QwenImageLoraLoader` の Status 行に寄せ、**特定 LoRA ノードへの依存はしない**（共通フックのみ）。

### 1.5 Dynamic bake と VBAR（致命バグの修正）

**症状:** 一回目生成は成功、Impact FaceDetailer 等で二回目 `ModelPatcherDynamic.load` すると `MemoryError: VBAR OOM`。

**原因:** bake 後に `module._v` を unpin/delete していた。`ModelVBAR.alloc` は offset が増えるだけの bump allocator。次 load で `_v` が無い → 再 `alloc` → OOM。

**修正（`_DYN_VER = 4`）:**

- bake 後も **`_v` を消さない**
- bake 済みキーを `patcher.patches` から pop
- 焼き込み前の `backup` を落とし、`restore_loaded_backups` が bake を巻き戻さないようにする
- `model._hswq_int8_baked_keys` + `_hswq_int8_baked_uuid` で追跡。新 LoRA（uuid 変更）で invalidate
- 毎回 load 後 `_strip_lowvram_for_baked_keys`（クローンで二重適用防止）

### 1.6 起動時配線

`__init__.py` import 時に `apply_comfy_quant_int8_patches()` を呼ぶ。失敗しても他ノード登録は継続（`logger.debug`）。

### 1.7 既知の限界

- Status はフックが発火したローダに依存する。極端に独自経路だけのカスタムノードは名前が `unknown_lora#N` になり得る。
- パッチ版番号（`_DYN_VER` 等）を上げたあとは **ComfyUI 再起動**が必要（ホットリロードで古いラッパが残る）。
- bake 後も `_v` は残るが、該当キーの LowVramPatch はクリア済み。焼き込み済み weight は Parameter 上に残る。

---

## ② 追加・修正したファイル名

コミット `516ac28` の対象（5 ファイル、`__pycache__` は対象外）:

| 種別 | パス | 行数目安（コミット差分） |
|------|------|--------------------------|
| **新規** | `patches/comfy_quant_int8.py` | +1455（実装本体） |
| **修正** | `__init__.py` | パッチ適用呼び出し + ログ文言 |
| **修正** | `nodes/hswq_loader_node.py` | INT8 自動判定ロード経路、タイトル |
| **修正** | `hswq/zimage_fp8_e4m3_unet.py` | `int8_tensorwise` / INT8 UNet 経路 |
| **修正** | `patches/zimage_fp8_torchcompile.py` | shape skip → `record_lora_shape_skip` |

変更統計: `5 files changed, 1574 insertions(+), 50 deletions(-)`

---

## ③ 追加・修正したコードの全文

以下はディスク上の **現行全文**（コミット `516ac28` 反映後）。  
新規ファイルはファイル全体。修正ファイルも「どの行がどう変わったか」を欠かさないため **ファイル全文**を収録する（差分だけだと前後文脈が欠けるため）。


### 3.1 `patches/comfy_quant_int8.py`（全文・1455 行）

```python
"""
ComfyUI core-safe patches for native comfy_quant INT8 (int8_tensorwise).

Upstream MixedPrecisionOps only quant-loads Linear / Embedding / MoE.
SD UNet INT8 checkpoints also store Conv2d weights as int8 + comfy_quant, which
fails with: Only Tensors of floating point and complex dtype can require gradients.

Also normalizes bare-string / double-encoded comfy_quant JSON some exporters write.

LoRA: native Linear already has convert_weight + set_weight (dequant → bake →
requant, same idea as BobJohnson24/ComfyUI-INT8-Fast). Injected Conv2d must
mirror that set_weight; without it ModelPatcher falls back to rounding into
int8 and LoRA deltas on Conv layers vanish.

Applied from ComfyUI-nunchaku-unofficial-loader so ComfyUI core updates do not wipe it.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# LoRA bake path logs (rate-limited so console stays readable)
_LORA_CONVERT_LOG_MAX = 0  # quiet; Status dump is enough
_LORA_SET_LOG_MAX = 0
_LORA_PATCHER_LOG_MAX = 0  # per-key bake lines off; Status dump is enough
_lora_convert_logs = 0
_lora_set_logs = 0
_lora_patcher_logs = 0
_lora_patcher_stats = {
    "calls": 0,
    "with_set_func": 0,
    "without_set_func": 0,
    "with_convert_func": 0,
}

# LoRA key attach / skip accounting (last load_lora_for_models call)
_lora_attach_last = {
    "lora_name": "",
    "strength_model": None,
    "strength_clip": None,
    "lora_file_keys": 0,
    "mapped_keys": 0,
    "applied_unet": 0,
    "applied_clip": 0,
    "applied_unet_keys": [],
    "applied_clip_keys": [],
    "not_mapped": [],
    "mapped_but_not_attached": [],
    "add_patches_skipped_unet": [],
}
# One entry per load_lora_for_models call (stacked loaders → multiple entries)
_lora_attach_history = []
# key -> "requant" | "int8_round" recorded during bake
_lora_bake_by_key = {}
# Set by LoraLoader.load_lora wrap (and cleared after attach)
_current_lora_name = None
_current_lora_strength_model = None
_current_lora_strength_clip = None
_lora_shape_skips = []  # list of (lora_name, key, reason)
_LORA_SKIP_PRINT_MAX = 40


def _console(msg: str) -> None:
    """Always visible in ComfyUI console (print + INFO)."""
    print(msg, flush=True)
    logger.info(msg)


def record_lora_shape_skip(lora_name: str, key: str, reason: str) -> None:
    """Called from LoraDiff reshape/numel skip path."""
    _lora_shape_skips.append((str(lora_name), str(key), str(reason)))


def _basename_lora(name: str) -> str:
    if not name:
        return name
    return os.path.basename(str(name).replace("\\", "/"))


# WeightAdapterBase class attrs — NOT filenames (was the lora=lora bug)
_ADAPTER_TYPE_NAMES = frozenset({"lora", "loha", "lokr", "oft", "boft", "glora"})


def _looks_like_lora_filename(s) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or s.lower() in _ADAPTER_TYPE_NAMES:
        return False
    low = s.lower()
    if low.endswith((".safetensors", ".pt", ".ckpt", ".bin", ".sft")):
        return True
    if "/" in s or "\\" in s:
        return True
    # Short folder-relative names without extension still count as filenames
    if len(s) >= 2 and not s.startswith("diffusion_model"):
        return True
    return False


def _lora_line(msg: str) -> None:
    """One visible console line (print only — no print+logger twin)."""
    print(msg, flush=True)


def _slot_skip_count(entry: dict) -> int:
    return len(entry.get("not_mapped") or []) + len(
        entry.get("mapped_but_not_attached") or []
    )


def _slot_applied_count(entry: dict) -> int:
    return int(entry.get("applied_unet") or 0) + int(entry.get("applied_clip") or 0)


def _format_lora_slot_line(slot_i: int, entry: dict, include_bake: bool = False) -> str:
    """lora名 / 適用キー数 / スキップキー数 — always present."""
    name = entry.get("lora_name") or "(unknown)"
    sm = entry.get("strength_model")
    sc = entry.get("strength_clip")
    u = int(entry.get("applied_unet") or 0)
    c = int(entry.get("applied_clip") or 0)
    applied = u + c
    skip = _slot_skip_count(entry)
    parts = [
        f"Slot {slot_i}:",
        f"lora名='{name}'",
        f"適用キー数={applied} (unet={u} clip={c})",
        f"スキップキー数={skip}",
    ]
    if sm is not None:
        parts.append(f"strength_model={sm}")
    if sc is not None:
        parts.append(f"strength_clip={sc}")
    if include_bake:
        verdict, rq, ir, nb = _per_lora_bake_verdict(entry)
        parts.append(f"bake rq={rq} ir={ir} nb={nb}")
        if verdict == "OK_requant":
            parts.append("→ APPLIED ✓")
        elif verdict == "BROKEN_int8_round":
            parts.append("→ BROKEN ✗")
        elif verdict == "N/A_CLIP_only":
            parts.append("→ CLIP_only ✓")
        else:
            parts.append(f"→ {verdict}")
    else:
        if applied > 0:
            parts.append("→ APPLIED ✓")
        else:
            parts.append("→ SKIPPED ✗")
    return f"[HSWQ LoRA Status] {' | '.join(parts)}"


def _log_lora_slot_attach(entry: dict) -> None:
    """Emit one Status line immediately when a LoRA is attached (any loader)."""
    n = len(_lora_attach_history)
    if n == 1:
        _lora_line("[HSWQ LoRA Status] Processing LoRA slot(s):")
    _lora_line(_format_lora_slot_line(n, entry, include_bake=False))
    _lora_line(
        f"[HSWQ LoRA Status]   file_keys={entry.get('lora_file_keys', 0)} "
        f"mapped={entry.get('mapped_keys', 0)} "
        f"not_mapped={len(entry.get('not_mapped') or [])} "
        f"mapped_not_attached={len(entry.get('mapped_but_not_attached') or [])}"
    )


def _set_current_lora_name(name, strength_model=None, strength_clip=None) -> None:
    """Store real filename/UI name; never store adapter type 'lora'."""
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    if _looks_like_lora_filename(name):
        _current_lora_name = _basename_lora(name)
    if strength_model is not None:
        _current_lora_strength_model = strength_model
    if strength_clip is not None:
        _current_lora_strength_clip = strength_clip


def _path_is_under_loras_dir(path: str) -> bool:
    """True if path is inside any registered loras/ folder (any loader)."""
    if not path:
        return False
    try:
        import folder_paths

        bases = folder_paths.get_folder_paths("loras") or []
    except Exception:
        bases = []
    norm = os.path.normcase(os.path.abspath(str(path)))
    for base in bases:
        try:
            b = os.path.normcase(os.path.abspath(str(base)))
            if norm == b or norm.startswith(b + os.sep):
                return True
        except Exception:
            continue
    # Fallback when folder list not ready yet
    low = str(path).replace("\\", "/").lower()
    return "/loras/" in low or low.endswith("/loras")


def _resolve_lora_name(loaded_patches=None) -> str:
    """Filename for the LoRA currently being attached (any loader → common hooks)."""
    global _current_lora_name
    if _looks_like_lora_filename(_current_lora_name):
        return _basename_lora(_current_lora_name)

    try:
        import inspect

        # Common local names used by many LoRA loader nodes / helpers
        keys = (
            "lora_name",
            "lora_path",
            "lora",
            "path",
            "filename",
            "file_path",
            "lora_file",
            "name",
        )
        for frame in inspect.stack()[1:24]:
            loc = frame.frame.f_locals
            for key in keys:
                cand = loc.get(key)
                if _looks_like_lora_filename(cand):
                    return _basename_lora(cand)
            # Widget-style dicts: {'lora': '<file>', 'on': True, 'strength': ...}
            for cand in loc.values():
                if not isinstance(cand, dict):
                    continue
                ui = cand.get("lora")
                if _looks_like_lora_filename(ui) and (
                    "strength" in cand or "on" in cand or "strengthTwo" in cand
                ):
                    return _basename_lora(ui)
    except Exception:
        pass

    return f"unknown_lora#{len(_lora_attach_history) + 1}"


def reset_int8_lora_log_counters() -> None:
    global _lora_convert_logs, _lora_set_logs, _lora_patcher_logs
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    _lora_convert_logs = 0
    _lora_set_logs = 0
    _lora_patcher_logs = 0
    _lora_patcher_stats.update(
        calls=0, with_set_func=0, without_set_func=0, with_convert_func=0
    )
    _lora_shape_skips.clear()
    _lora_attach_history.clear()
    _lora_bake_by_key.clear()
    _current_lora_name = None
    _current_lora_strength_model = None
    _current_lora_strength_clip = None
    _lora_attach_last.update(
        lora_name="",
        strength_model=None,
        strength_clip=None,
        lora_file_keys=0,
        mapped_keys=0,
        applied_unet=0,
        applied_clip=0,
        applied_unet_keys=[],
        applied_clip_keys=[],
        not_mapped=[],
        mapped_but_not_attached=[],
        add_patches_skipped_unet=[],
    )
    dump_int8_lora_bake_stats._dumped_this_load = False


def summarize_int8_lora_capability(model) -> dict:
    """Scan loaded MODEL / diffusion_model and print LoRA hook readiness."""
    try:
        from comfy.ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = type(None)

    diffusion = model
    # ModelPatcher -> BaseModel -> diffusion_model
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        diffusion = model.model.diffusion_model
    elif hasattr(model, "diffusion_model"):
        diffusion = model.diffusion_model

    n_lin = n_conv = 0
    lin_set = conv_set = 0
    lin_cvt = conv_cvt = 0
    lin_q = conv_q = 0
    sample_missing = []

    for name, mod in diffusion.named_modules():
        cls = type(mod).__name__
        is_lin = "Linear" in cls
        is_conv = "Conv2d" in cls
        if not is_lin and not is_conv:
            continue
        has_set = callable(getattr(mod, "set_weight", None))
        has_cvt = callable(getattr(mod, "convert_weight", None))
        w = getattr(mod, "weight", None)
        is_q = False
        if QuantizedTensor is not type(None):
            is_q = isinstance(w, QuantizedTensor) or isinstance(
                getattr(w, "data", None), QuantizedTensor
            )
        layout = getattr(mod, "layout_type", None)
        if is_lin:
            n_lin += 1
            lin_set += int(has_set)
            lin_cvt += int(has_cvt)
            lin_q += int(is_q or layout is not None)
        else:
            n_conv += 1
            conv_set += int(has_set)
            conv_cvt += int(has_cvt)
            conv_q += int(is_q or layout is not None)
            if (not has_set or not has_cvt) and len(sample_missing) < 5:
                sample_missing.append(
                    f"{name} set={has_set} convert={has_cvt} layout={layout}"
                )

    _lora_line("[HSWQ INT8 LoRA] ===== load summary =====")
    _lora_line(
        f"[HSWQ INT8 LoRA] Linear: {n_lin}  set_weight={lin_set}  convert_weight={lin_cvt}  quantized/layout={lin_q}"
    )
    _lora_line(
        f"[HSWQ INT8 LoRA] Conv2d: {n_conv}  set_weight={conv_set}  convert_weight={conv_cvt}  quantized/layout={conv_q}"
    )
    if conv_set < n_conv or conv_cvt < n_conv:
        _lora_line(
            "[HSWQ INT8 LoRA] WARN: some Conv2d lack set/convert — LoRA on those layers will round into int8 and die"
        )
        for s in sample_missing:
            _lora_line(f"[HSWQ INT8 LoRA]   missing: {s}")
    else:
        _lora_line(
            "[HSWQ INT8 LoRA] OK: Conv2d has set_weight+convert_weight (dequant -> bake -> requant)"
        )
    _lora_line("[HSWQ INT8 LoRA] =========================")
    return {
        "linear": n_lin,
        "conv2d": n_conv,
        "linear_set_weight": lin_set,
        "conv_set_weight": conv_set,
    }


def decode_comfy_quant_conf(raw):
    """Decode a comfy_quant marker into a dict layer config."""
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if torch.is_tensor(raw):
        conf = json.loads(raw.numpy().tobytes())
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        conf = raw

    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        if parsed is conf:
            return {"format": conf}
        conf = parsed

    if isinstance(conf, dict):
        return conf
    raise TypeError(f"comfy_quant config must be a dict or format string, got {type(conf).__name__}")


def checkpoint_looks_like_comfy_quant_int8(state_dict_or_path) -> bool:
    """True if checkpoint has comfy_quant INT8 markers (native MixedPrecisionOps path).

    Accepts a loaded state_dict, or a filesystem path (probes via safetensors without full load).
    """
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_int8(str(state_dict_or_path))

    state_dict = state_dict_or_path
    has_marker = False
    has_int8 = False
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(".comfy_quant"):
            has_marker = True
            conf = decode_comfy_quant_conf(value)
            if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                return True
        if key.endswith(".weight") and value.dtype == torch.int8:
            has_int8 = True
    return has_marker and has_int8


def _probe_path_comfy_quant_int8(path: str) -> bool:
    """Lightweight safetensors probe for int8_tensorwise."""
    import torch

    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:16]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                    return True
            if comfy_keys:
                for k in keys:
                    if not k.endswith(".weight"):
                        continue
                    if f.get_tensor(k).dtype == torch.int8:
                        return True
                    break
            meta = f.metadata() or {}
            if "_quantization_metadata" in meta:
                try:
                    qm = json.loads(meta["_quantization_metadata"])
                    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
                    for v in layers.values():
                        if isinstance(v, str) and v == "int8_tensorwise":
                            return True
                        if isinstance(v, dict) and v.get("format") == "int8_tensorwise":
                            return True
                except (TypeError, json.JSONDecodeError):
                    pass
    except Exception as e:
        logger.debug("[HSWQ INT8] probe failed for %s: %s", path, e)
        return False
    return False


def _normalize_comfy_quant_tensor(value):
    import torch

    conf = decode_comfy_quant_conf(value)
    if conf is None:
        return None
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _patch_convert_old_quants() -> bool:
    try:
        import torch
        import comfy.utils as utils_module
    except ImportError:
        return False

    original = getattr(utils_module, "convert_old_quants", None)
    if original is None or getattr(original, "_hswq_int8_patched", False):
        return False

    def convert_old_quants_pre(state_dict, model_prefix="", metadata=None):
        if metadata is None:
            metadata = {}
        # Normalize string layer configs in metadata before upstream json.dumps(v).
        if isinstance(metadata, dict) and "_quantization_metadata" in metadata:
            try:
                quant_meta = json.loads(metadata["_quantization_metadata"])
            except (TypeError, json.JSONDecodeError):
                quant_meta = None
            if isinstance(quant_meta, dict) and isinstance(quant_meta.get("layers"), dict):
                layers = quant_meta["layers"]
                changed = False
                for k, v in list(layers.items()):
                    if isinstance(v, str):
                        layers[k] = {"format": v}
                        changed = True
                    elif not isinstance(v, dict):
                        raise TypeError(
                            f"quantization layer config for {k} must be dict or format string, got {type(v).__name__}"
                        )
                if changed:
                    metadata = dict(metadata)
                    metadata["_quantization_metadata"] = json.dumps(quant_meta)

        state_dict, metadata = original(state_dict, model_prefix=model_prefix, metadata=metadata)

        # Re-normalize any .comfy_quant tensors (file-embedded or metadata-written).
        for key in list(state_dict.keys()):
            if not key.endswith(".comfy_quant"):
                continue
            normalized = _normalize_comfy_quant_tensor(state_dict[key])
            if normalized is None:
                state_dict.pop(key, None)
            else:
                state_dict[key] = normalized
        return state_dict, metadata

    convert_old_quants_pre._hswq_int8_patched = True
    utils_module.convert_old_quants = convert_old_quants_pre
    return True


def _make_quantized_conv2d(ops_module, MixedPrecisionOps, disabled):
    """Build MixedPrecisionOps.Conv2d class using current comfy.ops helpers."""
    import torch

    CastWeightBiasOp = ops_module.CastWeightBiasOp
    QuantizedTensor = ops_module.QuantizedTensor
    cast_bias_weight = ops_module.cast_bias_weight
    uncast_bias_weight = ops_module.uncast_bias_weight
    run_every_op = ops_module.run_every_op
    _load_quantized_module = ops_module._load_quantized_module
    _quantized_weight_state_dict = ops_module._quantized_weight_state_dict
    _quantized_apply = ops_module._quantized_apply

    class Conv2d(torch.nn.Module, CastWeightBiasOp):
        _disabled_formats = disabled
        _hswq_quant_conv2d = True

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        ):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.padding_mode = padding_mode
            self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
            self._orig_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])

            if bias:
                self.bias = torch.nn.Parameter(
                    torch.empty(out_channels, **self.factory_kwargs), requires_grad=False
                )
            else:
                self.register_parameter("bias", None)

            self.weight = None
            self.quant_format = None
            self.layout_type = None
            self._full_precision_mm = MixedPrecisionOps._full_precision_mm
            self._full_precision_mm_config = False

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, *args):
            _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=False)

        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            return _quantized_weight_state_dict(self, sd, prefix)

        def _conv_forward(self, input, weight, bias):
            if self.padding_mode != "zeros":
                return torch.nn.functional.conv2d(
                    torch.nn.functional.pad(
                        input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    weight,
                    bias,
                    self.stride,
                    (0, 0),
                    self.dilation,
                    self.groups,
                )
            return torch.nn.functional.conv2d(
                input, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

        def forward_comfy_cast_weights(self, input):
            # Mirror MixedPrecision Linear: when weight is QuantizedTensor and
            # Dynamic VRAM uses weight_lowvram_function, want_requant=True so
            # post_cast dequant → LoRA → requant (want_requant=False left QT
            # in the resident path after the first step and killed LoRA).
            want_requant = isinstance(getattr(self, "weight", None), QuantizedTensor)
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input,
                offloadable=True,
                compute_dtype=getattr(input, "dtype", None),
                want_requant=want_requant,
            )
            x = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, input, *args, **kwargs):
            run_every_op()
            return self.forward_comfy_cast_weights(input)

        def convert_weight(self, weight, inplace=False, **kwargs):
            # Same contract as MixedPrecisionOps.Linear: LoRA / ModelPatcher
            # dequant → calculate_weight → set_weight (see ComfyUI-INT8-Fast bake path).
            global _lora_convert_logs
            out = weight.dequantize() if isinstance(weight, QuantizedTensor) else weight
            if _lora_convert_logs < _LORA_CONVERT_LOG_MAX:
                _lora_convert_logs += 1
                wdtype = getattr(weight, "dtype", None)
                odtype = getattr(out, "dtype", None)
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.convert_weight #{_lora_convert_logs}: "
                    f"in={type(weight).__name__}/{wdtype} -> out={type(out).__name__}/{odtype} "
                    f"layout={getattr(self, 'layout_type', None)}"
                )
            return out

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            # Mirror MixedPrecisionOps.Linear.set_weight so Conv2d LoRA bake
            # does not fall through to stochastic_rounding(..., int8), which
            # destroys float LoRA deltas (INT8-Fast: normal LoRA loader works).
            global _lora_set_logs
            layout = getattr(self, "layout_type", None)
            path = "requant" if layout is not None else "cast_only"
            if _lora_set_logs < _LORA_SET_LOG_MAX:
                _lora_set_logs += 1
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.set_weight #{_lora_set_logs}: "
                    f"path={path} float_in={getattr(weight, 'dtype', None)} "
                    f"shape={tuple(weight.shape) if hasattr(weight, 'shape') else '?'} "
                    f"seed={seed} layout={layout}"
                )
            if layout is not None:
                weight = self.weight.requantize_from_float(
                    weight,
                    scale="recalculate",
                    stochastic_rounding=seed,
                    inplace_ops=True,
                ).to(self.weight.dtype)
            else:
                weight = weight.to(self.weight.dtype)
            if return_weight:
                return weight

            assert inplace_update is False
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def _apply(self, fn, recurse=True):
            return _quantized_apply(self, fn, recurse)

        @property
        def _reversed_padding_repeated_twice(self):
            return tuple(x for x in reversed(self.padding) for _ in range(2))

    return Conv2d


def _patch_ops_decode_and_conv() -> bool:
    try:
        import comfy.ops as ops_module
    except ImportError:
        return False

    ops_module._decode_comfy_quant_conf = decode_comfy_quant_conf

    original_load = getattr(ops_module, "_load_quantized_module", None)
    if original_load is None:
        return False

    if not getattr(original_load, "_hswq_int8_decode_patched", False):

        def _load_quantized_module_patched(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=False,
        ):
            key = f"{prefix}comfy_quant"
            if key in state_dict:
                normalized = _normalize_comfy_quant_tensor(state_dict[key])
                if normalized is None:
                    state_dict.pop(key, None)
                else:
                    state_dict[key] = normalized
            return original_load(
                module,
                super_load,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
                load_extra_params=load_extra_params,
            )

        _load_quantized_module_patched._hswq_int8_decode_patched = True
        ops_module._load_quantized_module = _load_quantized_module_patched

    # Also normalize Embedding's direct json.loads path by wrapping Embedding._load_from_state_dict
    # is covered if convert_old_quants + file markers are normalized; keep load wrapper as safety.

    original_mp = getattr(ops_module, "mixed_precision_ops", None)
    if original_mp is None or not callable(original_mp):
        return False
    if getattr(original_mp, "_hswq_int8_conv_patched", False):
        return True

    def mixed_precision_ops_force_conv(
        quant_config=None, compute_dtype=None, full_precision_mm=False, disabled=None
    ):
        if quant_config is None:
            quant_config = {}
        if compute_dtype is None:
            import torch

            compute_dtype = torch.bfloat16
        if disabled is None:
            disabled = []
        result = original_mp(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        # Always replace inherited manual_cast.Conv2d with INT8-capable Conv2d.
        result.Conv2d = _make_quantized_conv2d(ops_module, result, disabled)
        return result

    mixed_precision_ops_force_conv._hswq_int8_conv_patched = True
    ops_module.mixed_precision_ops = mixed_precision_ops_force_conv
    return True


def _patch_lowvram_patch_float_intermediate() -> bool:
    """Fix LowVramPatch intermediate_dtype for INT8 / QuantizedTensor weights.

    Upstream LowVramPatch passes intermediate_dtype=weight.dtype. When the
    weight is still int8/Char (or a QuantizedTensor), LoRA matmul casts to
    int8 and either errors or silently produces a no-op delta — same bug as
    BobJohnson24/ComfyUI-INT8-Fast#76.
    """
    try:
        import torch
        import comfy.lora
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    LowVramPatch = getattr(mp, "LowVramPatch", None)
    if LowVramPatch is None:
        return False
    original = getattr(LowVramPatch, "__call__", None)
    if original is None or getattr(original, "_hswq_int8_lora_dtype", False):
        return getattr(original, "_hswq_int8_lora_dtype", False)

    def __call__(self, weight):
        patches = (
            self.prepared_patches
            if self.prepared_patches is not None
            else self.patches[self.key]
        )
        w = weight
        if isinstance(w, QuantizedTensor):
            w = w.dequantize()
        dtype = getattr(w, "dtype", None)
        if dtype is not None and hasattr(dtype, "is_floating_point") and not dtype.is_floating_point:
            idtype = torch.float32
        elif dtype is not None and hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
            idtype = dtype
        else:
            idtype = torch.float32
        return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype)

    __call__._hswq_int8_lora_dtype = True
    LowVramPatch.__call__ = __call__
    return True


def _get_baked_key_set(model) -> set:
    s = getattr(model, "_hswq_int8_baked_keys", None)
    if s is None:
        s = set()
        model._hswq_int8_baked_keys = s
    return s


def _maybe_invalidate_baked_keys(patcher) -> None:
    """If patches_uuid changed (new LoRA), allow those keys to be baked again."""
    model = patcher.model
    baked_uuid = getattr(model, "_hswq_int8_baked_uuid", None)
    cur = getattr(patcher, "patches_uuid", None)
    if baked_uuid is None or cur is None:
        return
    if baked_uuid != cur and patcher.patches:
        _get_baked_key_set(model).clear()
        model._hswq_int8_baked_uuid = None


def _strip_lowvram_for_baked_keys(patcher) -> int:
    """Dynamic.load re-attaches LowVramPatch; clear it for already-baked keys.

    Shared modules keep their VBAR ``_v`` across loads. Re-attaching LoRA on
    top of baked INT8 weights would double-apply; clearing lowvram avoids that.
    """
    _maybe_invalidate_baked_keys(patcher)
    baked = getattr(patcher.model, "_hswq_int8_baked_keys", None)
    if not baked:
        return 0
    cleared = 0
    for name, module in patcher.model.named_modules():
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in baked:
                continue
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
                cleared += 1
            # Drop from this patcher's dict so later loads do not re-attach
            if key in patcher.patches:
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
    return cleared


def _bake_int8_patches_on_dynamic_patcher(patcher, device_to) -> int:
    """Bake LoRA into INT8 modules after ModelPatcherDynamic.load.

    Dynamic VRAM attaches LowVramPatch on weight_lowvram_function and asserts
    force_patch_weights=False. For comfy_quant INT8 that path often leaves
    LoRA attached in the patcher dict but visually inert (keys count OK,
    bake logs absent). We bake via convert_weight/set_weight (requant).

    Critical VBAR rule (2nd-gen FaceDetailer OOM):
      ModelVBAR.alloc is a bump allocator (offset only grows). Deleting
      module._v after bake makes the next load call alloc() again → VBAR OOM.
      Keep ``_v``. Clear LowVramPatch, bake, then pop patches + drop the
      pre-bake backup entry so restore_loaded_backups does not undo bake.
    """
    if not getattr(patcher, "patches", None):
        return 0
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return 0

    _maybe_invalidate_baked_keys(patcher)
    already = _get_baked_key_set(patcher.model)
    baked = 0
    for name, module in patcher.model.named_modules():
        keys_to_bake = []
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in patcher.patches:
                continue
            if key in already:
                # Already baked under this patches_uuid; clear re-attached LowVramPatch
                attr = param_key + "_lowvram_function"
                if getattr(module, attr, None) is not None:
                    setattr(module, attr, None)
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
                continue
            weight, set_func, convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            is_qt = isinstance(weight, QuantizedTensor)
            # Bake when requant hooks exist (Linear/Conv INT8), or QT without
            # set_func would still be broken — skip those and log below.
            if set_func is None:
                if is_qt:
                    _console(
                        f"[HSWQ INT8 LoRA] WARN cannot bake {key}: "
                        "QuantizedTensor but no set_weight (int8_round risk)"
                    )
                continue
            keys_to_bake.append((param_key, key))

        if not keys_to_bake:
            continue

        # Clear LowVramPatch so bake uses Parameter + set_weight, not lazy patch.
        # Do NOT unpin/delete module._v — that causes 2nd-load VBAR OOM.
        for param_key, _key in keys_to_bake:
            if hasattr(module, param_key + "_lowvram_function"):
                setattr(module, param_key + "_lowvram_function", None)

        for _param_key, key in keys_to_bake:
            patcher.patch_weight_to_device(key, device_to=device_to)
            # Drop pre-bake backup so the next Dynamic.load restore keeps baked weights
            if key in patcher.backup:
                try:
                    del patcher.backup[key]
                except KeyError:
                    pass
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            already.add(key)
            baked += 1

    if baked > 0:
        patcher.model._hswq_int8_baked_uuid = getattr(patcher, "patches_uuid", None)

    return baked


def _patch_model_patcher_dynamic_int8_lora_bake() -> bool:
    """After ModelPatcherDynamic.load, bake INT8 LoRA via set_weight."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    _DYN_VER = 4
    if getattr(original, "_hswq_int8_lora_bake_ver", 0) >= _DYN_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_dynamic_load", original)

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False, dirty=False):
        result = true_orig(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        # Load re-attaches LowVramPatch for any keys still in patches / clones
        _strip_lowvram_for_baked_keys(self)
        if self.patches:
            n = _bake_int8_patches_on_dynamic_patcher(self, device_to=device_to)
            if n > 0 or _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
                dump_int8_lora_bake_stats(force=True)
        elif _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
            # Patches already consumed by a prior bake; still emit Status once if needed
            dump_int8_lora_bake_stats(force=False)
        return result

    load._hswq_int8_lora_bake = True
    load._hswq_int8_lora_bake_ver = _DYN_VER
    load._hswq_orig_dynamic_load = true_orig
    Dynamic.load = load
    return True


def _patch_model_patcher_lora_logs() -> bool:
    """Log whether LoRA bake uses set_weight (requant) or int8_round fallback."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    original = getattr(mp.ModelPatcher, "patch_weight_to_device", None)
    if original is None or getattr(original, "_hswq_int8_lora_log", False):
        return getattr(original, "_hswq_int8_lora_log", False)

    def patch_weight_to_device_logged(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        global _lora_patcher_logs
        weight, set_func, convert_func = mp.get_key_weight(self.model, key)
        if key in self.patches:
            _lora_patcher_stats["calls"] += 1
            if set_func is not None:
                _lora_patcher_stats["with_set_func"] += 1
            else:
                _lora_patcher_stats["without_set_func"] += 1
            if convert_func is not None:
                _lora_patcher_stats["with_convert_func"] += 1

            path = "requant" if set_func is not None else "int8_round"
            _lora_bake_by_key[key] = path
            if _lora_patcher_logs < _LORA_PATCHER_LOG_MAX:
                _lora_patcher_logs += 1
                wdtype = getattr(weight, "dtype", None)
                warn = ""
                if set_func is None and wdtype is not None and str(wdtype) in ("torch.int8", "int8"):
                    warn = "  << BROKEN for INT8 (LoRA delta will be destroyed)"
                owners = [
                    e["lora_name"]
                    for e in _lora_attach_history
                    if key in (e.get("applied_unet_keys") or [])
                ]
                owner_s = ",".join(owners[:3]) if owners else "-"
                if len(owners) > 3:
                    owner_s += f"+{len(owners) - 3}"
                _console(
                    f"[HSWQ INT8 LoRA] bake #{_lora_patcher_logs}: key={key} "
                    f"path={path} lora={owner_s} weight_dtype={wdtype} "
                    f"convert={'yes' if convert_func else 'no'} "
                    f"set={'yes' if set_func else 'no'}{warn}"
                )
            # After stacked UNet keys are baked, dump per-LoRA summary once
            target = sum(int(e.get("applied_unet") or 0) for e in _lora_attach_history)
            if target <= 0:
                target = int(_lora_attach_last.get("applied_unet") or 0)
            # Unique baked keys may be less than sum (shared keys across LoRAs)
            unique_target = len(
                {
                    k
                    for e in _lora_attach_history
                    for k in (e.get("applied_unet_keys") or [])
                }
            ) or target
            if (
                unique_target > 0
                and _lora_patcher_stats["calls"] >= unique_target
                and not getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False)
            ):
                # Do NOT set the flag before dump (that made dump a no-op).
                dump_int8_lora_bake_stats(force=False)


        return original(
            self,
            key,
            device_to=device_to,
            inplace_update=inplace_update,
            return_weight=return_weight,
            force_cast=force_cast,
        )

    patch_weight_to_device_logged._hswq_int8_lora_log = True
    mp.ModelPatcher.patch_weight_to_device = patch_weight_to_device_logged
    return True


def _per_lora_bake_verdict(entry: dict) -> tuple[str, int, int, int]:
    """Return (verdict, requant, int8_round, not_baked) for one LoRA attach entry."""
    unet_keys = entry.get("applied_unet_keys") or []
    clip_n = int(entry.get("applied_clip") or 0)
    unet_n = int(entry.get("applied_unet") or 0)
    if unet_n == 0 and clip_n > 0:
        return ("N/A_CLIP_only", 0, 0, 0)
    if unet_n == 0:
        return ("SKIP_no_keys", 0, 0, 0)
    requant = 0
    int8_round = 0
    not_baked = 0
    for k in unet_keys:
        path = _lora_bake_by_key.get(k)
        if path == "requant":
            requant += 1
        elif path == "int8_round":
            int8_round += 1
        else:
            not_baked += 1
    if int8_round > 0:
        return ("BROKEN_int8_round", requant, int8_round, not_baked)
    if requant == 0 and not_baked == unet_n:
        return ("WARN_not_baked_yet", requant, int8_round, not_baked)
    if requant > 0 and int8_round == 0:
        return ("OK_requant", requant, int8_round, not_baked)
    return ("PARTIAL", requant, int8_round, not_baked)


def dump_int8_lora_bake_stats(force: bool = False) -> None:
    """Full Status dump: lora名 / 適用キー数 / スキップキー数 (+ bake if any)."""
    if not force and getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False):
        return
    dump_int8_lora_bake_stats._dumped_this_load = True

    history = list(_lora_attach_history) if _lora_attach_history else []
    if not history and (_lora_attach_last.get("mapped_keys") or 0) > 0:
        history = [dict(_lora_attach_last)]

    n = len(history)
    _lora_line(f"[HSWQ LoRA Status] ===== bake summary ({n} slot(s)) =====")
    if not history:
        _lora_line(
            "[HSWQ LoRA Status] Slot -: | lora名='(none)' | 適用キー数=0 | スキップキー数=0 | → SKIPPED ✗"
        )
    ok_n = 0
    for i, a in enumerate(history, 1):
        line = _format_lora_slot_line(i, a, include_bake=True)
        _lora_line(line)
        verdict, _rq, _ir, _nb = _per_lora_bake_verdict(a)
        if verdict in ("OK_requant", "N/A_CLIP_only") or _slot_applied_count(a) > 0:
            if verdict != "BROKEN_int8_round":
                ok_n += 1
    _lora_line(
        f"[HSWQ LoRA Status] Summary: {ok_n}/{n} LoRA(s) with applied keys"
    )

    s = _lora_patcher_stats
    if s["calls"] == 0:
        _lora_line("[HSWQ LoRA Bake] not yet (model not on GPU)")
        return
    _lora_line(
        f"[HSWQ LoRA Bake] total={s['calls']} requant={s['with_set_func']} "
        f"int8_round={s['without_set_func']} shape_skip={len(_lora_shape_skips)}"
    )
    if s["without_set_func"] > 0:
        _lora_line(
            "[HSWQ LoRA Bake] WARNING: int8_round used — those layers are broken"
        )
    else:
        _lora_line("[HSWQ LoRA Bake] path OK (all requant)")
    if _lora_shape_skips:
        for name, key, reason in _lora_shape_skips[:_LORA_SKIP_PRINT_MAX]:
            _lora_line(
                f"[HSWQ LoRA Bake] shape_skip | '{name}' | {key} | {reason}"
            )


def _patch_lora_loader_name_context() -> bool:
    """Capture name from nodes.LoraLoader when any node calls it."""
    try:
        import nodes as nodes_mod
    except ImportError:
        return False

    LoraLoader = getattr(nodes_mod, "LoraLoader", None)
    if LoraLoader is None:
        return False
    original = getattr(LoraLoader, "load_lora", None)
    if original is None:
        return False
    _NAME_VER = 6
    if getattr(original, "_hswq_lora_name_ctx_ver", 0) >= _NAME_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_lora", original)

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
        prev = (
            _current_lora_name,
            _current_lora_strength_model,
            _current_lora_strength_clip,
        )
        _set_current_lora_name(lora_name, strength_model, strength_clip)
        try:
            return true_orig(self, model, clip, lora_name, strength_model, strength_clip)
        finally:
            (
                _current_lora_name,
                _current_lora_strength_model,
                _current_lora_strength_clip,
            ) = prev

    load_lora._hswq_lora_name_ctx = True
    load_lora._hswq_lora_name_ctx_ver = _NAME_VER
    load_lora._hswq_orig_load_lora = true_orig
    LoraLoader.load_lora = load_lora
    return True


def _patch_loras_folder_path_name() -> bool:
    """Any loader that resolves folder_paths 'loras' → capture filename."""
    try:
        import folder_paths
    except ImportError:
        return False

    _PATH_VER = 3
    ok = False

    for fname in ("get_full_path", "get_full_path_or_raise"):
        original = getattr(folder_paths, fname, None)
        if original is None:
            continue
        if getattr(original, "_hswq_lora_path_name_ver", 0) >= _PATH_VER:
            ok = True
            continue
        true_orig = getattr(original, "_hswq_orig_get_full_path", original)

        def _make(orig):
            def wrapped(folder_name, filename):
                if folder_name == "loras":
                    _set_current_lora_name(filename)
                return orig(folder_name, filename)

            wrapped._hswq_lora_path_name_ver = _PATH_VER
            wrapped._hswq_orig_get_full_path = orig
            return wrapped

        setattr(folder_paths, fname, _make(true_orig))
        ok = True
    return ok


def _patch_load_torch_file_lora_name() -> bool:
    """Any loader that load_torch_file(lora_path) → capture basename."""
    try:
        import comfy.utils as utils_mod
    except ImportError:
        return False
    original = getattr(utils_mod, "load_torch_file", None)
    if original is None:
        return False
    _TORCH_VER = 1
    if getattr(original, "_hswq_lora_torch_name_ver", 0) >= _TORCH_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_torch_file", original)

    def load_torch_file(ckpt, *args, **kwargs):
        if isinstance(ckpt, (str, os.PathLike)):
            p = str(ckpt)
            if _path_is_under_loras_dir(p):
                _set_current_lora_name(p)
        return true_orig(ckpt, *args, **kwargs)

    load_torch_file._hswq_lora_torch_name_ver = _TORCH_VER
    load_torch_file._hswq_orig_load_torch_file = true_orig
    utils_mod.load_torch_file = load_torch_file
    return True


def _patch_load_lora_key_counts() -> bool:
    """Wrap load_lora + load_lora_for_models for applied/skipped key counts."""
    try:
        import comfy.lora as lora_mod
        import comfy.sd as sd_mod
        import comfy.weight_adapter as weight_adapter
    except ImportError:
        return False

    orig_load_lora = getattr(lora_mod, "load_lora", None)
    orig_for_models = getattr(sd_mod, "load_lora_for_models", None)
    if orig_load_lora is None or orig_for_models is None:
        return False

    _KEY_VER = 6
    if getattr(orig_for_models, "_hswq_lora_key_count_ver", 0) >= _KEY_VER:
        _patch_lora_loader_name_context()
        _patch_loras_folder_path_name()
        _patch_load_torch_file_lora_name()
        return True

    if getattr(orig_for_models, "_hswq_lora_key_count", False):
        orig_for_models = getattr(
            orig_for_models, "_hswq_orig_for_models", orig_for_models
        )
    if getattr(orig_load_lora, "_hswq_lora_key_count", False):
        orig_load_lora = getattr(orig_load_lora, "_hswq_orig_load_lora", orig_load_lora)

    _ctx = {"patch_dict": {}, "not_mapped": [], "file_keys": 0}

    def load_lora_counted(lora, to_load, log_missing=True):
        patch_dict = {}
        loaded_keys = set()
        for x in to_load:
            alpha_name = "{}.alpha".format(x)
            alpha = None
            if alpha_name in lora.keys():
                alpha = lora[alpha_name].item()
                loaded_keys.add(alpha_name)

            dora_scale_name = "{}.dora_scale".format(x)
            dora_scale = None
            if dora_scale_name in lora.keys():
                dora_scale = lora[dora_scale_name]
                loaded_keys.add(dora_scale_name)

            for adapter_cls in weight_adapter.adapters:
                adapter = adapter_cls.load(x, lora, alpha, dora_scale, loaded_keys)
                if adapter is not None:
                    patch_dict[to_load[x]] = adapter
                    loaded_keys.update(adapter.loaded_keys)
                    continue

            w_norm_name = "{}.w_norm".format(x)
            b_norm_name = "{}.b_norm".format(x)
            w_norm = lora.get(w_norm_name, None)
            b_norm = lora.get(b_norm_name, None)

            if w_norm is not None:
                loaded_keys.add(w_norm_name)
                patch_dict[to_load[x]] = ("diff", (w_norm,))
                if b_norm is not None:
                    loaded_keys.add(b_norm_name)
                    patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                        "diff",
                        (b_norm,),
                    )

            diff_name = "{}.diff".format(x)
            diff_weight = lora.get(diff_name, None)
            if diff_weight is not None:
                patch_dict[to_load[x]] = ("diff", (diff_weight,))
                loaded_keys.add(diff_name)

            diff_bias_name = "{}.diff_b".format(x)
            diff_bias = lora.get(diff_bias_name, None)
            if diff_bias is not None:
                patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                    "diff",
                    (diff_bias,),
                )
                loaded_keys.add(diff_bias_name)

            set_weight_name = "{}.set_weight".format(x)
            set_weight = lora.get(set_weight_name, None)
            if set_weight is not None:
                patch_dict[to_load[x]] = ("set", (set_weight,))
                loaded_keys.add(set_weight_name)

        not_mapped = [x for x in lora.keys() if x not in loaded_keys]
        _ctx["patch_dict"] = patch_dict
        _ctx["not_mapped"] = not_mapped
        _ctx["file_keys"] = len(lora) if hasattr(lora, "keys") else 0

        if log_missing:
            for x in not_mapped:
                logging.warning("lora key not loaded: {}".format(x))

        return patch_dict

    def load_lora_for_models_counted(
        model, clip, lora, strength_model, strength_clip, lora_metadata=None
    ):
        new_model, new_clip = orig_for_models(
            model, clip, lora, strength_model, strength_clip, lora_metadata
        )
        loaded = _ctx.get("patch_dict") or {}
        not_mapped = list(_ctx.get("not_mapped") or [])
        file_key_count = int(_ctx.get("file_keys") or 0)
        lora_name = _resolve_lora_name(loaded)

        unet_keys = set(new_model.patches.keys()) if new_model is not None else set()
        if new_clip is not None and hasattr(new_clip, "patcher"):
            clip_keys = set(new_clip.patcher.patches.keys())
        else:
            clip_keys = set()

        applied_unet_keys = []
        applied_clip_keys = []
        mapped_but_not = []
        add_patches_miss = []
        for x in loaded:
            key = x if isinstance(x, str) else x[0]
            in_u = key in unet_keys
            in_c = key in clip_keys
            if in_u:
                applied_unet_keys.append(key)
            if in_c:
                applied_clip_keys.append(key)
            if not in_u and not in_c:
                mapped_but_not.append(x)
                add_patches_miss.append(x)

        applied_unet = len(applied_unet_keys)
        applied_clip = len(applied_clip_keys)

        entry = {
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "lora_file_keys": file_key_count,
            "mapped_keys": len(loaded),
            "applied_unet": applied_unet,
            "applied_clip": applied_clip,
            "applied_unet_keys": list(applied_unet_keys),
            "applied_clip_keys": list(applied_clip_keys),
            "not_mapped": sorted(str(x) for x in not_mapped),
            "mapped_but_not_attached": list(mapped_but_not),
            "add_patches_skipped_unet": list(add_patches_miss),
        }
        _lora_attach_last.update(entry)
        _lora_attach_history.append(dict(entry))
        _log_lora_slot_attach(entry)
        return (new_model, new_clip)

    load_lora_counted._hswq_lora_key_count = True
    load_lora_counted._hswq_orig_load_lora = orig_load_lora
    load_lora_for_models_counted._hswq_lora_key_count = True
    load_lora_for_models_counted._hswq_lora_key_count_ver = _KEY_VER
    load_lora_for_models_counted._hswq_orig_for_models = orig_for_models
    lora_mod.load_lora = load_lora_counted
    sd_mod.load_lora_for_models = load_lora_for_models_counted
    _patch_lora_loader_name_context()
    _patch_loras_folder_path_name()
    _patch_load_torch_file_lora_name()
    return True


def apply_comfy_quant_int8_patches() -> bool:
    """Install INT8 comfy_quant patches once. Returns True if applied (or already applied)."""
    global _PATCHES_APPLIED
    ok_keys = _patch_load_lora_key_counts()
    ok_name = _patch_lora_loader_name_context()
    ok_path = _patch_loras_folder_path_name()
    ok_torch = _patch_load_torch_file_lora_name()
    ok_lowvram = _patch_lowvram_patch_float_intermediate()
    ok_dyn_bake = _patch_model_patcher_dynamic_int8_lora_bake()
    if _PATCHES_APPLIED:
        return True
    ok_utils = _patch_convert_old_quants()
    ok_ops = _patch_ops_decode_and_conv()
    ok_lora_log = _patch_model_patcher_lora_logs()
    if ok_ops:
        _PATCHES_APPLIED = True
        _console(
            "[HSWQ INT8] comfy_quant patches applied "
            f"(Conv2d quant load + decode"
            f"{' + convert_old_quants' if ok_utils else ''}"
            f"{' + LoRA bake logs' if ok_lora_log else ''}"
            f"{' + LoRA key counts' if ok_keys else ''}"
            f"{' + LoRA name' if ok_name or ok_path or ok_torch else ''}"
            f"{' + LowVramPatch float dtype' if ok_lowvram else ''}"
            f"{' + Dynamic INT8 LoRA bake' if ok_dyn_bake else ''})"
        )
        return True
    logger.warning(
        "[HSWQ INT8] Failed to apply comfy_quant patches (ops=%s utils=%s)",
        ok_ops,
        ok_utils,
    )
    return False
```

### 3.2 `__init__.py`（全文・946 行）

```python
import logging
import os
from pathlib import Path

__version__ = "3.1.8"

import torch
import yaml
from packaging.version import InvalidVersion, Version

# SDXL MultiGPU support (from comfyui-multigpu)
import comfy.model_management as mm

# vanilla and LTS compatibility snippet
try:
    from comfy_compatibility.vanilla import prepare_vanilla_environment

    prepare_vanilla_environment()

    from comfy.model_downloader import add_known_models
    from comfy.model_downloader_types import HuggingFile

    capability = torch.cuda.get_device_capability(0 if torch.cuda.is_available() else None)
    sm = f"{capability[0]}{capability[1]}"
    precision = "fp4" if sm == "120" else "int4"

    # add known models

    models_yaml_path = Path(__file__).parent / "test_data" / "models.yaml"
    with open(models_yaml_path, "r") as f:
        nunchaku_models_yaml = yaml.safe_load(f)

    NUNCHAKU_SVDQ_MODELS = []
    for model in nunchaku_models_yaml["models"]:
        filename = model["filename"]
        if not filename.startswith("svdq-"):
            continue
        if "{precision}" in filename:
            filename = filename.format(precision=precision)
        NUNCHAKU_SVDQ_MODELS.append(HuggingFile(repo_id=model["repo_id"], filename=filename))

    NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS = [
        HuggingFile(repo_id="nunchaku-tech/nunchaku-t5", filename="awq-int4-flux.1-t5xxl.safetensors"),
    ]

    add_known_models("diffusion_models", *NUNCHAKU_SVDQ_MODELS)
    add_known_models("text_encoders", *NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS)
except (ImportError, ModuleNotFoundError):
    pass

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-nunchaku Initialization " + "=" * 40)

# SDXL MultiGPU initialization
sdxl_logger = logging.getLogger("SDXL")
sdxl_logger.propagate = False
if not sdxl_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    sdxl_logger.addHandler(handler)
    sdxl_logger.setLevel(logging.INFO)

# SDXL device management
current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()
current_unet_offload_device = mm.unet_offload_device()

def set_current_device(device):
    """Set the current device context for SDXL operations."""
    global current_device
    current_device = device
    sdxl_logger.debug(f"[SDXL Initialization] current_device set to: {device}")

def set_current_text_encoder_device(device):
    """Set the current text encoder device context for CLIP models."""
    global current_text_encoder_device
    current_text_encoder_device = device
    sdxl_logger.debug(f"[SDXL Initialization] current_text_encoder_device set to: {device}")

def set_current_unet_offload_device(device):
    """Set the current UNet offload device context."""
    global current_unet_offload_device
    current_unet_offload_device = device
    sdxl_logger.debug(f"[SDXL Initialization] current_unet_offload_device set to: {device}")

def get_current_device():
    """Get the current device context for SDXL operations at runtime."""
    return current_device

def get_current_text_encoder_device():
    """Get the current text encoder device context for CLIP models at runtime."""
    return current_text_encoder_device

def get_current_unet_offload_device():
    """Get the current UNet offload device context at runtime."""
    return current_unet_offload_device

def get_torch_device_patched():
    """Return SDXL-aware device selection for patched mm.get_torch_device."""
    from .device_utils import get_device_list, is_accelerator_available
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_device) if str(current_device) in devs else torch.device("cpu")
    sdxl_logger.debug(f"[SDXL Core Patching] get_torch_device_patched returning device: {device} (current_device={current_device})")
    return device

def text_encoder_device_patched():
    """Return SDXL-aware text encoder device for patched mm.text_encoder_device."""
    from .device_utils import get_device_list, is_accelerator_available
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_text_encoder_device) if str(current_text_encoder_device) in devs else torch.device("cpu")
    sdxl_logger.info(f"[SDXL Core Patching] text_encoder_device_patched returning device: {device} (current_text_encoder_device={current_text_encoder_device})")
    return device

def unet_offload_device_patched():
    """Return SDXL-aware UNet offload device for patched mm.unet_offload_device."""
    from .device_utils import get_device_list, is_accelerator_available
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_unet_offload_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_unet_offload_device) if str(current_unet_offload_device) in devs else torch.device("cpu")
    sdxl_logger.debug(f"[SDXL Core Patching] unet_offload_device_patched returning device: {device} (current_unet_offload_device={current_unet_offload_device})")
    return device

sdxl_logger.info(f"[SDXL Core Patching] Patching mm.get_torch_device, mm.text_encoder_device, mm.unet_offload_device")
sdxl_logger.info(f"[SDXL DEBUG] Initial current_device: {current_device}")
sdxl_logger.info(f"[SDXL DEBUG] Initial current_text_encoder_device: {current_text_encoder_device}")
sdxl_logger.info(f"[SDXL DEBUG] Initial current_unet_offload_device: {current_unet_offload_device}")

mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched
mm.unet_offload_device = unet_offload_device_patched

from .utils import get_package_version, get_plugin_version

# -------------------------------------------------------------------------
# HSWQ Pin Cache: comfy.pinned_memory の pin_memory / unpin_memory を
# monkey-patch し、cudaHostRegister/Unregister の繰り返しを回避する。
#
# FaceDetailer 等でモデルが UNet → VAE → UNet と切り替わるたび、
# 全 weight の pin バッファが破棄・再作成されるのを防ぐ。
# unpin 時にバッファをキャッシュし、再 pin 時に再利用する。
# -------------------------------------------------------------------------
try:
    import collections
    import comfy.pinned_memory as _pm_mod
    import comfy.memory_management as _mem_mod
    import comfy.model_management as _mm_mod
    from comfy.cli_args import args as _cli_args

    _pin_cache_logger = logging.getLogger("HSWQ_PinCache")

    _PIN_BUFFER_POOL = collections.defaultdict(list)
    _PIN_CACHE_TOTAL = 0
    _MAX_PIN_CACHE_BYTES = 16 * 1024 * 1024 * 1024
    _pin_cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "unpins": 0, "swaps": 0}
    # This cache is enabled only for the HSWQ Batched Detailer (SEGS) node.
    # Outside of that node, we delegate to ComfyUI's original pin/unpin to avoid side effects.
    _pm_mod._hswq_pin_cache_active = False

    _orig_pm_pin = _pm_mod.pin_memory
    _orig_pm_unpin = _pm_mod.unpin_memory

    def _cached_pin_memory(module):
        global _PIN_CACHE_TOTAL
        if module.pin_failed or _cli_args.disable_pinned_memory or _pm_mod.get_pin(module) is not None:
            return

        if not getattr(_pm_mod, "_hswq_pin_cache_active", False):
            return _orig_pm_pin(module)

        size = _mem_mod.vram_aligned_size([module.weight, module.bias])

        pool = _PIN_BUFFER_POOL.get(size)
        if pool:
            pin = pool.pop()
            module._pin = pin
            _PIN_CACHE_TOTAL -= size
            _pin_cache_stats["hits"] += 1
            h = _pin_cache_stats["hits"]
            if h <= 3 or h % 200 == 0:
                _pin_cache_logger.warning(
                    "[HSWQ PinCache] HIT size=%d pool_total=%.1f MB hits=%d misses=%d",
                    size, _PIN_CACHE_TOTAL / (1024 * 1024), h, _pin_cache_stats["misses"],
                )
            return True

        _pin_cache_stats["misses"] += 1
        pin = torch.empty((size,), dtype=torch.uint8)
        if _mm_mod.pin_memory(pin):
            module._pin = pin
        else:
            module.pin_failed = True
            return False
        return True

    def _cached_unpin_memory(module):
        global _PIN_CACHE_TOTAL
        if _pm_mod.get_pin(module) is None:
            return 0

        if not getattr(_pm_mod, "_hswq_pin_cache_active", False):
            return _orig_pm_unpin(module)

        pin = module._pin
        size = pin.numel() * pin.element_size()
        del module._pin

        _pin_cache_stats["unpins"] += 1
        u = _pin_cache_stats["unpins"]

        if _PIN_CACHE_TOTAL + size <= _MAX_PIN_CACHE_BYTES:
            _PIN_BUFFER_POOL[size].append(pin)
            _PIN_CACHE_TOTAL += size
            if u <= 3 or u % 200 == 0:
                _pin_cache_logger.warning(
                    "[HSWQ PinCache] STORE size=%d pool_total=%.1f MB pool_keys=%d unpins=%d",
                    size, _PIN_CACHE_TOTAL / (1024 * 1024),
                    len(_PIN_BUFFER_POOL), u,
                )
        else:
            # Adaptive eviction: evict buffers of DIFFERENT sizes to make room.
            # When models switch (e.g. ZImage→Lumina2), this replaces the old
            # model's cached buffers with the new model's, allowing both to coexist.
            freed = 0
            for other_size in list(_PIN_BUFFER_POOL.keys()):
                if other_size == size:
                    continue
                other_pool = _PIN_BUFFER_POOL[other_size]
                while other_pool and _PIN_CACHE_TOTAL + size > _MAX_PIN_CACHE_BYTES:
                    old_pin = other_pool.pop()
                    _mm_mod.unpin_memory(old_pin)
                    _PIN_CACHE_TOTAL -= other_size
                    freed += other_size
                if _PIN_CACHE_TOTAL + size <= _MAX_PIN_CACHE_BYTES:
                    break

            if _PIN_CACHE_TOTAL + size <= _MAX_PIN_CACHE_BYTES:
                _PIN_BUFFER_POOL[size].append(pin)
                _PIN_CACHE_TOTAL += size
                _pin_cache_stats["swaps"] += 1
                s = _pin_cache_stats["swaps"]
                if s <= 3 or s % 100 == 0:
                    _pin_cache_logger.warning(
                        "[HSWQ PinCache] SWAP size=%d freed=%.1f MB pool_total=%.1f MB swaps=%d",
                        size, freed / (1024 * 1024),
                        _PIN_CACHE_TOTAL / (1024 * 1024), s,
                    )
            else:
                _mm_mod.unpin_memory(pin)
                _pin_cache_stats["evictions"] += 1
                e = _pin_cache_stats["evictions"]
                if e <= 3 or e % 100 == 0:
                    _pin_cache_logger.warning(
                        "[HSWQ PinCache] EVICT size=%d (%.1f MB / %.1f MB) evictions=%d",
                        size, _PIN_CACHE_TOTAL / (1024 * 1024),
                        _MAX_PIN_CACHE_BYTES / (1024 * 1024), e,
                    )

        return size

    _pm_mod.pin_memory = _cached_pin_memory
    _pm_mod.unpin_memory = _cached_unpin_memory

    _pin_cache_logger.warning(
        "[HSWQ PinCache] pin buffer cache installed (max %.1f GB)",
        _MAX_PIN_CACHE_BYTES / (1024 ** 3),
    )

except Exception as e:
    logger.debug("HSWQ PinCache: not installed: %s", e)

# -------------------------------------------------------------------------
# HSWQ PinDebug: comfy.model_management.pin_memory 呼び出しの実態をログに出す
# (PinCache がキャッシュヒットした場合は model_management.pin_memory を
#  呼ばないため、このログにはキャッシュミス時のみ出力される)
# -------------------------------------------------------------------------
try:
    import inspect
    import comfy.model_management as _mm_pindebug

    _orig_mm_pin = getattr(_mm_pindebug, "pin_memory", None)

    if _orig_mm_pin is not None and not getattr(_orig_mm_pin, "_hswq_pindebug_wrapped", False):
        _pin_dbg_logger = logging.getLogger("HSWQ_PinDebug")
        _pin_dbg_call_count = [0]

        def _hswq_debug_pin_memory(tensor):
            _pin_dbg_call_count[0] += 1
            c = _pin_dbg_call_count[0]
            try:
                if c <= 5 or c % 500 == 0:
                    _pin_dbg_logger.warning(
                        "[HSWQ PinDebug] pin_memory called (#%d): device=%s, dtype=%s, shape=%s, nbytes=%d",
                        c,
                        getattr(tensor, "device", None),
                        getattr(tensor, "dtype", None),
                        tuple(getattr(tensor, "shape", [])),
                        getattr(tensor, "nbytes", -1),
                    )
                    stack = inspect.stack()[1]
                    _pin_dbg_logger.warning(
                        "[HSWQ PinDebug] caller: %s:%d (%s)",
                        stack.filename,
                        stack.lineno,
                        stack.function,
                    )
            except Exception:
                pass
            return _orig_mm_pin(tensor)

        setattr(_hswq_debug_pin_memory, "_hswq_pindebug_wrapped", True)
        _mm_pindebug.pin_memory = _hswq_debug_pin_memory
        _pin_dbg_logger.warning("[HSWQ PinDebug] model_management.pin_memory debug wrapper installed")
except Exception as e:
    logger.debug("HSWQ PinDebug: not installed: %s", e)

# Native comfy_quant INT8 (int8_tensorwise): Conv2d quant load + comfy_quant decode
# (ComfyUI core MixedPrecisionOps only covers Linear; SD UNet INT8 needs Conv2d.)
try:
    from .patches.comfy_quant_int8 import apply_comfy_quant_int8_patches
    apply_comfy_quant_int8_patches()
except Exception as e:
    logger.debug("HSWQ INT8 comfy_quant patches not applied: %s", e)

# HSWQ&Nunchaku Ultimate SD Upscale: apply copy_ / FP8 bias / embedder / Lumina compat patches in this extension
try:
    from .usdu_compat_patches import apply_usdu_compat_patches
    apply_usdu_compat_patches()
except Exception as e:
    logger.debug("USDU compat patches not applied: %s", e)

# torch.compile + FP8 + Lumina/Flux + comfy_kitchen 互換パッチ
# NunchakuFluxLoraStacker のパッチモジュールから安全なパッチのみ選択適用:
#   - LoraDiff.calculate_weight: reshape != weight.shape の LoRA をスキップ (根本対策)
#   - lumina.modulate / apply_gate: hidden_dim 不一致時の安全フォールバック
#
# 注意: グローバルな F.linear / matmul / mul / add パッチは適用しない。
# これらは RuntimeError をキャッチしてテンソルをスライスするが、ComfyUI 本体が
# 同じ RuntimeError を使って不正な LoRA を検出・スキップしているため、
# グローバルパッチが ComfyUI のエラーハンドリングを横取りし、壊れた次元の
# テンソルが後続の RMSNorm 等でクラッシュを引き起こす。
try:
    import importlib
    _lora_stacker_patches = None
    _lora_stacker_root = Path(__file__).parent.parent / "ComfyUI-NunchakuFluxLoraStacker" / "patches"
    _lora_stacker_patch_file = _lora_stacker_root / "zimage_fp8_torchcompile.py"
    if _lora_stacker_patch_file.exists():
        spec = importlib.util.spec_from_file_location(
            "nunchaku_lora_stacker_patches.zimage_fp8_torchcompile",
            str(_lora_stacker_patch_file),
        )
        if spec and spec.loader:
            _lora_stacker_patches = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_lora_stacker_patches)

    _applied = []
    if _lora_stacker_patches:
        # LoRA reshape != weight.shape をスキップ (根本対策: 不正 LoRA の適用を防ぐ)
        _fn = getattr(_lora_stacker_patches, "_apply_lumina_modulate_patch", None)
        if _fn and _fn():
            _applied.append("lumina.modulate")
        _fn = getattr(_lora_stacker_patches, "_apply_lumina_apply_gate_patch", None)
        if _fn and _fn():
            _applied.append("lumina.apply_gate")

        # LoraDiff.calculate_weight パッチは apply_patches() 末尾にあるため直接適用
        try:
            import comfy.weight_adapter.lora as _lora_mod
            _LoraDiff = getattr(_lora_mod, "LoraDiff", None)
            if _LoraDiff is not None:
                _orig_cw = getattr(_LoraDiff, "calculate_weight", None)
                if _orig_cw and not getattr(_orig_cw, "_hswq_lora_skip_patched", False):
                    import torch as _torch_lora

                    def _lora_skip_calculate_weight(
                        self, weight, key, strength, strength_model, offset,
                        function, intermediate_dtype=None, original_weight=None,
                    ):
                        if intermediate_dtype is None:
                            intermediate_dtype = _torch_lora.float32
                        v = self.weights
                        reshape = v[5]
                        if reshape is not None and tuple(reshape) != weight.shape:
                            logger.warning(
                                "LoRA %s: skip %s (reshape %s != weight %s) [HSWQ compat]",
                                self.name, key, list(reshape), list(weight.shape),
                            )
                            return weight
                        try:
                            lora_diff = _torch_lora.mm(
                                v[0].flatten(start_dim=1), v[1].flatten(start_dim=1)
                            )
                            if lora_diff.numel() != weight.numel():
                                logger.warning(
                                    "LoRA %s: skip %s (numel %d != %d) [HSWQ compat]",
                                    self.name, key, lora_diff.numel(), weight.numel(),
                                )
                                return weight
                        except Exception:
                            return weight
                        return _orig_cw(
                            self, weight=weight, key=key, strength=strength,
                            strength_model=strength_model, offset=offset,
                            function=function, intermediate_dtype=intermediate_dtype,
                            original_weight=original_weight,
                        )

                    _lora_skip_calculate_weight._hswq_lora_skip_patched = True
                    _LoraDiff.calculate_weight = _lora_skip_calculate_weight
                    _applied.append("LoraDiff.calculate_weight")
        except Exception as e:
            logger.debug("LoraDiff patch failed: %s", e)

    if _applied:
        logger.info("z_image/FP8/torch.compile compat patches applied: %s", ", ".join(_applied))
    else:
        logger.debug("No torch.compile compat patches applied")
except Exception as e:
    logger.debug("Torch compile compat patches not applied: %s", e)

# Check if _patch_model method exists in NunchakuZImageTransformer2DModel
try:
    from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
    if hasattr(NunchakuZImageTransformer2DModel, '_patch_model'):
        logger.info("NunchakuZImageTransformer2DModel._patch_model method found - patch may not be required")
    else:
        logger.warning("NunchakuZImageTransformer2DModel._patch_model method NOT found - patch is required")

    # ---------------------------------------------------------------------
    # ComfyUI ModelPatcher integration:
    # - ComfyUI passes ControlNet patches via transformer_options["patches"]["double_block"]
    # - Nunchaku Z-Image forward does not natively call those patches.
    # This monkey-patch wraps each transformer block and calls double_block patches
    # after each block, matching ComfyUI's NextDiT/QwenImage patching behavior.
    # ---------------------------------------------------------------------
    if not getattr(NunchakuZImageTransformer2DModel, "_comfyui_mp_patched", False):
        import inspect
        from comfy.ldm.flux.layers import EmbedND

        _orig_forward = NunchakuZImageTransformer2DModel.forward

        def _apply_double_block_patches(
            parent: "NunchakuZImageTransformer2DModel",
            block_index: int,
            unified_in,
            unified_out,
            adaln_input,
        ):
            """
            Apply ComfyUI ModelPatcher's double_block patches after a transformer block.
            IMPORTANT: do NOT change module hierarchy (LoRA matching depends on `layers.N.*` paths).
            """
            transformer_options = getattr(parent, "_comfyui_transformer_options", None)
            if not isinstance(transformer_options, dict):
                return unified_out

            patches = transformer_options.get("patches", {})
            if not isinstance(patches, dict):
                return unified_out

            double_block_patches = patches.get("double_block", [])
            if not double_block_patches:
                return unified_out

            # ComfyUI conventions
            transformer_options["block_index"] = block_index
            transformer_options.setdefault("block_type", "double")

            original_x_list = getattr(parent, "_comfyui_original_x_list", None)
            cap_len = getattr(parent, "_comfyui_cap_len", None)
            if not isinstance(cap_len, int) or cap_len < 0:
                cap_feats_list = getattr(parent, "_comfyui_cap_feats", None)
                if isinstance(cap_feats_list, list) and len(cap_feats_list) > 0 and hasattr(cap_feats_list[0], "shape"):
                    cap_len = int(cap_feats_list[0].shape[0])
                else:
                    cap_len = 0

            img_len = getattr(parent, "_comfyui_img_len", None)
            if not isinstance(img_len, int) or img_len < 0:
                img_len = 0

            unified = unified_out
            for p in double_block_patches:
                # Z-Image-Turbo uses List[torch.Tensor] for x, but ZImageControlPatch expects (B,C,H,W) tensor
                patch_x = original_x_list
                if isinstance(original_x_list, list) and len(original_x_list) > 0:
                    patch_x = torch.stack(original_x_list, dim=0)  # (B, C, F, H, W)
                    if patch_x.shape[2] == 1:
                        patch_x = patch_x.squeeze(2)  # (B, C, H, W)

                patch_in = {"x": patch_x, "block_index": block_index, "transformer_options": transformer_options}

                if unified is not None and hasattr(unified, "shape"):
                    # diffusers Z-Image order: [img_tokens, txt_tokens]
                    patch_in["img"] = unified[:, :img_len]
                    patch_in["txt"] = unified[:, img_len:img_len + cap_len]
                else:
                    patch_in["img"] = unified
                    patch_in["txt"] = None

                if unified_in is not None and hasattr(unified_in, "shape"):
                    patch_in["img_input"] = unified_in[:, :img_len]
                else:
                    patch_in["img_input"] = None

                # Build ComfyUI-compatible RoPE freqs for image tokens (pe) using EmbedND (Lumina style)
                pe_cached = getattr(parent, "_comfyui_pe_img", None)
                pe_key = getattr(parent, "_comfyui_pe_key", None)
                rope_options = transformer_options.get("rope_options", None) if isinstance(transformer_options, dict) else None
                h_scale = 1.0
                w_scale = 1.0
                h_start = 0.0
                w_start = 0.0
                if isinstance(rope_options, dict):
                    try:
                        h_scale = float(rope_options.get("scale_y", 1.0))
                        w_scale = float(rope_options.get("scale_x", 1.0))
                        h_start = float(rope_options.get("shift_y", 0.0))
                        w_start = float(rope_options.get("shift_x", 0.0))
                    except Exception:
                        h_scale = 1.0
                        w_scale = 1.0
                        h_start = 0.0
                        w_start = 0.0

                want_key = (img_len, cap_len, getattr(patch_x, "shape", None), h_scale, w_scale, h_start, w_start, unified.device, unified.dtype)
                if pe_cached is None or pe_key != want_key:
                    # Default Z-Image settings from Nunchaku docs: rope_theta=256, axes_dims=[32,48,48] (sum=128)
                    rope_theta = 256
                    axes_dims = (32, 48, 48)
                    head_dim = sum(axes_dims)
                    rope_embedder = getattr(parent, "_comfyui_rope_embedder", None)
                    if rope_embedder is None or getattr(rope_embedder, "theta", None) != rope_theta:
                        rope_embedder = EmbedND(dim=head_dim, theta=rope_theta, axes_dim=list(axes_dims))
                        parent._comfyui_rope_embedder = rope_embedder

                    b = int(patch_x.shape[0]) if hasattr(patch_x, "shape") else 1
                    ids = torch.zeros((b, img_len, 3), dtype=torch.float32, device=unified.device)

                    x_list = getattr(parent, "_comfyui_original_x_list", None)
                    h_tokens = 0
                    w_tokens = 0
                    try:
                        if isinstance(x_list, list) and len(x_list) > 0 and hasattr(x_list[0], "shape"):
                            _, f, h, w = x_list[0].shape
                            patch_size = int(getattr(parent, "_comfyui_patch_size", 2))
                            h_tokens = h // patch_size
                            w_tokens = w // patch_size
                    except Exception:
                        h_tokens = 0
                        w_tokens = 0

                    if h_tokens > 0 and w_tokens > 0:
                        n = min(img_len, h_tokens * w_tokens)
                        cap_offset = float(cap_len + 1)
                        ids[:, :n, 0] = cap_offset
                        ys = (torch.arange(h_tokens, dtype=torch.float32, device=unified.device) * h_scale + h_start).view(-1, 1).repeat(1, w_tokens).flatten()
                        xs = (torch.arange(w_tokens, dtype=torch.float32, device=unified.device) * w_scale + w_start).view(1, -1).repeat(h_tokens, 1).flatten()
                        ids[:, :n, 1] = ys[:n]
                        ids[:, :n, 2] = xs[:n]

                    pe_img = rope_embedder(ids).movedim(1, 2)  # (B, seq, 1, head_dim/2, 2, 2)
                    pe_img = pe_img.to(dtype=unified.dtype).contiguous()
                    parent._comfyui_pe_img = pe_img
                    parent._comfyui_pe_key = want_key
                    pe_cached = pe_img

                patch_in["pe"] = pe_cached
                patch_in["vec"] = adaln_input
                patch_in["block_type"] = transformer_options.get("block_type", "double")

                patch_out = p(patch_in)
                if isinstance(patch_out, dict) and unified is not None and hasattr(unified, "shape"):
                    if "img" in patch_out:
                        unified[:, :img_len] = patch_out["img"]
                    if "txt" in patch_out:
                        unified[:, img_len:img_len + cap_len] = patch_out["txt"]

            return unified

        def _ensure_layers_patched_in_place(parent: "NunchakuZImageTransformer2DModel"):
            if getattr(parent, "_comfyui_layers_patched_in_place", False):
                return
            try:
                layers = getattr(parent, "layers", None)
            except Exception:
                layers = None
            if layers is None:
                return
            try:
                for idx, layer in enumerate(layers):
                    if getattr(layer, "_comfyui_double_block_patched", False):
                        continue
                    orig_layer_forward = layer.forward

                    def _layer_forward_patched(*args, __orig=orig_layer_forward, __idx=idx, __parent=parent, **kwargs):
                        unified_in = args[0] if len(args) > 0 else None
                        adaln_input = args[3] if len(args) > 3 else None
                        out = __orig(*args, **kwargs)
                        return _apply_double_block_patches(__parent, __idx, unified_in, out, adaln_input)

                    layer.forward = _layer_forward_patched
                    layer._comfyui_double_block_patched = True
                parent._comfyui_layers_patched_in_place = True
                logger.info("[ZImageTurbo] Patched transformer block forwards (in-place) for double_block ControlNet patches")
            except Exception as e:
                logger.exception(f"[ZImageTurbo] Failed to patch layer forwards in-place: {e}")

        def _patched_forward(self, x, t, cap_feats=None, *args, control=None, transformer_options=None, **kwargs):
            # Store for block wrappers
            if isinstance(transformer_options, dict):
                self._comfyui_transformer_options = transformer_options
            else:
                self._comfyui_transformer_options = {}
            self._comfyui_original_x_list = x
            self._comfyui_cap_feats = cap_feats
            try:
                if isinstance(cap_feats, list) and len(cap_feats) > 0 and hasattr(cap_feats[0], "shape"):
                    self._comfyui_cap_len = int(cap_feats[0].shape[0])
            except Exception:
                self._comfyui_cap_len = 0

            # Z-Image (diffusers) unified token order is: img tokens first, then txt tokens.
            # Compute img token length from the input x list shape and patch sizes (default: 2 / 1).
            try:
                patch_size = int(kwargs.get("patch_size", 2))
            except Exception:
                patch_size = 2
            try:
                f_patch_size = int(kwargs.get("f_patch_size", 1))
            except Exception:
                f_patch_size = 1

            try:
                if isinstance(x, list) and len(x) > 0 and hasattr(x[0], "shape"):
                    # x element shape: (C, F, H, W)
                    _, f, h, w = x[0].shape
                    img_tokens = (h // patch_size) * (w // patch_size) * (f // f_patch_size)
                    # SEQ_MULTI_OF in diffusers ZImage is 32
                    img_tokens = int(((img_tokens + 31) // 32) * 32)
                    self._comfyui_img_len = img_tokens
                else:
                    self._comfyui_img_len = 0
            except Exception:
                self._comfyui_img_len = 0

            _ensure_layers_patched_in_place(self)

            # Log once per forward if patches exist (avoid spam)
            try:
                patches = self._comfyui_transformer_options.get("patches", {})
                dbl = patches.get("double_block", []) if isinstance(patches, dict) else []
                if dbl and not getattr(self, "_comfyui_logged_double_block", False):
                    logger.info(f"[ZImageTurbo] double_block patches detected: {len(dbl)} (will apply after each block)")
                    self._comfyui_logged_double_block = True
            except Exception:
                pass

            # Call original forward with only supported kwargs
            try:
                sig = inspect.signature(_orig_forward)
                allowed = set(sig.parameters.keys())
                call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            except Exception:
                call_kwargs = kwargs
            return _orig_forward(self, x, t, cap_feats=cap_feats, **call_kwargs)

        NunchakuZImageTransformer2DModel.forward = _patched_forward
        NunchakuZImageTransformer2DModel._comfyui_mp_patched = True
        logger.info("[ZImageTurbo] Installed ModelPatcher-compatible forward() patch")
except ImportError as e:
    logger.warning(f"Could not import NunchakuZImageTransformer2DModel to check _patch_model: {e}")

nunchaku_full_version = get_package_version("nunchaku").split("+")[0].strip()

logger.info(f"Nunchaku version: {nunchaku_full_version}")
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")


min_nunchaku_version = "1.0.0"
nunchaku_version = nunchaku_full_version.split("+")[0].strip()
nunchaku_major_minor_patch_version = ".".join(nunchaku_version.split(".")[:3])

try:
    if Version(nunchaku_major_minor_patch_version) < Version(min_nunchaku_version):
        logger.warning(
            f"ComfyUI-nunchaku {get_plugin_version()} requires nunchaku >= v{min_nunchaku_version}, "
            f"but found nunchaku {nunchaku_full_version}. Please update nunchaku."
        )
except InvalidVersion:
    logger.warning(
        f"Could not parse nunchaku version: {nunchaku_full_version}. "
        f"Please ensure you have at least v{min_nunchaku_version}."
    )

NODE_CLASS_MAPPINGS = {}

# Checkpoint Loader (SDXL): NunchakuSDXLIntegratedLoader + NunchakuSDXLDiTLoaderDualCLIP
# Nunchaku Ultimate SD Upscale
# (Other Nunchaku SDXL nodes removed)
try:
    from .nodes.models.sdxl import NunchakuSDXLDiTLoaderDualCLIP, NunchakuSDXLIntegratedLoader

    NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLDiTLoaderDualCLIP"] = NunchakuSDXLDiTLoaderDualCLIP
    NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLIntegratedLoader"] = NunchakuSDXLIntegratedLoader
except (ImportError, ModuleNotFoundError) as e:
    logger.exception(f"Node `NunchakuSDXLDiTLoaderDualCLIP` or `NunchakuSDXLIntegratedLoader` import failed: {e}")
    # Try alternative import method using absolute path
    try:
        import importlib.util
        from pathlib import Path

        # Get the directory where __init__.py is located
        current_dir = Path(__file__).parent.resolve()
        sdxl_path = current_dir / "nodes" / "models" / "sdxl.py"

        if not sdxl_path.exists():
            raise FileNotFoundError(f"sdxl.py not found at {sdxl_path}")

        spec = importlib.util.spec_from_file_location(
            "nodes.models.sdxl",
            str(sdxl_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {sdxl_path}")

        sdxl_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdxl_module)
        if hasattr(sdxl_module, "NunchakuSDXLDiTLoaderDualCLIP"):
            NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLDiTLoaderDualCLIP"] = sdxl_module.NunchakuSDXLDiTLoaderDualCLIP
        if hasattr(sdxl_module, "NunchakuSDXLIntegratedLoader"):
            NODE_CLASS_MAPPINGS["NunchakuUssoewwinSDXLIntegratedLoader"] = sdxl_module.NunchakuSDXLIntegratedLoader
        logger.info(f"Successfully loaded NunchakuSDXLDiTLoaderDualCLIP and NunchakuSDXLIntegratedLoader using alternative method from {sdxl_path}")
    except Exception as e2:
        logger.exception(f"Alternative import method also failed: {e2}")

try:
    from .nodes.nunchaku_usdu import (
        NunchakuUltimateSDUpscale,
    )
    NODE_CLASS_MAPPINGS["NunchakuUltimateSDUpscale"] = NunchakuUltimateSDUpscale
    logger.info("Nunchaku Ultimate SD Upscale nodes registered successfully")
except Exception as e:
    logger.error(f"Failed to register Nunchaku Ultimate SD Upscale nodes: {e}", exc_info=True)

try:
    from .nodes.hswq_save_image import NunchakuSaveImage

    NODE_CLASS_MAPPINGS["NunchakuSaveImage"] = NunchakuSaveImage
    logger.info("Nunchaku Save Image node registered successfully")
except Exception as e:
    logger.error(f"Failed to register Nunchaku Save Image node: {e}", exc_info=True)

# SDXL MultiGPU node registration (UNET + CLIP, ref CheckpointLoaderSimple)
try:
    import folder_paths
    import comfy.sd
    from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
    from .device_utils import get_device_list

    _UNETLoaderBase = GLOBAL_NODE_CLASS_MAPPINGS.get("UNETLoader")
    if _UNETLoaderBase is None:
        sdxl_logger.warning("[SDXL] UNETLoader not found in GLOBAL_NODE_CLASS_MAPPINGS")
    else:

        class NunchakuUssoewwinCheckpointLoaderSDXL(_UNETLoaderBase):
            """Checkpoint Loader (SDXL) with device selection. Ref: CheckpointLoaderSimple."""

            @classmethod
            def INPUT_TYPES(cls):
                base = _UNETLoaderBase.INPUT_TYPES()
                base_req = dict(base.get("required", {}))
                base_opt = dict(base.get("optional", {}))
                devices = get_device_list()
                default_dev = devices[1] if len(devices) > 1 else devices[0]
                req = {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "SDXL checkpoint to load MODEL and CLIP from (same as standard Load Checkpoint)."}),
                    "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                }
                opt = {"device": (devices, {"default": default_dev})}
                return {"required": req, "optional": opt}

            RETURN_TYPES = ("MODEL", "CLIP")
            OUTPUT_TOOLTIPS = ("The UNet diffusion model from checkpoint.", "The CLIP model from the SDXL checkpoint.")
            FUNCTION = "load_checkpoint"
            CATEGORY = "loaders"
            TITLE = "Checkpoint Loader (SDXL)"

            def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
                original_device = get_current_device()
                if device is not None:
                    set_current_device(device)
                try:
                    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                    model_options = {}
                    if weight_dtype == "fp8_e4m3fn":
                        model_options["dtype"] = torch.float8_e4m3fn
                    elif weight_dtype == "fp8_e4m3fn_fast":
                        model_options["dtype"] = torch.float8_e4m3fn
                        model_options["fp8_optimizations"] = True
                    elif weight_dtype == "fp8_e5m2":
                        model_options["dtype"] = torch.float8_e5m2
                    
                    out = comfy.sd.load_checkpoint_guess_config(
                        ckpt_path,
                        output_vae=False,
                        output_clip=True,
                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                        model_options=model_options,
                    )
                    model, clip, _v = out[:3]
                    return (model, clip)
                finally:
                    set_current_device(original_device)

        NODE_CLASS_MAPPINGS["NunchakuUssoewwinCheckpointLoaderSDXL"] = NunchakuUssoewwinCheckpointLoaderSDXL
        sdxl_logger.info("[SDXL] Registered NunchakuUssoewwinCheckpointLoaderSDXL node (MODEL + CLIP)")

        class UssoewwinCheckpointLoaderZImageTurbo(_UNETLoaderBase):
            """Checkpoint Loader (Z Image Turbo) with device selection. Ref: CheckpointLoaderSimple."""

            @classmethod
            def INPUT_TYPES(cls):
                base = _UNETLoaderBase.INPUT_TYPES()
                base_req = dict(base.get("required", {}))
                base_opt = dict(base.get("optional", {}))
                devices = get_device_list()
                default_dev = devices[1] if len(devices) > 1 else devices[0]
                req = {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Z Image Turbo checkpoint to load MODEL and CLIP from (same as standard Load Checkpoint)."}),
                    "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                }
                opt = {"device": (devices, {"default": default_dev})}
                return {"required": req, "optional": opt}

            RETURN_TYPES = ("MODEL",)
            OUTPUT_TOOLTIPS = ("The transformer diffusion model from checkpoint.",)
            FUNCTION = "load_checkpoint"
            CATEGORY = "loaders"
            TITLE = "Checkpoint Loader (Z Image Turbo)"

            def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
                original_device = get_current_device()
                if device is not None:
                    set_current_device(device)
                try:
                    import comfy.ops
                    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                    model_options = {}
                    is_fp8 = False
                    if weight_dtype == "fp8_e4m3fn":
                        model_options["dtype"] = torch.float8_e4m3fn
                        model_options["custom_operations"] = comfy.ops.fp8_ops
                        is_fp8 = True
                    elif weight_dtype == "fp8_e5m2":
                        model_options["dtype"] = torch.float8_e5m2
                        model_options["custom_operations"] = comfy.ops.fp8_ops
                        is_fp8 = True

                    out = comfy.sd.load_checkpoint_guess_config(
                        ckpt_path,
                        output_vae=False,
                        output_clip=False,
                        model_options=model_options,
                    )
                    model = out[0]
                    
                    # Debug: check parameter dtypes and layer types
                    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                        dm = model.model.diffusion_model
                        total_params = 0
                        fp8_params = 0
                        bf16_params = 0
                        for name, param in dm.named_parameters():
                            total_params += param.numel()
                            if param.dtype == torch.float8_e4m3fn or param.dtype == torch.float8_e5m2:
                                fp8_params += param.numel()
                            elif param.dtype == torch.bfloat16 or param.dtype == torch.float16:
                                bf16_params += param.numel()
                        sdxl_logger.info(f"[ZIT DEBUG] Total params: {total_params:,}, FP8: {fp8_params:,}, BF16/FP16: {bf16_params:,}")
                        # Check layer type
                        for name, module in dm.named_modules():
                            if hasattr(module, 'weight') and module.weight is not None:
                                is_fp8_forced = isinstance(module, comfy.ops.fp8_ops.Linear) if hasattr(comfy.ops, 'fp8_ops') else False
                                sdxl_logger.info(f"[ZIT DEBUG] Layer {name}: {type(module)}, is_fp8_ops_forced={is_fp8_forced}")
                                break
                    
                    # For FP8: force manual_cast_dtype to bfloat16 (ComfyUI default would cast FP8 to float32 otherwise)
                    if is_fp8 and hasattr(model, 'model') and hasattr(model.model, 'manual_cast_dtype'):
                        model.model.manual_cast_dtype = torch.bfloat16
                        sdxl_logger.info(f"[ZIT FP8] Forced manual_cast_dtype to bfloat16 for FP8 VRAM optimization")
                    
                    return (model,)
                finally:
                    set_current_device(original_device)

        NODE_CLASS_MAPPINGS["ZITCheckpointLoader"] = UssoewwinCheckpointLoaderZImageTurbo
        sdxl_logger.info(f"[Z Image Turbo] Registered ZITCheckpointLoader node, RETURN_TYPES={UssoewwinCheckpointLoaderZImageTurbo.RETURN_TYPES}")
except Exception as e:
    sdxl_logger.exception(f"[SDXL] Failed to register NunchakuUssoewwinCheckpointLoaderSDXL: {e}")

# Z Image FP8 E4M3 専用 UNet Loader（DiT Loader は init から除外）
try:
    from .hswq.zimage_fp8_e4m3_unet import HSWQFP8E4M3UNetLoader
    NODE_CLASS_MAPPINGS["HSWQFP8E4M3UNetLoader"] = HSWQFP8E4M3UNetLoader
    logger.info("Registered HSWQ FP8 E4M3 UNet Loader (ComfyUI standard UNet loader wrapper)")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ FP8 E4M3 UNet Loader not registered: %s", e)

# HSWQ Loader registration
try:
    from .nodes.hswq_loader_node import HSWQLoader
    NODE_CLASS_MAPPINGS["HSWQLoader"] = HSWQLoader
    sdxl_logger.info("[SDXL] Registered HSWQLoader node (FP8/INT8 Loader)")
except (ImportError, ModuleNotFoundError) as e:
    sdxl_logger.exception(f"[SDXL] Failed to register HSWQLoader: {e}")

# HSWQ Batched Detailer (SEGS) - phase-split version to minimize model switching
try:
    from .nodes.hswq_batched_detailer import HSWQBatchedDetailer
    NODE_CLASS_MAPPINGS["HSWQBatchedDetailer"] = HSWQBatchedDetailer
    logger.info("Registered HSWQ Batched Detailer (SEGS) (phase-split model-switch optimization)")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ Batched Detailer not registered: %s", e)

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
WEB_DIRECTORY = "js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku Initialization ")))
```

### 3.3 `nodes/hswq_loader_node.py`（全文・441 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import folder_paths
import safetensors.torch
import comfy.sd
import comfy.utils
import comfy.ops

# --- Efficient FP8 Dequantization ---

def dequantize_fp8_weight(weight_fp8: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Dequantize FP8 weight to target dtype efficiently.
    Uses bfloat16 as intermediate to minimize VRAM (2 bytes vs 4 bytes for float32).
    FP8 (1 byte) -> bfloat16 (2 bytes) -> target_dtype
    """
    # Use bfloat16 as intermediate for VRAM efficiency (half the memory of float32)
    w = weight_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)
    if target_dtype != torch.bfloat16:
        w = w.to(target_dtype)
    return w

# --- Custom Layers for Runtime Scaling ---
# These classes inherit from nn.Linear/nn.Conv2d for LoRA compatibility
# They expose a 'weight' property that returns dequantized weights for LoRA patching

class HSWQLinear(nn.Linear):
    """
    FP8 Linear layer with runtime dequantization.
    - Inherits from nn.Linear for LoRA compatibility
    - Stores weights in FP8 (1 byte per element) for 50% VRAM savings vs FP16
    - Dequantizes to compute dtype only during forward pass
    - Exposes 'weight' property for LoRA compatibility
    """
    # ComfyUI LowVram compatibility attributes
    comfy_cast_weights = False
    weight_function = []
    bias_function = []
    
    def __init__(self, original_linear: nn.Linear, weight_tensor: torch.Tensor, scale: torch.Tensor):
        # Initialize parent with same dimensions but no actual weight creation
        # We use device='meta' to create a shell without allocating memory
        nn.Module.__init__(self)  # Skip nn.Linear.__init__ to avoid weight allocation
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Store FP8 weight (1 byte per element) - this is the VRAM saving
        self.register_buffer("weight_fp8", weight_tensor.detach().clone(), persistent=False)
        self.register_buffer("hswq_scale", scale.clone().detach(), persistent=False)
        
        # Store original dtype for weight property
        self._compute_dtype = torch.float16
        
        # ComfyUI LowVram compatibility - instance-level lists
        self.weight_function = []
        self.bias_function = []
        
        if original_linear.bias is not None:
            # Keep bias as a parameter for compatibility
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    @property
    def weight(self):
        """Return dequantized weight for LoRA compatibility."""
        return dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, self._compute_dtype)
    
    @weight.setter
    def weight(self, value):
        """Allow weight setting for LoRA compatibility."""
        if value is not None:
            # Store patched weight (handle both Tensor and Parameter)
            val = value.data if isinstance(value, nn.Parameter) else value
            if not hasattr(self, '_patched_weight') or self._patched_weight is None:
                self.register_buffer("_patched_weight", None, persistent=False)
            self._patched_weight = val.detach()
    
    def __setattr__(self, name, value):
        """Override to handle weight assignment without register_parameter error."""
        if name == 'weight' and isinstance(value, (torch.Tensor, nn.Parameter)):
            # Call the property setter directly to avoid recursion
            HSWQLinear.weight.fset(self, value)
            return
        super().__setattr__(name, value)
    
    def set_weight(self, weight, inplace_update=False, seed=None):
        """ComfyUI ModelPatcher calls this to apply LoRA patches."""
        if not hasattr(self, '_patched_weight') or self._patched_weight is None:
            self.register_buffer("_patched_weight", None, persistent=False)
        val = weight.data if isinstance(weight, nn.Parameter) else weight
        self._patched_weight = val.detach()
        if not hasattr(HSWQLinear, '_lora_log_count'):
            HSWQLinear._lora_log_count = 0
        HSWQLinear._lora_log_count += 1
        if HSWQLinear._lora_log_count <= 3:
            print(f"[HSWQ-LoRA] Linear layer patched: {self.out_features}x{self.in_features}")
        elif HSWQLinear._lora_log_count == 4:
            print(f"[HSWQ-LoRA] ... (more layers being patched)")

    def forward(self, input):
        target_dtype = input.dtype
        self._compute_dtype = target_dtype  # Update for weight property
        
        # If LoRA patched weight exists, use it instead of FP8 dequantization
        if hasattr(self, '_patched_weight') and self._patched_weight is not None:
            w = self._patched_weight.to(target_dtype)
        else:
            # Dequantize FP8 -> bfloat16 -> target_dtype (VRAM efficient path)
            w = dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, target_dtype)
        
        # Apply LowVram weight_function patches (ComfyUI dynamic LoRA)
        for f in self.weight_function:
            w = f(w)
        
        bias = self.bias.to(target_dtype) if self.bias is not None else None
        # Apply LowVram bias_function patches
        if bias is not None:
            for f in self.bias_function:
                bias = f(bias)
        
        return F.linear(input, w, bias)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Override to include dequantized weight in state_dict for LoRA compatibility."""
        # Add dequantized weight (this is what LoRA key mapping needs)
        destination[prefix + 'weight'] = self.weight if keep_vars else self.weight.detach()
        # Add bias if present
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Override to expose weight as a parameter for ModelPatcher compatibility."""
        # Yield the dequantized weight as if it were a parameter
        yield prefix + ('.' if prefix else '') + 'weight', self.weight
        if self.bias is not None:
            yield prefix + ('.' if prefix else '') + 'bias', self.bias

class HSWQConv2d(nn.Conv2d):
    """
    FP8 Conv2d layer with runtime dequantization.
    - Inherits from nn.Conv2d for LoRA compatibility
    - Stores weights in FP8 (1 byte per element) for 50% VRAM savings vs FP16
    - Dequantizes to compute dtype only during forward pass
    - Exposes 'weight' property for LoRA compatibility
    """
    # ComfyUI LowVram compatibility attributes
    comfy_cast_weights = False
    weight_function = []
    bias_function = []
    
    def __init__(self, original_conv: nn.Conv2d, weight_tensor: torch.Tensor, scale: torch.Tensor):
        # Initialize parent without weight allocation
        nn.Module.__init__(self)  # Skip nn.Conv2d.__init__ to avoid weight allocation
        
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.padding_mode = original_conv.padding_mode
        # Required for nn.Conv2d __repr__ compatibility
        self.transposed = getattr(original_conv, 'transposed', False)
        self.output_padding = getattr(original_conv, 'output_padding', (0, 0))
        
        # Store FP8 weight (1 byte per element) - this is the VRAM saving
        self.register_buffer("weight_fp8", weight_tensor.detach().clone(), persistent=False)
        self.register_buffer("hswq_scale", scale.clone().detach(), persistent=False)
        
        # Store original dtype for weight property
        self._compute_dtype = torch.float16
        
        # ComfyUI LowVram compatibility - instance-level lists
        self.weight_function = []
        self.bias_function = []
        
        if original_conv.bias is not None:
            # Keep bias as a parameter for compatibility
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    @property
    def weight(self):
        """Return dequantized weight for LoRA compatibility."""
        return dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, self._compute_dtype)
    
    @weight.setter
    def weight(self, value):
        """Allow weight setting for LoRA compatibility."""
        if value is not None:
            # Store patched weight (handle both Tensor and Parameter)
            val = value.data if isinstance(value, nn.Parameter) else value
            if not hasattr(self, '_patched_weight') or self._patched_weight is None:
                self.register_buffer("_patched_weight", None, persistent=False)
            self._patched_weight = val.detach()
    
    def __setattr__(self, name, value):
        """Override to handle weight assignment without register_parameter error."""
        if name == 'weight' and isinstance(value, (torch.Tensor, nn.Parameter)):
            # Call the property setter directly to avoid recursion
            HSWQConv2d.weight.fset(self, value)
            return
        super().__setattr__(name, value)
    
    def set_weight(self, weight, inplace_update=False, seed=None):
        """ComfyUI ModelPatcher calls this to apply LoRA patches."""
        if not hasattr(self, '_patched_weight') or self._patched_weight is None:
            self.register_buffer("_patched_weight", None, persistent=False)
        val = weight.data if isinstance(weight, nn.Parameter) else weight
        self._patched_weight = val.detach()
        if not hasattr(HSWQConv2d, '_lora_log_count'):
            HSWQConv2d._lora_log_count = 0
        HSWQConv2d._lora_log_count += 1
        if HSWQConv2d._lora_log_count <= 2:
            print(f"[HSWQ-LoRA] Conv2d layer patched: {self.out_channels}x{self.in_channels}")

    def forward(self, input):
        target_dtype = input.dtype
        self._compute_dtype = target_dtype  # Update for weight property
        
        # If LoRA patched weight exists, use it instead of FP8 dequantization
        if hasattr(self, '_patched_weight') and self._patched_weight is not None:
            w = self._patched_weight.to(target_dtype)
        else:
            # Dequantize FP8 -> bfloat16 -> target_dtype (VRAM efficient path)
            w = dequantize_fp8_weight(self.weight_fp8, self.hswq_scale, target_dtype)
        
        # Apply LowVram weight_function patches (ComfyUI dynamic LoRA)
        for f in self.weight_function:
            w = f(w)
        
        bias = self.bias.to(target_dtype) if self.bias is not None else None
        # Apply LowVram bias_function patches
        if bias is not None:
            for f in self.bias_function:
                bias = f(bias)
        
        return F.conv2d(input, w, bias, self.stride, self.padding, self.dilation, self.groups)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Override to include dequantized weight in state_dict for LoRA compatibility."""
        # Add dequantized weight (this is what LoRA key mapping needs)
        destination[prefix + 'weight'] = self.weight if keep_vars else self.weight.detach()
        # Add bias if present
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Override to expose weight as a parameter for ModelPatcher compatibility."""
        # Yield the dequantized weight as if it were a parameter
        yield prefix + ('.' if prefix else '') + 'weight', self.weight
        if self.bias is not None:
            yield prefix + ('.' if prefix else '') + 'bias', self.bias

# --- Loader Node ---

class HSWQLoader:
    """
    HSWQ V2 (Scaled FP8) Full VRAM-Optimized Loader
    
    Loads models with .scale keys and replaces layers with custom HSWQ layers
    that store weights in FP8 but compute in FP16 (runtime descaling).
    This achieves:
    - 50% Storage Reduction
    - 50% VRAM Reduction (FP8 storage)
    - High Quality (Scaled reconstruction)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_hswq_checkpoint_vram_opt"
    CATEGORY = "HSWQ"
    TITLE = "HSWQ FP8/INT8 Loader (VRAM Opt)"

    def load_hswq_checkpoint_vram_opt(self, ckpt_name):
        from ..patches.comfy_quant_int8 import (
            apply_comfy_quant_int8_patches,
            checkpoint_looks_like_comfy_quant_int8,
            reset_int8_lora_log_counters,
            summarize_int8_lora_capability,
        )

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        print(f"[HSWQ] Loading checkpoint: {ckpt_name}")

        # Native comfy_quant INT8 (int8_tensorwise): use ComfyUI MixedPrecisionOps path.
        # Requires Conv2d quant patch from this extension (core only loads Linear).
        if checkpoint_looks_like_comfy_quant_int8(ckpt_path):
            apply_comfy_quant_int8_patches()
            reset_int8_lora_log_counters()
            print("[HSWQ] Detected comfy_quant INT8 (int8_tensorwise); using native MixedPrecisionOps path")
            model, clip, vae, _clipvision = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            summarize_int8_lora_capability(model)
            return (model, clip, vae)

        # 1. Load state_dict to CPU first (legacy Scaled FP8 .scale path)
        state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")

        scale_map = {}  # key: weight_key, value: scale_tensor
        weight_map = {}  # key: weight_key, value: fp8_weight_tensor (stored BEFORE dequantization)

        # Extract scales and weights, then DEQUANTIZE for ComfyUI's load_state_dict
        # This is crucial: PyTorch's load_state_dict cannot load FP8 tensors into FP16 nn.Parameters
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if key.endswith(".scale"):
                if key.endswith(".weight.scale"):
                    weight_key = key.replace(".weight.scale", ".weight")
                else:
                    weight_key = key.replace(".scale", ".weight")

                if weight_key in state_dict:
                    fp8_weight = state_dict[weight_key]
                    scale = state_dict[key]

                    # Store original FP8 weight for HSWQ layer replacement later
                    scale_map[weight_key] = scale.clone()
                    weight_map[weight_key] = fp8_weight.clone()

                    # DEQUANTIZE to FP16 for ComfyUI's load_state_dict compatibility
                    # This ensures the model loads correctly and LoRA can be applied
                    dequantized = dequantize_fp8_weight(fp8_weight, scale, torch.float16)
                    state_dict[weight_key] = dequantized

                    # Mark scale key for removal (ComfyUI doesn't need it)
                    keys_to_remove.append(key)

        # Remove scale keys from state_dict
        for key in keys_to_remove:
            del state_dict[key]

        print(f"[HSWQ] Dequantized {len(scale_map)} FP8 weights to FP16 for model loading")

        # 2. Inject into ComfyUI loading mechanism
        # Monkey patch load_torch_file to avoid re-reading the file from disk
        original_loader = comfy.utils.load_torch_file

        def patched_loader(path, safe_load=False, device=None, return_metadata=False):
            if os.path.normpath(path) == os.path.normpath(ckpt_path):
                if return_metadata:
                    # safetensors header info might be lost here but usually empty dict is enough for inference
                    # unless strict metadata checking is performed.
                    return state_dict, {}
                return state_dict
            return original_loader(path, safe_load, device, return_metadata=return_metadata)

        comfy.utils.load_torch_file = patched_loader

        try:
            # 3. Standard Load (uses patched loader with DEQUANTIZED weights)
            # Now ComfyUI can properly load all weights and LoRA will work correctly
            model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
        finally:
            # Restore original loader
            comfy.utils.load_torch_file = original_loader

        # 4. Dynamic Layer Replacement
        print("[HSWQ] Replacing layers with FP8 Runtime-Descaling variants...")
        hswq_replaced_count = 0
        fp8_param_bytes = 0
        unet = model.model.diffusion_model

        # Collect modules to replace (avoid modifying during iteration)
        modules_to_replace = []
        for name, module in unet.named_modules():
            key_name = f"model.diffusion_model.{name}.weight"
            if key_name in scale_map:
                modules_to_replace.append((name, module, scale_map[key_name], weight_map[key_name]))

        for name, module, scale, weight_val in modules_to_replace:
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = unet.get_submodule(parent_name)
            else:
                parent_name = ""
                child_name = name
                parent = unet

            if isinstance(module, nn.Linear):
                new_layer = HSWQLinear(module, weight_val, scale)
                setattr(parent, child_name, new_layer)
                hswq_replaced_count += 1
                fp8_param_bytes += weight_val.numel()  # FP8 = 1 byte per element
            elif isinstance(module, nn.Conv2d):
                new_layer = HSWQConv2d(module, weight_val, scale)
                setattr(parent, child_name, new_layer)
                hswq_replaced_count += 1
                fp8_param_bytes += weight_val.numel()

        fp8_mb = fp8_param_bytes / (1024 * 1024)
        fp16_equivalent_mb = (fp8_param_bytes * 2) / (1024 * 1024)
        print(f"[HSWQ] Replaced {hswq_replaced_count} layers with FP8 storage.")
        print(f"[HSWQ] VRAM for FP8 weights: {fp8_mb:.1f} MB (vs {fp16_equivalent_mb:.1f} MB if FP16)")

        # Reset LoRA log counters for fresh tracking
        HSWQLinear._lora_log_count = 0
        HSWQConv2d._lora_log_count = 0

        # Verify state_dict contains weights for LoRA key mapping
        test_sd = model.model.state_dict()
        weight_keys = [k for k in test_sd.keys() if k.endswith(".weight") and "diffusion_model" in k]
        print(f"[HSWQ] Model state_dict has {len(weight_keys)} diffusion_model weight keys for LoRA mapping")

        # Test that set_weight is discoverable
        test_layer = unet.get_submodule("input_blocks.1.0.out_layers.3")
        has_set_weight = hasattr(test_layer, "set_weight") and callable(getattr(test_layer, "set_weight", None))
        print(f"[HSWQ] HSWQ layers have set_weight method: {has_set_weight}")

        # Clear CPU memory
        del state_dict
        del weight_map
        del scale_map
        del test_sd
        import gc

        gc.collect()

        return (model, clip, vae)
```

### 3.4 `hswq/zimage_fp8_e4m3_unet.py`（全文・2545 行）

```python
from __future__ import annotations
import torch


import os
import sys
import json
import glob
import hashlib
import inspect

import traceback
import math
import time
import random
import logging

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

_HSWQ_DIR = os.path.dirname(os.path.realpath(__file__))
_ROOT_DIR = os.path.dirname(_HSWQ_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator
from comfy_api.internal import register_versions, ComfyAPIWithVersion
from comfy_api.version_list import supported_versions
from comfy_api.latest import io, ComfyExtension

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers

if args.enable_manager:
    import comfyui_manager

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=16384

class CLIPTextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."
    SEARCH_ALIASES = ["text", "prompt", "text prompt", "positive prompt", "negative prompt", "encode text", "text encoder", "encode prompt"]

    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens), )


class ConditioningCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_1": ("CONDITIONING", ), "conditioning_2": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"

    CATEGORY = "conditioning"
    SEARCH_ALIASES = ["combine", "merge conditioning", "combine prompts", "merge prompts", "mix prompts", "add prompt"]

    def combine(self, conditioning_1, conditioning_2):
        return (conditioning_1 + conditioning_2, )

class ConditioningAverage :
    SEARCH_ALIASES = ["blend prompts", "interpolate conditioning", "mix prompts", "style fusion", "weighted blend"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ), "conditioning_from": ("CONDITIONING", ),
                              "conditioning_to_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addWeighted"

    CATEGORY = "conditioning"

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
        out = []

        if len(conditioning_from) > 1:
            logging.warning("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )

class ConditioningConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_to": ("CONDITIONING",),
            "conditioning_from": ("CONDITIONING",),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"

    CATEGORY = "conditioning"

    def concat(self, conditioning_to, conditioning_from):
        out = []

        if len(conditioning_from) > 1:
            logging.warning("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from),1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out, )

class ConditioningSetArea:
    SEARCH_ALIASES = ["regional prompt", "area prompt", "spatial conditioning", "localized prompt"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"area": (height // 8, width // 8, y // 8, x // 8),
                                                                "strength": strength,
                                                                "set_area_to_bounds": False})
        return (c, )

class ConditioningSetAreaPercentage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                              "height": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                              "x": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                              "y": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"area": ("percentage", height, width, y, x),
                                                                "strength": strength,
                                                                "set_area_to_bounds": False})
        return (c, )

class ConditioningSetAreaStrength:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"strength": strength})
        return (c, )


class ConditioningSetMask:
    SEARCH_ALIASES = ["masked prompt", "regional inpaint conditioning", "mask conditioning"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "mask": ("MASK", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "set_cond_area": (["default", "mask bounds"],),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, mask, set_cond_area, strength):
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        c = node_helpers.conditioning_set_values(conditioning, {"mask": mask,
                                                                "set_area_to_bounds": set_area_to_bounds,
                                                                "mask_strength": strength})
        return (c, )

class ConditioningZeroOut:
    SEARCH_ALIASES = ["null conditioning", "clear conditioning"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"

    CATEGORY = "advanced/conditioning"

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c, )

class ConditioningSetTimestepRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_range"

    CATEGORY = "advanced/conditioning"

    def set_range(self, conditioning, start, end):
        c = node_helpers.conditioning_set_values(conditioning, {"start_percent": start,
                                                                "end_percent": end})
        return (c, )

class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"

    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent images back into pixel space images."
    SEARCH_ALIASES = ["decode", "decode latent", "latent to image", "render latent"]

    def decode(self, vae, samples):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        images = vae.decode(latent)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )

class VAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ), "vae": ("VAE", ),
                             "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "advanced": True}),
                             "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "advanced": True}),
                             "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time.", "advanced": True}),
                             "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap.", "advanced": True}),
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "_for_testing"

    def decode(self, vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(samples["samples"], tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )

class VAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent"
    SEARCH_ALIASES = ["encode", "encode image", "image to latent"]

    def encode(self, vae, pixels):
        t = vae.encode(pixels)
        return ({"samples":t}, )

class VAEEncodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE", ), "vae": ("VAE", ),
                             "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "advanced": True}),
                             "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "advanced": True}),
                             "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to encode at a time.", "advanced": True}),
                             "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap.", "advanced": True}),
                            }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "_for_testing"

    def encode(self, vae, pixels, tile_size, overlap, temporal_size=64, temporal_overlap=8):
        t = vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
        return ({"samples": t}, )

class VAEEncodeForInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", ), "mask": ("MASK", ), "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/inpaint"

    def encode(self, vae, pixels, mask, grow_mask_by=6):
        downscale_ratio = vae.spacial_compression_encode()
        x = (pixels.shape[1] // downscale_ratio) * downscale_ratio
        y = (pixels.shape[2] // downscale_ratio) * downscale_ratio
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % downscale_ratio) // 2
            y_offset = (pixels.shape[2] % downscale_ratio) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        if grow_mask_by == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        t = vae.encode(pixels)

        return ({"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}, )


class InpaintModelConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "pixels": ("IMAGE", ),
                             "mask": ("MASK", ),
                             "noise_mask": ("BOOLEAN", {"default": True, "tooltip": "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."}),
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/inpaint"

    def encode(self, positive, negative, pixels, vae, mask, noise_mask=True):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)


class SaveLatent:
    SEARCH_ALIASES = ["export latent"]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ),
                              "filename_prefix": ("STRING", {"default": "latents/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "save"

    OUTPUT_NODE = True

    CATEGORY = "_for_testing"

    def save(self, samples, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        # support save metadata for latent sharing
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = None
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        file = f"{filename}_{counter:05}_.latent"

        results: list[FileLocator] = []
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": "output"
        })

        file = os.path.join(full_output_folder, file)

        output = {}
        output["latent_tensor"] = samples["samples"].contiguous()
        output["latent_format_version_0"] = torch.tensor([])

        comfy.utils.save_torch_file(output, file, metadata=metadata)
        return { "ui": { "latents": results } }


class LoadLatent:
    SEARCH_ALIASES = ["import latent", "open latent"]

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".latent")]
        return {"required": {"latent": [sorted(files), ]}, }

    CATEGORY = "_for_testing"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "load"

    def load(self, latent):
        latent_path = folder_paths.get_annotated_filepath(latent)
        latent = safetensors.torch.load_file(latent_path, device="cpu")
        multiplier = 1.0
        if "latent_format_version_0" not in latent:
            multiplier = 1.0 / 0.18215
        samples = {"samples": latent["latent_tensor"].float() * multiplier}
        return (samples, )

    @classmethod
    def IS_CHANGED(s, latent):
        image_path = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        if not folder_paths.exists_annotated_filepath(latent):
            return "Invalid latent file: {}".format(latent)
        return True


class CheckpointLoader:
    SEARCH_ALIASES = ["load model", "model loader"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "config_name": (folder_paths.get_filename_list("configs"), ),
                              "ckpt_name": (folder_paths.get_filename_list("checkpoints"), )}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "advanced/loaders"
    DEPRECATED = True

    def load_checkpoint(self, config_name, ckpt_name):
        config_path = folder_paths.get_full_path("configs", config_name)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

class CheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."
    SEARCH_ALIASES = ["load model", "checkpoint", "model loader", "load checkpoint", "ckpt", "model"]

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]

class DiffusersLoader:
    SEARCH_ALIASES = ["load diffusers model"]

    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,), }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "advanced/loaders/deprecated"

    def load_checkpoint(self, model_path, output_vae=True, output_clip=True):
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break

        return comfy.diffusers_load.load_diffusers(model_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))


class unCLIPCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out

class CLIPSetLastLayer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip": ("CLIP", ),
                              "stop_at_clip_layer": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1, "advanced": True}),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "set_last_layer"

    CATEGORY = "conditioning"

    def set_last_layer(self, clip, stop_at_clip_layer):
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,)

class LoraLoader:
    ESSENTIALS_CATEGORY = "Image Generation"

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."
    SEARCH_ALIASES = ["lora", "load lora", "apply lora", "lora loader", "lora model"]

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class LoraLoaderModelOnly(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)

class VAELoader:
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]
    @staticmethod
    def vae_list(s):
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
            else:
                for tae in s.video_taes:
                    if v.startswith(tae):
                        vaes.append(v)

        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        vaes.append("pixel_space")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (s.vae_list(s), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "loaders"

    #TODO: scale factor?
    def load_vae(self, vae_name):
        metadata = None
        if vae_name == "pixel_space":
            sd = {}
            sd["pixel_space_vae"] = torch.tensor(1.0)
        elif vae_name in self.image_taes:
            sd = self.load_taesd(vae_name)
        else:
            if os.path.splitext(vae_name)[0] in self.video_taes:
                vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return (vae,)

class ControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"
    SEARCH_ALIASES = ["controlnet", "control net", "cn", "load controlnet", "controlnet loader"]

    def load_controlnet(self, control_net_name):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        if controlnet is None:
            raise RuntimeError("ERROR: controlnet file is invalid and does not contain a valid controlnet model.")
        return (controlnet,)

class DiffControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, model, control_net_name):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path, model)
        return (controlnet,)


class ControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"

    DEPRECATED = True
    CATEGORY = "conditioning/controlnet"

    def apply_controlnet(self, conditioning, control_net, image, strength):
        if strength == 0:
            return (conditioning, )

        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return (c, )


class ControlNetApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             },
                "optional": {"vae": ("VAE", ),
                             }
    }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "conditioning/controlnet"
    SEARCH_ALIASES = ["controlnet", "apply controlnet", "use controlnet", "control net"]

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None, extra_concat=[]):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


class HSWQFP8E4M3UNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "int8_tensorwise"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "advanced/loaders"
    TITLE = "HSWQ FP8 E4M3 UNet Loader"

    def load_unet(self, unet_name, weight_dtype):
        from ..patches.comfy_quant_int8 import (
            apply_comfy_quant_int8_patches,
            checkpoint_looks_like_comfy_quant_int8,
            reset_int8_lora_log_counters,
            summarize_int8_lora_capability,
        )

        # INT8 Conv2d + comfy_quant decode patches (core only handles Linear).
        apply_comfy_quant_int8_patches()

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        # Auto-detect native comfy_quant INT8 UNet; do not force float8 dtype over int8 weights.
        is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(unet_path)

        if is_int8:
            model_options = {}
            reset_int8_lora_log_counters()
            logging.info("[HSWQ INT8] Loading UNet via MixedPrecisionOps (int8_tensorwise / comfy_quant)")
            print(f"[HSWQ INT8] Loading UNet: {unet_name}", flush=True)
        else:
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        if is_int8:
            summarize_int8_lora_capability(model)

        return (model,)

class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nstable_diffusion: clip-l\nstable_cascade: clip-g\nsd3: t5 xxl/ clip-g / clip-l\nstable_audio: t5 base\nmochi: t5 xxl\ncosmos: old t5 xxl\nlumina2: gemma 2 2B\nwan: umt5 xxl\n hidream: llama-3.1 (Recommend) or t5\nomnigen2: qwen vl 2.5 3B"

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)

class DualCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image", "hunyuan_video_15", "kandinsky5", "kandinsky5_image", "ltxv", "newbie", "ace"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nsdxl: clip-l, clip-g\nsd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\nflux: clip-l, t5\nhidream: at least one of t5 or llama, recommended t5 and llama\nhunyuan_image: qwen2.5vl 7b and byt5 small\nnewbie: gemma-3-4b-it, jina clip v2"

    def load_clip(self, clip_name1, clip_name2, type, device="default"):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)

class CLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"

    CATEGORY = "loaders"

    def load_clip(self, clip_name):
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        if clip_vision is None:
            raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")
        return (clip_vision,)

class CLIPVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "image": ("IMAGE",),
                              "crop": (["center", "none"],)
                             }}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip_vision, image, crop):
        crop_image = True
        if crop != "center":
            crop_image = False
        output = clip_vision.encode_image(image, crop=crop_image)
        return (output,)

class StyleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "style_model_name": (folder_paths.get_filename_list("style_models"), )}}

    RETURN_TYPES = ("STYLE_MODEL",)
    FUNCTION = "load_style_model"

    CATEGORY = "loaders"

    def load_style_model(self, style_model_name):
        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model_name)
        style_model = comfy.sd.load_style_model(style_model_path)
        return (style_model,)


class StyleModelApply:
    SEARCH_ALIASES = ["style transfer"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                             "strength_type": (["multiply", "attn_bias"], ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, conditioning, style_model, clip_vision_output, strength, strength_type):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            # even if the strength is 1.0 (i.e, no change), if there's already a mask, we have to add to it
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                # math.log raises an error if the argument is zero
                # torch.log returns -inf, which is what we want
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                # get the size of the mask image
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                # grab the existing mask
                mask = keys.get("attention_mask", None)
                # create a default mask if it doesn't exist
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                # convert the mask dtype, because it might be boolean
                # we want it to be interpreted as a bias
                if mask.dtype == torch.bool:
                    # log(True) = log(1) = 0
                    # log(False) = log(0) = -inf
                    mask = torch.log(mask.to(dtype=torch.float16))
                # now we make the mask bigger to add space for our new tokens
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                # copy over the old mask, in quandrants
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                # now fill in the attention bias to our redux tokens
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)

class unCLIPConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "noise_augmentation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_adm"

    CATEGORY = "conditioning"

    def apply_adm(self, conditioning, clip_vision_output, strength, noise_augmentation):
        if strength == 0:
            return (conditioning, )

        c = node_helpers.conditioning_set_values(conditioning, {"unclip_conditioning": [{"clip_vision_output": clip_vision_output, "strength": strength, "noise_augmentation": noise_augmentation}]}, append=True)
        return (c, )

class GLIGENLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "gligen_name": (folder_paths.get_filename_list("gligen"), )}}

    RETURN_TYPES = ("GLIGEN",)
    FUNCTION = "load_gligen"

    CATEGORY = "loaders"

    def load_gligen(self, gligen_name):
        gligen_path = folder_paths.get_full_path_or_raise("gligen", gligen_name)
        gligen = comfy.sd.load_gligen(gligen_path)
        return (gligen,)

class GLIGENTextBoxApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ),
                              "clip": ("CLIP", ),
                              "gligen_textbox_model": ("GLIGEN", ),
                              "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                              "width": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning/gligen"

    def append(self, conditioning_to, clip, gligen_textbox_model, text, width, height, x, y):
        c = []
        cond, cond_pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled="unprojected")
        for t in conditioning_to:
            n = [t[0], t[1].copy()]
            position_params = [(cond_pooled, height // 8, width // 8, y // 8, x // 8)]
            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]

            n[1]['gligen'] = ("position", gligen_textbox_model, prev + position_params)
            c.append(n)
        return (c, )

class EmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"

    CATEGORY = "latent"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."
    SEARCH_ALIASES = ["empty", "empty latent", "new latent", "create latent", "blank latent", "blank"]

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples": latent, "downscale_ratio_spacial": 8}, )


class LatentFromBatch:
    SEARCH_ALIASES = ["select from batch", "pick latent", "batch subset"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                              "length": ("INT", {"default": 1, "min": 1, "max": 64}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "frombatch"

    CATEGORY = "latent/batch"

    def frombatch(self, samples, batch_index, length):
        s = samples.copy()
        s_in = samples["samples"]
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s["samples"] = s_in[batch_index:batch_index + length].clone()
        if "noise_mask" in samples:
            masks = samples["noise_mask"]
            if masks.shape[0] == 1:
                s["noise_mask"] = masks.clone()
            else:
                if masks.shape[0] < s_in.shape[0]:
                    masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
                s["noise_mask"] = masks[batch_index:batch_index + length].clone()
        if "batch_index" not in s:
            s["batch_index"] = [x for x in range(batch_index, batch_index+length)]
        else:
            s["batch_index"] = samples["batch_index"][batch_index:batch_index + length]
        return (s,)

class RepeatLatentBatch:
    SEARCH_ALIASES = ["duplicate latent", "clone latent"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 64}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat"

    CATEGORY = "latent/batch"

    def repeat(self, samples, amount):
        s = samples.copy()
        s_in = samples["samples"]

        s["samples"] = s_in.repeat((amount,) + ((1,) * (s_in.ndim - 1)))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat((math.ceil(s_in.shape[0] / masks.shape[0]),) + ((1,) * (masks.ndim - 1)))[:s_in.shape[0]]
            s["noise_mask"] = samples["noise_mask"].repeat((amount,) + ((1,) * (samples["noise_mask"].ndim - 1)))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
        return (s,)

class LatentUpscale:
    SEARCH_ALIASES = ["enlarge latent", "resize latent"]

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, samples, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = samples
        else:
            s = samples.copy()

            if width == 0:
                height = max(64, height)
                width = max(64, round(samples["samples"].shape[-1] * height / samples["samples"].shape[-2]))
            elif height == 0:
                width = max(64, width)
                height = max(64, round(samples["samples"].shape[-2] * width / samples["samples"].shape[-1]))
            else:
                width = max(64, width)
                height = max(64, height)

            s["samples"] = comfy.utils.common_upscale(samples["samples"], width // 8, height // 8, upscale_method, crop)
        return (s,)

class LatentUpscaleBy:
    SEARCH_ALIASES = ["enlarge latent", "resize latent", "scale latent"]

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), "upscale_method": (s.upscale_methods,),
                              "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, samples, upscale_method, scale_by):
        s = samples.copy()
        width = round(samples["samples"].shape[-1] * scale_by)
        height = round(samples["samples"].shape[-2] * scale_by)
        s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, "disabled")
        return (s,)

class LatentRotate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "rotation": (["none", "90 degrees", "180 degrees", "270 degrees"],),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "rotate"

    CATEGORY = "latent/transform"

    def rotate(self, samples, rotation):
        s = samples.copy()
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        s["samples"] = torch.rot90(samples["samples"], k=rotate_by, dims=[3, 2])
        return (s,)

class LatentFlip:
    SEARCH_ALIASES = ["mirror latent"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "flip_method": (["x-axis: vertically", "y-axis: horizontally"],),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "flip"

    CATEGORY = "latent/transform"

    def flip(self, samples, flip_method):
        s = samples.copy()
        if flip_method.startswith("x"):
            s["samples"] = torch.flip(samples["samples"], dims=[2])
        elif flip_method.startswith("y"):
            s["samples"] = torch.flip(samples["samples"], dims=[3])

        return (s,)

class LatentComposite:
    SEARCH_ALIASES = ["overlay latent", "layer latent", "paste latent"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples_to": ("LATENT",),
                              "samples_from": ("LATENT",),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "feather": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"

    CATEGORY = "latent"

    def composite(self, samples_to, samples_from, x, y, composite_method="normal", feather=0):
        x =  x // 8
        y = y // 8
        feather = feather // 8
        samples_out = samples_to.copy()
        s = samples_to["samples"].clone()
        samples_to = samples_to["samples"]
        samples_from = samples_from["samples"]
        if feather == 0:
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        else:
            samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
            mask = torch.ones_like(samples_from)
            for t in range(feather):
                if y != 0:
                    mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))

                if y + samples_from.shape[2] < samples_to.shape[2]:
                    mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                if x != 0:
                    mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                if x + samples_from.shape[3] < samples_to.shape[3]:
                    mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
            rev_mask = torch.ones_like(mask) - mask
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask
        samples_out["samples"] = s
        return (samples_out,)

class LatentBlend:
    SEARCH_ALIASES = ["mix latents", "interpolate latents"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples1": ("LATENT",),
            "samples2": ("LATENT",),
            "blend_factor": ("FLOAT", {
                "default": 0.5,
                "min": 0,
                "max": 1,
                "step": 0.01
            }),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend"

    CATEGORY = "_for_testing"

    def blend(self, samples1, samples2, blend_factor:float, blend_mode: str="normal"):

        samples_out = samples1.copy()
        samples1 = samples1["samples"]
        samples2 = samples2["samples"]

        if samples1.shape != samples2.shape:
            samples2.permute(0, 3, 1, 2)
            samples2 = comfy.utils.common_upscale(samples2, samples1.shape[3], samples1.shape[2], 'bicubic', crop='center')
            samples2.permute(0, 2, 3, 1)

        samples_blended = self.blend_mode(samples1, samples2, blend_mode)
        samples_blended = samples1 * blend_factor + samples_blended * (1 - blend_factor)
        samples_out["samples"] = samples_blended
        return (samples_out,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

class LatentCrop:
    SEARCH_ALIASES = ["trim latent", "cut latent"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "crop"

    CATEGORY = "latent/transform"

    def crop(self, samples, width, height, x, y):
        s = samples.copy()
        samples = samples['samples']
        x =  x // 8
        y = y // 8

        #enfonce minimum size of 64
        if x > (samples.shape[3] - 8):
            x = samples.shape[3] - 8
        if y > (samples.shape[2] - 8):
            y = samples.shape[2] - 8

        new_height = height // 8
        new_width = width // 8
        to_x = new_width + x
        to_y = new_height + y
        s['samples'] = samples[:,:,y:to_y, x:to_x]
        return (s,)

class SetLatentNoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "mask": ("MASK",),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"

    CATEGORY = "latent/inpaint"

    def set_mask(self, samples, mask):
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return (s,)

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out, )

class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."
    SEARCH_ALIASES = ["sampler", "sample", "generate", "denoise", "diffuse", "txt2img", "img2img"]

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

class KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], {"advanced": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "advanced": True}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "advanced": True}),
                    "return_with_leftover_noise": (["disable", "enable"], {"advanced": True}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

class SaveImage:
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
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    ESSENTIALS_CATEGORY = "Basics"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."
    SEARCH_ALIASES = ["save", "save image", "export image", "output image", "write image", "download"]

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class PreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    SEARCH_ALIASES = ["preview", "preview image", "show image", "view image", "display image", "image viewer"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"
    ESSENTIALS_CATEGORY = "Basics"
    SEARCH_ALIASES = ["load image", "open image", "import image", "image input", "upload image", "read image", "image loader"]

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

            if img.format == "MPO":
                break  # ignore all frames except the first one for MPO format

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class LoadImageMask:
    SEARCH_ALIASES = ["import mask", "alpha mask", "channel mask"]

    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    def load_image(self, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        i = node_helpers.pillow(Image.open, image_path)
        i = node_helpers.pillow(ImageOps.exif_transpose, i)
        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)

    @classmethod
    def IS_CHANGED(s, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class LoadImageOutput(LoadImage):
    SEARCH_ALIASES = ["output image", "previous generation"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("COMBO", {
                    "image_upload": True,
                    "image_folder": "output",
                    "remote": {
                        "route": "/internal/files/output",
                        "refresh_button": True,
                        "control_after_refresh": "first",
                    },
                }),
            }
        }

    DESCRIPTION = "Load an image from the output folder. When the refresh button is clicked, the node will update the image list and automatically select the first image, allowing for easy iteration."
    EXPERIMENTAL = True
    FUNCTION = "load_image"


class ImageScale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    ESSENTIALS_CATEGORY = "Image Tools"
    SEARCH_ALIASES = ["resize", "resize image", "scale image", "image resize", "zoom", "zoom in", "change size"]

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)

class ImageScaleBy:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, scale_by):
        samples = image.movedim(-1,1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

class ImageInvert:
    SEARCH_ALIASES = ["reverse colors"]
    ESSENTIALS_CATEGORY = "Image Tools"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"

    CATEGORY = "image"

    def invert(self, image):
        s = 1.0 - image
        return (s,)

class ImageBatch:
    SEARCH_ALIASES = ["combine images", "merge images", "stack images"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image1": ("IMAGE",), "image2": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"

    CATEGORY = "image"
    DEPRECATED = True

    def batch(self, image1, image2):
        if image1.shape[-1] != image2.shape[-1]:
            if image1.shape[-1] > image2.shape[-1]:
                image2 = torch.nn.functional.pad(image2, (0,1), mode='constant', value=1.0)
            else:
                image1 = torch.nn.functional.pad(image1, (0,1), mode='constant', value=1.0)
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
        s = torch.cat((image1, image2), dim=0)
        return (s,)

class EmptyImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1), )

class ImagePadForOutpaint:
    SEARCH_ALIASES = ["extend canvas", "expand image"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1, "advanced": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask.unsqueeze(0))


NODE_CLASS_MAPPINGS = {
    "KSampler": KSampler,
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "CLIPTextEncode": CLIPTextEncode,
    "CLIPSetLastLayer": CLIPSetLastLayer,
    "VAEDecode": VAEDecode,
    "VAEEncode": VAEEncode,
    "VAEEncodeForInpaint": VAEEncodeForInpaint,
    "VAELoader": VAELoader,
    "EmptyLatentImage": EmptyLatentImage,
    "LatentUpscale": LatentUpscale,
    "LatentUpscaleBy": LatentUpscaleBy,
    "LatentFromBatch": LatentFromBatch,
    "RepeatLatentBatch": RepeatLatentBatch,
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "LoadImage": LoadImage,
    "LoadImageMask": LoadImageMask,
    "LoadImageOutput": LoadImageOutput,
    "ImageScale": ImageScale,
    "ImageScaleBy": ImageScaleBy,
    "ImageInvert": ImageInvert,
    "ImageBatch": ImageBatch,
    "ImagePadForOutpaint": ImagePadForOutpaint,
    "EmptyImage": EmptyImage,
    "ConditioningAverage": ConditioningAverage ,
    "ConditioningCombine": ConditioningCombine,
    "ConditioningConcat": ConditioningConcat,
    "ConditioningSetArea": ConditioningSetArea,
    "ConditioningSetAreaPercentage": ConditioningSetAreaPercentage,
    "ConditioningSetAreaStrength": ConditioningSetAreaStrength,
    "ConditioningSetMask": ConditioningSetMask,
    "KSamplerAdvanced": KSamplerAdvanced,
    "SetLatentNoiseMask": SetLatentNoiseMask,
    "LatentComposite": LatentComposite,
    "LatentBlend": LatentBlend,
    "LatentRotate": LatentRotate,
    "LatentFlip": LatentFlip,
    "LatentCrop": LatentCrop,
    "LoraLoader": LoraLoader,
    "CLIPLoader": CLIPLoader,
    "HSWQFP8E4M3UNetLoader": HSWQFP8E4M3UNetLoader,
    "DualCLIPLoader": DualCLIPLoader,
    "CLIPVisionEncode": CLIPVisionEncode,
    "StyleModelApply": StyleModelApply,
    "unCLIPConditioning": unCLIPConditioning,
    "ControlNetApply": ControlNetApply,
    "ControlNetApplyAdvanced": ControlNetApplyAdvanced,
    "ControlNetLoader": ControlNetLoader,
    "DiffControlNetLoader": DiffControlNetLoader,
    "StyleModelLoader": StyleModelLoader,
    "CLIPVisionLoader": CLIPVisionLoader,
    "VAEDecodeTiled": VAEDecodeTiled,
    "VAEEncodeTiled": VAEEncodeTiled,
    "unCLIPCheckpointLoader": unCLIPCheckpointLoader,
    "GLIGENLoader": GLIGENLoader,
    "GLIGENTextBoxApply": GLIGENTextBoxApply,
    "InpaintModelConditioning": InpaintModelConditioning,

    "CheckpointLoader": CheckpointLoader,
    "DiffusersLoader": DiffusersLoader,

    "LoadLatent": LoadLatent,
    "SaveLatent": SaveLatent,

    "ConditioningZeroOut": ConditioningZeroOut,
    "ConditioningSetTimestepRange": ConditioningSetTimestepRange,
    "LoraLoaderModelOnly": LoraLoaderModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Sampling
    "KSampler": "KSampler",
    "KSamplerAdvanced": "KSampler (Advanced)",
    # Loaders
    "CheckpointLoader": "Load Checkpoint With Config (DEPRECATED)",
    "CheckpointLoaderSimple": "Load Checkpoint",
    "VAELoader": "Load VAE",
    "LoraLoader": "Load LoRA (Model and CLIP)",
    "LoraLoaderModelOnly": "Load LoRA",
    "CLIPLoader": "Load CLIP",
    "ControlNetLoader": "Load ControlNet Model",
    "DiffControlNetLoader": "Load ControlNet Model (diff)",
    "StyleModelLoader": "Load Style Model",
    "CLIPVisionLoader": "Load CLIP Vision",
    "UNETLoader": "Load Diffusion Model",
    # Conditioning
    "CLIPVisionEncode": "CLIP Vision Encode",
    "StyleModelApply": "Apply Style Model",
    "CLIPTextEncode": "CLIP Text Encode (Prompt)",
    "CLIPSetLastLayer": "CLIP Set Last Layer",
    "ConditioningCombine": "Conditioning (Combine)",
    "ConditioningAverage ": "Conditioning (Average)",
    "ConditioningConcat": "Conditioning (Concat)",
    "ConditioningSetArea": "Conditioning (Set Area)",
    "ConditioningSetAreaPercentage": "Conditioning (Set Area with Percentage)",
    "ConditioningSetMask": "Conditioning (Set Mask)",
    "ControlNetApply": "Apply ControlNet (OLD)",
    "ControlNetApplyAdvanced": "Apply ControlNet",
    # Latent
    "VAEEncodeForInpaint": "VAE Encode (for Inpainting)",
    "SetLatentNoiseMask": "Set Latent Noise Mask",
    "VAEDecode": "VAE Decode",
    "VAEEncode": "VAE Encode",
    "LatentRotate": "Rotate Latent",
    "LatentFlip": "Flip Latent",
    "LatentCrop": "Crop Latent",
    "EmptyLatentImage": "Empty Latent Image",
    "LatentUpscale": "Upscale Latent",
    "LatentUpscaleBy": "Upscale Latent By",
    "LatentComposite": "Latent Composite",
    "LatentBlend": "Latent Blend",
    "LatentFromBatch" : "Latent From Batch",
    "RepeatLatentBatch": "Repeat Latent Batch",
    # Image
    "SaveImage": "Save Image",
    "PreviewImage": "Preview Image",
    "LoadImage": "Load Image",
    "LoadImageMask": "Load Image (as Mask)",
    "LoadImageOutput": "Load Image (from Outputs)",
    "ImageScale": "Upscale Image",
    "ImageScaleBy": "Upscale Image By",
    "ImageInvert": "Invert Image",
    "ImagePadForOutpaint": "Pad Image for Outpainting",
    "ImageBatch": "Batch Images",
    "ImageCrop": "Image Crop",
    "ImageStitch": "Image Stitch",
    "ImageBlend": "Image Blend",
    "ImageBlur": "Image Blur",
    "ImageQuantize": "Image Quantize",
    "ImageSharpen": "Image Sharpen",
    "ImageScaleToTotalPixels": "Scale Image to Total Pixels",
    "GetImageSize": "Get Image Size",
    # _for_testing
    "VAEDecodeTiled": "VAE Decode (Tiled)",
    "VAEEncodeTiled": "VAE Encode (Tiled)",
}

EXTENSION_WEB_DIRS = {}

# Dictionary of successfully loaded module names and associated directories.
LOADED_MODULE_DIRS = {}


def get_module_name(module_path: str) -> str:
    """
    Returns the module name based on the given module path.
    Examples:
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.disabled") -> "custom_nodes
    Args:
        module_path (str): The path of the module.
    Returns:
        str: The module name.
    """
    base_path = os.path.basename(module_path)
    if os.path.isfile(module_path):
        base_path = os.path.splitext(base_path)[0]
    return base_path


async def load_custom_node(module_path: str, ignore=set(), module_parent="custom_nodes") -> bool:
    module_name = get_module_name(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
        sys_module_name = module_name
    elif os.path.isdir(module_path):
        sys_module_name = module_path.replace(".", "_x_")

    try:
        logging.debug("Trying to load custom node {}".format(module_path))
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(sys_module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(sys_module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[sys_module_name] = module
        module_spec.loader.exec_module(module)

        LOADED_MODULE_DIRS[module_name] = os.path.abspath(module_dir)

        try:
            from comfy_config import config_parser

            project_config = config_parser.extract_node_configuration(module_path)

            web_dir_name = project_config.tool_comfy.web

            if web_dir_name:
                web_dir_path = os.path.join(module_path, web_dir_name)

                if os.path.isdir(web_dir_path):
                    project_name = project_config.project.name

                    EXTENSION_WEB_DIRS[project_name] = web_dir_path

                    logging.info("Automatically register web folder {} for {}".format(web_dir_name, project_name))
        except Exception as e:
            logging.warning(f"Unable to parse pyproject.toml due to lack dependency pydantic-settings, please run 'pip install -r requirements.txt': {e}")

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if os.path.isdir(web_dir):
                EXTENSION_WEB_DIRS[module_name] = web_dir

        # V1 node definition
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
                if name not in ignore:
                    NODE_CLASS_MAPPINGS[name] = node_cls
                    node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            return True
        # V3 Extension Definition
        elif hasattr(module, "comfy_entrypoint"):
            entrypoint = getattr(module, "comfy_entrypoint")
            if not callable(entrypoint):
                logging.warning(f"comfy_entrypoint in {module_path} is not callable, skipping.")
                return False
            try:
                if inspect.iscoroutinefunction(entrypoint):
                    extension = await entrypoint()
                else:
                    extension = entrypoint()
                if not isinstance(extension, ComfyExtension):
                    logging.warning(f"comfy_entrypoint in {module_path} did not return a ComfyExtension, skipping.")
                    return False
                await extension.on_load()
                node_list = await extension.get_node_list()
                if not isinstance(node_list, list):
                    logging.warning(f"comfy_entrypoint in {module_path} did not return a list of nodes, skipping.")
                    return False
                for node_cls in node_list:
                    node_cls: io.ComfyNode
                    schema = node_cls.GET_SCHEMA()
                    if schema.node_id not in ignore:
                        NODE_CLASS_MAPPINGS[schema.node_id] = node_cls
                        node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))
                    if schema.display_name is not None:
                        NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name
                return True
            except Exception as e:
                logging.warning(f"Error while calling comfy_entrypoint in {module_path}: {e}")
                return False
        else:
            logging.warning(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS or NODES_LIST (need one).")
            return False
    except Exception as e:
        logging.warning(traceback.format_exc())
        logging.warning(f"Cannot import {module_path} module for custom nodes: {e}")
        return False

async def init_external_custom_nodes():
    """
    Initializes the external custom nodes.

    This function loads custom nodes from the specified folder paths and imports them into the application.
    It measures the import times for each custom node and logs the results.

    Returns:
        None
    """
    base_node_names = set(NODE_CLASS_MAPPINGS.keys())
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(os.path.realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py":
                continue
            if module_path.endswith(".disabled"):
                continue
            if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                logging.info(f"Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                continue

            if args.enable_manager:
                if comfyui_manager.should_be_disabled(module_path):
                    logging.info(f"Blocked by policy: {module_path}")
                    continue

            time_before = time.perf_counter()
            success = await load_custom_node(module_path, base_node_names, module_parent="custom_nodes")
            node_import_times.append((time.perf_counter() - time_before, module_path, success))

    if len(node_import_times) > 0:
        logging.info("\nImport times for custom nodes:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

async def init_builtin_extra_nodes():
    """
    Initializes the built-in extra nodes in ComfyUI.

    This function loads the extra node files located in the "comfy_extras" directory and imports them into ComfyUI.
    If any of the extra node files fail to import, a warning message is logged.

    Returns:
        None
    """
    extras_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_extras")
    extras_files = [
        "nodes_latent.py",
        "nodes_hypernetwork.py",
        "nodes_upscale_model.py",
        "nodes_post_processing.py",
        "nodes_mask.py",
        "nodes_compositing.py",
        "nodes_rebatch.py",
        "nodes_model_merging.py",
        "nodes_tomesd.py",
        "nodes_clip_sdxl.py",
        "nodes_canny.py",
        "nodes_freelunch.py",
        "nodes_custom_sampler.py",
        "nodes_hypertile.py",
        "nodes_model_advanced.py",
        "nodes_model_downscale.py",
        "nodes_images.py",
        "nodes_video_model.py",
        "nodes_train.py",
        "nodes_dataset.py",
        "nodes_sag.py",
        "nodes_perpneg.py",
        "nodes_stable3d.py",
        "nodes_sdupscale.py",
        "nodes_photomaker.py",
        "nodes_pixart.py",
        "nodes_cond.py",
        "nodes_morphology.py",
        "nodes_stable_cascade.py",
        "nodes_differential_diffusion.py",
        "nodes_ip2p.py",
        "nodes_model_merging_model_specific.py",
        "nodes_pag.py",
        "nodes_align_your_steps.py",
        "nodes_attention_multiply.py",
        "nodes_advanced_samplers.py",
        "nodes_webcam.py",
        "nodes_audio.py",
        "nodes_sd3.py",
        "nodes_gits.py",
        "nodes_controlnet.py",
        "nodes_hunyuan.py",
        "nodes_eps.py",
        "nodes_flux.py",
        "nodes_lora_extract.py",
        "nodes_torch_compile.py",
        "nodes_mochi.py",
        "nodes_slg.py",
        "nodes_mahiro.py",
        "nodes_lt_upsampler.py",
        "nodes_lt_audio.py",
        "nodes_lt.py",
        "nodes_hooks.py",
        "nodes_load_3d.py",
        "nodes_cosmos.py",
        "nodes_video.py",
        "nodes_lumina2.py",
        "nodes_wan.py",
        "nodes_lotus.py",
        "nodes_hunyuan3d.py",
        "nodes_primitive.py",
        "nodes_cfg.py",
        "nodes_optimalsteps.py",
        "nodes_hidream.py",
        "nodes_fresca.py",
        "nodes_apg.py",
        "nodes_preview_any.py",
        "nodes_ace.py",
        "nodes_string.py",
        "nodes_camera_trajectory.py",
        "nodes_edit_model.py",
        "nodes_tcfg.py",
        "nodes_context_windows.py",
        "nodes_qwen.py",
        "nodes_chroma_radiance.py",
        "nodes_model_patch.py",
        "nodes_easycache.py",
        "nodes_audio_encoder.py",
        "nodes_rope.py",
        "nodes_logic.py",
        "nodes_resolution.py",
        "nodes_nop.py",
        "nodes_kandinsky5.py",
        "nodes_wanmove.py",
        "nodes_image_compare.py",
        "nodes_zimage.py",
        "nodes_glsl.py",
        "nodes_lora_debug.py",
        "nodes_textgen.py",
        "nodes_color.py",
        "nodes_toolkit.py",
        "nodes_replacements.py",
        "nodes_nag.py",
        "nodes_sdpose.py",
        "nodes_math.py",
    ]

    import_failed = []
    for node_file in extras_files:
        if not await load_custom_node(os.path.join(extras_dir, node_file), module_parent="comfy_extras"):
            import_failed.append(node_file)

    return import_failed


async def init_builtin_api_nodes():
    api_nodes_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_api_nodes")
    api_nodes_files = sorted(glob.glob(os.path.join(api_nodes_dir, "nodes_*.py")))

    import_failed = []
    for node_file in api_nodes_files:
        if not await load_custom_node(node_file, module_parent="comfy_api_nodes"):
            import_failed.append(os.path.basename(node_file))

    return import_failed

async def init_public_apis():
    register_versions([
        ComfyAPIWithVersion(
            version=getattr(v, "VERSION"),
            api_class=v
        ) for v in supported_versions
    ])

async def init_extra_nodes(init_custom_nodes=True, init_api_nodes=True):
    await init_public_apis()

    import_failed = await init_builtin_extra_nodes()

    import_failed_api = []
    if init_api_nodes:
        import_failed_api = await init_builtin_api_nodes()

    if init_custom_nodes:
        await init_external_custom_nodes()
    else:
        logging.info("Skipping loading of custom nodes")

    if len(import_failed_api) > 0:
        logging.warning("WARNING: some comfy_api_nodes/ nodes did not import correctly. This may be because they are missing some dependencies.\n")
        for node in import_failed_api:
            logging.warning("IMPORT FAILED: {}".format(node))
        logging.warning("\nThis issue might be caused by new missing dependencies added the last time you updated ComfyUI.")
        if args.windows_standalone_build:
            logging.warning("Please run the update script: update/update_comfyui.bat")
        else:
            logging.warning("Please do a: pip install -r requirements.txt")
        logging.warning("")

    if len(import_failed) > 0:
        logging.warning("WARNING: some comfy_extras/ nodes did not import correctly. This may be because they are missing some dependencies.\n")
        for node in import_failed:
            logging.warning("IMPORT FAILED: {}".format(node))
        logging.warning("\nThis issue might be caused by new missing dependencies added the last time you updated ComfyUI.")
        if args.windows_standalone_build:
            logging.warning("Please run the update script: update/update_comfyui.bat")
        else:
            logging.warning("Please do a: pip install -r requirements.txt")
        logging.warning("")

    return import_failed
UNETLoader = HSWQFP8E4M3UNetLoader
```

### 3.5 `patches/zimage_fp8_torchcompile.py`（全文・247 行）

```python
"""
Z Image + FP8 E4M3 + torch.compile 時の mat1/mat2 shape 不一致を防ぐパッチ（ComfyUI-nunchaku-unofficial-loader 内）。

- LoRA: reshape や lora 出力要素数が weight と一致しない場合はそのレイヤへの適用をスキップする。
- comfy.ops Linear: torch.compile 中、または input.shape[-1] != weight.shape[1] のときは
  3D/QuantizedTensor 経路を通さず、不一致時は入力を weight.shape[1] でスライスしてクラッシュを防ぐ。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False


def _apply_ops_patch() -> bool:
    """comfy.ops.mixed_precision_ops をラップし、torch.compile / shape 不一致時に 3D/FP8 input 経路をスキップする。"""
    try:
        import torch
        import comfy.ops as ops_module
    except ImportError:
        return False

    original_mixed_precision_ops = getattr(ops_module, "mixed_precision_ops", None)
    if original_mixed_precision_ops is None or not callable(original_mixed_precision_ops):
        return False

    def patched_mixed_precision_ops(quant_config=None, compute_dtype=None, full_precision_mm=False, disabled=None):
        if quant_config is None:
            quant_config = {}
        if disabled is None:
            disabled = []
        result = original_mixed_precision_ops(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        Linear = getattr(result, "Linear", None)
        if Linear is None:
            return result
        _orig_forward = getattr(Linear, "forward", None)
        if _orig_forward is None or not callable(_orig_forward):
            return result

        run_every_op = getattr(ops_module, "run_every_op", None)
        if run_every_op is None:
            return result

        def _forward_compile_safe(self, input, *args, **kwargs):
            run_every_op()
            in_features = getattr(self, "weight", None)
            in_features = in_features.shape[1] if in_features is not None else None
            compiling = torch.compiler.is_compiling()
            shape_mismatch = in_features is not None and input.shape[-1] != in_features
            if compiling or shape_mismatch:
                if shape_mismatch:
                    logger.warning(
                        "ComfyUI-nunchaku: Linear input.shape[-1]=%s != weight.shape[1]=%s, slicing input [z_image/FP8/torch.compile compat]",
                        input.shape[-1],
                        in_features,
                    )
                    input = input[..., :in_features].contiguous()
                if input.ndim == 3:
                    input_shape = input.shape
                    input = input.reshape(-1, input_shape[2])
                    compute_dtype = getattr(input, "dtype", None)
                    output = self.forward_comfy_cast_weights(input, compute_dtype, want_requant=False)
                    output = output.reshape((input_shape[0], input_shape[1], self.weight.shape[0]))
                else:
                    compute_dtype = getattr(input, "dtype", None)
                    output = self.forward_comfy_cast_weights(input, compute_dtype, want_requant=False)
                return output
            return _orig_forward(self, input, *args, **kwargs)

        Linear.forward = _forward_compile_safe
        return result

    ops_module.mixed_precision_ops = patched_mixed_precision_ops
    return True


def _apply_lora_patch() -> bool:
    """LoraDiff.calculate_weight をラップし、z_image/FP8 時は reshape ・要素数不一致でスキップする。"""
    try:
        import torch
        import comfy.weight_adapter.lora as lora_module
    except ImportError:
        return False

    LoraDiff = getattr(lora_module, "LoraDiff", None)
    if LoraDiff is None:
        return False
    _original_calculate_weight = getattr(LoraDiff, "calculate_weight", None)
    if _original_calculate_weight is None or not callable(_original_calculate_weight):
        return False

    def _patched_calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=None,
        original_weight=None,
    ):
        if intermediate_dtype is None:
            intermediate_dtype = torch.float32
        v = self.weights
        reshape = v[5]
        try:
            from .comfy_quant_int8 import record_lora_shape_skip
        except ImportError:
            record_lora_shape_skip = None

        if reshape is not None and tuple(reshape) != weight.shape:
            reason = f"reshape {list(reshape)} != weight.shape {list(weight.shape)}"
            logger.warning(
                "LoRA %s: skipping %s (%s) [z_image/FP8/torch.compile compat]",
                self.name, key, reason,
            )
            if record_lora_shape_skip is not None:
                record_lora_shape_skip(getattr(self, "name", "?"), key, reason)
            return weight
        try:
            lora_diff_flat = torch.mm(v[0].flatten(start_dim=1), v[1].flatten(start_dim=1))
            if lora_diff_flat.numel() != weight.numel():
                reason = (
                    f"lora output size {lora_diff_flat.numel()} != weight size {weight.numel()}"
                )
                logger.warning(
                    "LoRA %s: skipping %s (%s) [z_image/FP8/torch.compile compat]",
                    self.name, key, reason,
                )
                if record_lora_shape_skip is not None:
                    record_lora_shape_skip(getattr(self, "name", "?"), key, reason)
                return weight
        except Exception as e:
            reason = f"error during lora_diff_flat check: {e}"
            logger.warning(
                "LoRA %s: skipping %s (%s) [z_image/FP8/torch.compile compat]",
                self.name, key, reason,
            )
            if record_lora_shape_skip is not None:
                record_lora_shape_skip(getattr(self, "name", "?"), key, reason)
            return weight
        return _original_calculate_weight(
            self,
            weight=weight,
            key=key,
            strength=strength,
            strength_model=strength_model,
            offset=offset,
            function=function,
            intermediate_dtype=intermediate_dtype,
            original_weight=original_weight,
        )

    LoraDiff.calculate_weight = _patched_calculate_weight
    return True


def _apply_rmsnorm_patch() -> bool:
    """
    comfy.ops.RMSNorm.forward_comfy_cast_weights をラップし、
    normalized_shape[0] と input.shape[-1] が異なる場合でも落ちないようにする。

    - 形が一致している場合: 元の実装のまま
    - 形が不一致の場合: weight の方を input の次元数に合わせてスライス／パディングし、torch.rms_norm を直接呼ぶ
    """
    try:
        import torch
        import comfy.ops as ops_module
    except ImportError:
        return False

    RMSNorm = getattr(ops_module, "RMSNorm", None)
    if RMSNorm is None:
        return False

    _orig_forward_comfy = getattr(RMSNorm, "forward_comfy_cast_weights", None)
    if _orig_forward_comfy is None or getattr(_orig_forward_comfy, "_hswq_rmsnorm_patched", False):
        return False

    def _forward_comfy_cast_weights_safe(self, input):
        norm_shape = getattr(self, "normalized_shape", None)
        target_dim = None
        if isinstance(norm_shape, (tuple, list)) and len(norm_shape) > 0:
            try:
                target_dim = int(norm_shape[0])
            except Exception:
                target_dim = None
        elif isinstance(norm_shape, int):
            target_dim = norm_shape

        last_dim = input.shape[-1]

        if target_dim is not None and last_dim != target_dim:
            logger.warning(
                "ComfyUI-nunchaku: RMSNorm input.shape[-1]=%s != normalized_shape[0]=%s, adjusting to input dim [z_image/FP8/torch.compile compat]",
                last_dim,
                target_dim,
            )
            weight = getattr(self, "weight", None)
            eff_weight = None
            if weight is not None:
                if weight.shape[-1] >= last_dim:
                    eff_weight = weight[..., :last_dim]
                else:
                    pad = last_dim - weight.shape[-1]
                    eff_weight = torch.nn.functional.pad(weight, (0, pad), value=1.0)
            return torch.rms_norm(input, last_dim, eff_weight, self.eps)

        return _orig_forward_comfy(self, input)

    setattr(_forward_comfy_cast_weights_safe, "_hswq_rmsnorm_patched", True)
    RMSNorm.forward_comfy_cast_weights = _forward_comfy_cast_weights_safe
    return True


def apply_zimage_fp8_torchcompile_patches() -> bool:
    """
    Z Image FP8 E4M3 + torch.compile 互換パッチを適用する。
    重複適用はスキップする。戻り値はパッチが適用されたかどうか。
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return True
    ok_ops = _apply_ops_patch()
    ok_lora = _apply_lora_patch()
    ok_rmsnorm = _apply_rmsnorm_patch()
    if ok_ops:
        logger.info(
            "ComfyUI-nunchaku: applied z_image FP8/torch.compile ops patch (skip 3D/QuantizedTensor path when compiling or shape mismatch)"
        )
    if ok_lora:
        logger.info(
            "ComfyUI-nunchaku: applied z_image FP8/torch.compile LoRA compat patch (reshape/numel skip)"
        )
    if ok_rmsnorm:
        logger.info(
            "ComfyUI-nunchaku: applied z_image FP8/torch.compile RMSNorm compat patch (normalized_shape/input mismatch safe)"
        )
    _PATCHES_APPLIED = ok_ops or ok_lora or ok_rmsnorm
    return _PATCHES_APPLIED
```

---

## ④ その意味（モジュール／関数単位の技術解説）

### 4.1 `patches/comfy_quant_int8.py` — 全体の役割

拡張の **INT8 + LoRA 実装の単一入口**。ComfyUI コアを書き換えず、import 時に `apply_comfy_quant_int8_patches()` で既存モジュールの関数を差し替える。

#### 4.1.1 ログ／状態グローバル

| 名前 | 意味 |
|------|------|
| `_lora_attach_last` / `_lora_attach_history` | 直近およびスタックされた LoRA ごとの適用・スキップ集計 |
| `_lora_bake_by_key` | キー → `"requant"` または `"int8_round"` |
| `_current_lora_name` 他 | いまアタッチ中のファイル名・strength |
| `_ADAPTER_TYPE_NAMES` | `"lora"` 等をファイル名と誤認しないための拒否集合 |

`_lora_line` は **print のみ**（print + logger の二重出力を避ける）。Status 本体はこれで出す。

#### 4.1.2 `decode_comfy_quant_conf` / `checkpoint_looks_like_comfy_quant_int8`

- マーカーが tensor/bytes/str/二重 JSON でも dict に落とす。
- パス指定時は safetensors を軽く開き、`int8_tensorwise` または int8 weight + comfy_quant を検出。フルロード前に HSWQLoader が分岐できる。

#### 4.1.3 `_patch_convert_old_quants`

本体 `convert_old_quants` の前後で metadata `layers` の裸文字列を `{"format": ...}` に直し、`.comfy_quant` テンソルを正規化。エクスポータ差で落ちるロードを防ぐ。

#### 4.1.4 `_make_quantized_conv2d` / MixedPrecisionOps.Conv2d

Linear 側と同じ思想:

- `_load_from_state_dict` → `_load_quantized_module`
- `convert_weight` — 量子化 weight を計算用 float に戻す（LoRA 加算の前）
- `set_weight` — bake 後に再量子化して戻す（**requant 経路**）

これがないと ModelPatcher は int8 に直接丸め、**Conv 上の LoRA が死ぬ**。

#### 4.1.5 LoRA 名キャプチャ（ノード非依存）

| フック | 狙い |
|--------|------|
| `folder_paths.get_full_path(_or_raise)` folder=`loras` | UI 相対名 |
| `comfy.utils.load_torch_file`（loras 配下） | フルパスから basename |
| `nodes.LoraLoader.load_lora` | 標準ノード |
| `inspect.stack` ローカル探索 | Power LoRA / 辞書ウィジェット等 |

`_looks_like_lora_filename` でアダプタ型名を弾く（かつての `lora名=lora` バグ対策）。

#### 4.1.6 `_patch_load_lora_key_counts`

`comfy.lora.load_lora` と `comfy.sd.load_lora_for_models` をラップし、

- ファイルキー数、マップ数
- unet/clip 適用キー
- `not_mapped` / `mapped_but_not_attached`

を履歴に積み、`_log_lora_slot_attach` で即 Status 1 行を出す。

#### 4.1.7 Dynamic bake（`_bake_int8_patches_on_dynamic_patcher`）

Dynamic VRAM では LoRA が `LowVramPatch` のまま残り、INT8 `set_weight` が走らず「キー数は付いたが見た目に効かない」ことがある。`ModelPatcherDynamic.load` の直後に:

1. baked 済みキーの LowVram を剥がす
2. `set_func` があるキーだけ `patch_weight_to_device`（requant）
3. `_v` は消さない（VBAR OOM 防止）
4. `patches` / 該当 `backup` を整理

`dump_int8_lora_bake_stats` で bake サマリ（OK_requant / BROKEN_int8_round 等）。

#### 4.1.8 `summarize_int8_lora_capability`

ロード直後に diffusion_model を走査し、Linear/Conv2d の `set_weight`/`convert_weight` 有無を報告。Conv 欠落時は WARN。

### 4.2 `__init__.py` — 意味

拡張ロード時に必ず INT8 パッチを適用する。失敗しても import 全体は落とさない（他ノードを生かす）。HSWQLoader 登録ログを「FP8/INT8」に更新し、ユーザーに両対応であることを示す。

### 4.3 `nodes/hswq_loader_node.py` — 意味

従来の Scaled FP8（`.weight` + `.scale`）専用だった HSWQLoader に、

1. safetensors プローブで INT8 comfy_quant を検出
2. 検出時は本体 `load_checkpoint_guess_config`（パッチ済み MixedPrecisionOps）
3. 非 INT8 は従来 FP8 経路

という分岐を入れた。タイトルを `HSWQ FP8/INT8 Loader (VRAM Opt)` にし、UI 上でも両対応と分かるようにした。INT8 成功時は LoRA カウンタ reset + capability summary。

### 4.4 `hswq/zimage_fp8_e4m3_unet.py` — 意味

UNet 専用ローダ側でも同じ INT8 を使えるようにする。

- `weight_dtype` 選択肢に `int8_tensorwise`
- 自動検出または明示 dtype で INT8 経路
- INT8 テンソルに float8 を強制しない（破壊防止）

HSWQLoader と役割が重なるが、ワークフローで UNet 単体ロードを使う場合の入口。

### 4.5 `patches/zimage_fp8_torchcompile.py` — 意味

LoRA 適用時の shape/numel 不一致スキップを、`record_lora_shape_skip(lora_name, key, reason)` に流す。Status の `shape_skip` 集計と bake サマリに載せるため。スキップを黙殺すると「適用キー数」と実効のズレが追えない。

---

## 付録 A — 処理フロー（ロード）

```
チェックポイント選択
    │
    ├─ probe: int8_tensorwise / comfy_quant ?
    │         YES → load_checkpoint_guess_config
    │               └─ convert_old_quants（正規化）
    │               └─ MixedPrecisionOps.Linear + 注入 Conv2d
    │               └─ summarize_int8_lora_capability
    │
    └─ NO  → 従来 Scaled FP8（.scale dequant）経路
```

## 付録 B — 処理フロー（LoRA）

```
任意 LoRA ノード
    │
    ├─ folder_paths / load_torch_file / LoraLoader → ファイル名キャプチャ
    ├─ load_lora_for_models → 適用/スキップ集計 → Status（attach）
    └─ サンプリング時 ModelPatcherDynamic.load
          ├─ _strip_lowvram_for_baked_keys
          ├─ _bake_int8_patches_on_dynamic_patcher（_v 保持）
          └─ dump_int8_lora_bake_stats
```

## 付録 C — コンソールで見るべき行

| プレフィックス | 内容 |
|----------------|------|
| `[HSWQ INT8 LoRA] ===== load summary =====` | set/convert 準備 |
| `[HSWQ LoRA Status] Slot N: lora名=...` | アタッチ時 Status |
| `[HSWQ LoRA Status] ===== bake summary` | bake 後サマリ |
| `[HSWQ LoRA Bake] path OK (all requant)` | 健全 |
| `BROKEN_int8_round` / `int8_round` WARN | その層の LoRA は壊れている |

## 付録 D — 再発防止（実装者向け）

1. bake 後に `module._v` を delete しない（VBAR bump allocator）。
2. Status の `lora名` にアダプタ型文字列を入れない。
3. Conv2d に `set_weight`/`convert_weight` 無しで「LoRA 対応完了」と言わない。
4. パッチ版番号を上げたら再起動必須。
5. 本体 `ops.py` へ恒久パッチを戻さない（本拡張の存在理由）。

---

（本解説書はコミット `{COMMIT[:7]}` 時点のソース全文を埋め込んだ監査用一次資料である。）
