# INT8 Triton W8A8 Acceleration — Complete Technical Guide

> **Status (v3.2.7+):** This extension **no longer ships** its own INT8 Triton W8A8 Linear path.
> Acceleration is delegated to **ComfyUI + `comfy_kitchen`** (`int8_linear`: cuda → triton → eager).
> The sections below are a historical record of the removed Plan B implementation (v3.2.6).

Date: 2026-07-18  
Repository: `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools`  
Canonical commit (added): `bf8a8a6` (`feat: add INT8 W8A8 Triton accelerate with tiled rowwise quant`)  
Removed in: **v3.2.7**  
Related plan: `md/INT8_TRITON_W8A8_PUBLIC_ACCELERATION_PLAN.md`

This guide originally recorded the **shipped** INT8 Linear (W8A8) Triton acceleration in this custom node. That path has been removed; keep this file only as implementation history.

---

## 1. Confirmed overview

### 1.1 What this feature is

HSWQ `int8_tensorwise` loads use a **fused Triton INT8 Linear** path shipped in this extension:

1. Row-wise activation quantization  
2. INT8 GEMM  
3. Fused dequant (± bias)

Algorithm class matches INT8-Fast / forge kitchen fused Linear. Soft fallbacks when the fused path does not run: eager / `torch._int_mm` / `F.linear`.

### 1.2 Confirmed delivery surface

| Item | Confirmed fact |
|------|----------------|
| Runtime package | Triton installed by this repo’s `install.py` (Windows → `triton-windows<3.7`; Linux → `triton`; macOS → skipped) |
| Entry | MixedPrecisionOps `Linear._forward` wrap in `patches/comfy_quant_int8.py` (ops patch **v2**) |
| UI | BOOLEAN `triton_accelerate` (default **True**) + JS toggle label **Triton accelerate** |
| Stamp | `model.model_options["hswq_triton_accelerate"]` and per-module `_hswq_triton_accelerate` on `int8_tensorwise` Linears |
| Success log | `[HSWQ INT8 Triton] fused W8A8 Linear path active (Plan B)` |
| Wide-K layers | Tiled rowwise quant supports **K=10240** (soft cap default 131072); the old hard 8192 column raise does not apply |

### 1.3 Architecture (this repo)

```text
Loader UI (triton_accelerate BOOLEAN + JS toggle)
        |
        v
Stamp model_options["hswq_triton_accelerate"]
  + per-module _hswq_triton_accelerate on int8_tensorwise Linears
        |
        v
mixed_precision_ops(...) wraps Linear._forward  (ops patch v2)
        |
        v
_try_triton_int8_linear(...)
  gates: format / toggle / QuantizedTensor / no convrot / SM>=8.0 / M>16
        |
        v
patches/int8_triton_kernels.fused_int8_linear
        |
        v
patches/_int8_triton_kernels_impl.py
  1) tiled triton_quantize_rowwise (supports K>8192, e.g. 10240)
  2) INT8 matmul + dequant (scalar or per-row weight scale)
```

### 1.4 Forward decision order (Linear)

1. Not `int8_tensorwise` QuantizedTensor Linear → untouched.
2. Toggle OFF / Triton missing / SM too low / convrot / transposed → fall through.
3. `M ≤ 16` → dequant + `F.linear` (avoid launch overhead).
4. Else → fused Triton; on exception → log once, fall through to original `_forward`.

### 1.5 Install path

`install.py` installs Triton after `requirements.txt`:

- Windows: `triton-windows<3.7` (uninstall stock `triton` first)
- Linux: `triton`
- macOS: skip (CUDA required)
- Probe: `import triton` → READY / UNAVAILABLE

### 1.6 Wide-K fix (confirmed)

Z-Image INT8 FF layers use activation width **K=10240**. An earlier hard cap of 8192 columns aborted those launches. The shipped kernel uses **tiled** `tl.range` rowwise quant (tile default 1024, soft cap 131072). Confirmed runtime: fused path active with **no** `only supports <= 8192 … got 10240` error.

### 1.7 Confirmed scope boundaries

- No global `F.linear` monkeypatch
- No Triton attention rewrite (FA2 / SA2 / SA3 remain separate)
- No Nunchaku SVDQ arming via false-positive `"nunchaku" in __module__`

### 1.8 Target nodes only

| Node class | Display |
|------------|---------|
| `HSWQFP8E4M3UNetLoader` | HSWQ FP8 E4M3/INT8 UNet Loader |
| `NunchakuUssoewwinCheckpointLoaderSDXL` | HSWQ Checkpoint Loader (SDXL) |

---

## 2. Added and modified file names

| Path | Change | Role |
|------|--------|------|
| `patches/_int8_triton_kernels_impl.py` | **Added** | Triton JIT kernels + Python wrappers (quantize, GEMM scalar, GEMM per-row) |
| `patches/int8_triton_kernels.py` | **Added** | Soft-import API: `is_triton_int8_available`, `fused_int8_linear`, SM check |
| `js/hswq_triton_accelerate_toggle.js` | **Added** | Force BOOLEAN widget to UI toggle labeled "Triton accelerate" |
| `md/INT8_TRITON_W8A8_PUBLIC_ACCELERATION_PLAN.md` | **Added** | Implementation plan (design record) |
| `md/INT8_TRITON_W8A8_ACCELERATION_GUIDE.md` | **Added** | This complete guide |
| `patches/comfy_quant_int8.py` | **Modified** | Linear `_forward` wrap, stamp toggle, ops patch v2, loader kwargs |
| `install.py` | **Modified** | Triton install stage for public speed |
| `__init__.py` | **Modified** | SDXL loader `triton_accelerate` INPUT + signature |
| `hswq/zimage_fp8_e4m3_unet.py` | **Modified** | UNet loader `triton_accelerate` INPUT + signature |

---

## 3. Full source of added / modified code

New modules are reproduced **in full** below. Modified modules reproduce the **complete Triton-related regions** as they exist in the tree after `bf8a8a6` (including the tiled quant fix for K=10240).

### 3.1 `patches/int8_triton_kernels.py` (full file)

```python
"""Public INT8 Triton W8A8 wrappers (Plan B).

Soft-imports the fused kernel implementation. Missing Triton must never crash
graph load — callers check ``is_triton_int8_available()`` and fall back.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

LOG_PREFIX = "[HSWQ INT8 Triton]"

_AVAILABLE = False
_IMPORT_ERROR: Optional[BaseException] = None
_LOGGED_UNAVAILABLE = False

triton_quantize_rowwise = None
triton_int8_linear = None
triton_int8_linear_per_row = None

try:
    from ._int8_triton_kernels_impl import (  # noqa: F401
        triton_int8_linear as _triton_int8_linear,
        triton_int8_linear_per_row as _triton_int8_linear_per_row,
        triton_quantize_rowwise as _triton_quantize_rowwise,
    )

    triton_quantize_rowwise = _triton_quantize_rowwise
    triton_int8_linear = _triton_int8_linear
    triton_int8_linear_per_row = _triton_int8_linear_per_row
    _AVAILABLE = True
except Exception as exc:  # noqa: BLE001 — soft disable for public installs
    _IMPORT_ERROR = exc
    _AVAILABLE = False


def is_triton_int8_available() -> bool:
    """True when Triton imported and fused INT8 kernels are loadable."""
    global _LOGGED_UNAVAILABLE
    if _AVAILABLE:
        return True
    if not _LOGGED_UNAVAILABLE:
        _LOGGED_UNAVAILABLE = True
        logging.warning(
            "%s UNAVAILABLE — falling back to eager/_int_mm (%s)",
            LOG_PREFIX,
            _IMPORT_ERROR,
        )
    return False


def triton_import_error() -> Optional[BaseException]:
    return _IMPORT_ERROR


def cuda_sm_ok_for_triton_int8(min_sm: tuple[int, int] = (8, 0)) -> bool:
    """Ampere+ by default (plan success criterion). Soft-check only."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() >= min_sm
    except Exception:
        return False


def fused_int8_linear(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dispatch scalar vs per-row weight scale fused Triton INT8 Linear.

    Caller must enforce ``M > 16`` and accelerate toggle / availability.
    """
    if not _AVAILABLE or triton_int8_linear is None or triton_int8_linear_per_row is None:
        raise RuntimeError(f"{LOG_PREFIX} kernels not available: {_IMPORT_ERROR}")
    if compute_dtype is None:
        compute_dtype = x.dtype
    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([float(weight_scale)], device=x.device, dtype=torch.float32)
    if weight_scale.numel() == 1:
        return triton_int8_linear(x, weight_q, weight_scale, bias, compute_dtype=compute_dtype)
    return triton_int8_linear_per_row(x, weight_q, weight_scale, bias, compute_dtype=compute_dtype)
```

### 3.2 `js/hswq_triton_accelerate_toggle.js` (full file)

```javascript
import { app } from "../../scripts/app.js";

/** Same pattern as sdxl_lora_dynamic_v3.js: BOOLEAN backend + widget.type = "toggle". */
const NODE_IDS = new Set([
    "HSWQFP8E4M3UNetLoader",
    "NunchakuUssoewwinCheckpointLoaderSDXL",
]);
const WIDGET_NAME = "triton_accelerate";
const WIDGET_LABEL = "Triton accelerate";

function forceTritonToggle(node) {
    if (!node || !NODE_IDS.has(node.comfyClass)) return;
    const widgets = node.widgets || [];
    const w = widgets.find((x) => x.name === WIDGET_NAME);
    if (!w) return;
    w.type = "toggle";
    w.label = WIDGET_LABEL;
    if (w.computeSize) delete w.computeSize;
}

app.registerExtension({
    name: "nunchaku_ussoewwin.hswq_triton_accelerate_toggle",

    nodeCreated(node) {
        forceTritonToggle(node);
    },

    loadedGraphNode(node) {
        forceTritonToggle(node);
    },
});
```

### 3.3 `install.py` (full file after Triton stage)

```python
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")

# Windows public wheels (woct0rdho / triton-lang). Upper bound avoids pulling a
# Triton that breaks common torch builds; adjust when torch support matrix moves.
_WINDOWS_TRITON_SPEC = "triton-windows<3.7"
_LINUX_TRITON_SPEC = "triton"


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    print("[nunchaku-unofficial-loader][install] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _pip_quiet_uninstall(package):
    """Best-effort uninstall; missing package is not an error."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        package,
    ]
    print("[nunchaku-unofficial-loader][install] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _probe_triton_import():
    """Return True if `import triton` works under this interpreter."""
    code = (
        "try:\n"
        "    import triton  # noqa: F401\n"
        "except Exception:\n"
        "    raise SystemExit(1)\n"
        "raise SystemExit(0)\n"
    )
    rc = subprocess.call([sys.executable, "-c", code])
    return rc == 0


def _cuda_likely_available():
    """True if torch reports CUDA; False if no CUDA; None if torch missing."""
    code = (
        "try:\n"
        "    import torch\n"
        "    raise SystemExit(0 if torch.cuda.is_available() else 2)\n"
        "except Exception:\n"
        "    raise SystemExit(1)\n"
    )
    rc = subprocess.call([sys.executable, "-c", code])
    if rc == 0:
        return True
    if rc == 2:
        return False
    # torch not importable yet — still try Triton on Windows/Linux for GPU users
    # who install torch later.
    return None


def _install_triton_for_int8_speed():
    """
    Install a Triton runtime so Plan B INT8 fused Linear kernels can run.

    Windows needs triton-windows (stock triton wheels are not usable natively).
    Linux uses the standard triton package (often already pulled by torch).
    """
    system = platform.system()
    print(
        "[nunchaku-unofficial-loader][install] --- INT8 Triton speed environment ---",
        flush=True,
    )

    if system == "Darwin":
        print(
            "[nunchaku-unofficial-loader][install] macOS: skipping Triton "
            "(INT8 Triton acceleration requires NVIDIA CUDA).",
            flush=True,
        )
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: UNAVAILABLE",
            flush=True,
        )
        return

    cuda = _cuda_likely_available()
    if cuda is False:
        print(
            "[nunchaku-unofficial-loader][install] torch.cuda.is_available() is False; "
            "still attempting Triton install for users who switch to a CUDA torch later.",
            flush=True,
        )

    if _probe_triton_import() and system != "Windows":
        print(
            "[nunchaku-unofficial-loader][install] triton already importable.",
            flush=True,
        )
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: READY",
            flush=True,
        )
        return

    if system == "Windows":
        # Stock Linux `triton` on Windows is a common footgun.
        _pip_quiet_uninstall("triton")
        rc = _pip("install", "-U", _WINDOWS_TRITON_SPEC)
        if rc != 0:
            print(
                "[nunchaku-unofficial-loader][install] ERROR: failed to install %s "
                "(exit %d)." % (_WINDOWS_TRITON_SPEC, rc),
                flush=True,
            )
            print(
                "[nunchaku-unofficial-loader][install] Remediation (same python as ComfyUI):\n"
                '  "%s" -m pip install -U "%s"'
                % (sys.executable, _WINDOWS_TRITON_SPEC),
                flush=True,
            )
        else:
            print(
                "[nunchaku-unofficial-loader][install] installed %s"
                % _WINDOWS_TRITON_SPEC,
                flush=True,
            )
    else:
        # Linux (and other non-Darwin UNIX): standard triton.
        if not _probe_triton_import():
            rc = _pip("install", "-U", _LINUX_TRITON_SPEC)
            if rc != 0:
                print(
                    "[nunchaku-unofficial-loader][install] ERROR: failed to install %s "
                    "(exit %d)." % (_LINUX_TRITON_SPEC, rc),
                    flush=True,
                )
                print(
                    "[nunchaku-unofficial-loader][install] Remediation:\n"
                    '  "%s" -m pip install -U %s'
                    % (sys.executable, _LINUX_TRITON_SPEC),
                    flush=True,
                )
            else:
                print(
                    "[nunchaku-unofficial-loader][install] installed %s"
                    % _LINUX_TRITON_SPEC,
                    flush=True,
                )

    if _probe_triton_import():
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: READY",
            flush=True,
        )
    else:
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: UNAVAILABLE "
            "— INT8 will fall back to eager/_int_mm until Triton imports successfully.",
            flush=True,
        )
        if system == "Windows":
            print(
                "[nunchaku-unofficial-loader][install] Windows tips: use this exact "
                "python (-m pip), not a different pip.exe; portable embeds may need "
                "matching Python include/libs for Triton runtime.",
                flush=True,
            )


def main():
    # Python 3.12 removed pkgutil.ImpImporter. Some transitive source builds
    # (e.g. facexlib -> filterpy, which has no wheel) fail with
    # "AttributeError: module 'pkgutil' has no attribute 'ImpImporter'"
    # when the environment ships an old setuptools. Upgrade build tooling
    # first so those legacy sdist builds succeed.
    _pip("install", "-U", "pip", "setuptools", "wheel")

    if os.path.isfile(REQUIREMENTS):
        rc = _pip("install", "-r", REQUIREMENTS)
        if rc != 0:
            print(
                "[nunchaku-unofficial-loader][install] requirements install "
                "returned code %d" % rc,
                flush=True,
            )
            sys.exit(rc)
    else:
        print(
            "[nunchaku-unofficial-loader][install] requirements.txt not found: %s"
            % REQUIREMENTS,
            flush=True,
        )

    # After base deps: install Triton so Plan B INT8 kernels can deliver speed
    # for public users (does not rely on Comfy --enable-triton-backend).
    _install_triton_for_int8_speed()


if __name__ == "__main__":
    main()
```

### 3.4 `patches/_int8_triton_kernels_impl.py` (full file)

```python
import os
import time
import torch

# =============================================================================
# Runtime Kernel Configuration
# =============================================================================

LOG_PREFIX = "[HSWQ INT8 Triton]"


def _read_env_int(name: str, default_value: int) -> int:
	raw_value = os.environ.get(name)
	if raw_value is None:
		return default_value
	try:
		return int(raw_value)
	except ValueError:
		print(f"{LOG_PREFIX} Invalid {name}={raw_value!r}; using {default_value}.")
		return default_value


_ENABLE_TRITON_AUTOTUNE = os.environ.get("INT8_TRITON_AUTOTUNE", "0") == "1"
# Soft upper bound only (tiled kernel supports wide K; e.g. Z-Image FF K=10240).
# Set INT8_TRITON_ROWWISE_QUANT_MAX_COLS=0 to disable the cap.
TRITON_ROWWISE_QUANT_MAX_COLS = max(0, _read_env_int("INT8_TRITON_ROWWISE_QUANT_MAX_COLS", 131072))
TRITON_ROWWISE_QUANT_TILE = max(128, _read_env_int("INT8_TRITON_ROWWISE_QUANT_TILE", 1024))

_FIXED_KERNEL_CONFIG = {
	"BLOCK_M": max(16, _read_env_int("INT8_TRITON_BLOCK_M", 128)),
	"BLOCK_N": max(16, _read_env_int("INT8_TRITON_BLOCK_N", 128)),
	"BLOCK_K": max(16, _read_env_int("INT8_TRITON_BLOCK_K", 64)),
	"GROUP_SIZE_M": max(1, _read_env_int("INT8_TRITON_GROUP_SIZE_M", 8)),
	"num_warps": max(1, _read_env_int("INT8_TRITON_NUM_WARPS", 4)),
	"num_stages": max(1, _read_env_int("INT8_TRITON_NUM_STAGES", 4)),
}

import triton
import triton.language as tl
from triton.language.extra import libdevice

_AUTOTUNE_CONFIGS = [
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
	triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
	triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
]

_KERNEL_CONFIG_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M", "num_warps", "num_stages")


def _sanitize_kernel_config(config: dict) -> dict:
	return {
		"BLOCK_M": max(16, int(config["BLOCK_M"])),
		"BLOCK_N": max(16, int(config["BLOCK_N"])),
		"BLOCK_K": max(16, int(config["BLOCK_K"])),
		"GROUP_SIZE_M": max(1, int(config["GROUP_SIZE_M"])),
		"num_warps": max(1, int(config["num_warps"])),
		"num_stages": max(1, int(config["num_stages"])),
	}


def get_fixed_kernel_config() -> dict:
	return dict(_FIXED_KERNEL_CONFIG)


def is_fixed_kernel_mode() -> bool:
	return not _ENABLE_TRITON_AUTOTUNE


def set_fixed_kernel_config(config: dict, source: str = "runtime", silent: bool = False) -> dict:
	global _FIXED_KERNEL_CONFIG
	try:
		merged = dict(_FIXED_KERNEL_CONFIG)
		for key in _KERNEL_CONFIG_KEYS:
			if key in config:
				merged[key] = config[key]
		_FIXED_KERNEL_CONFIG = _sanitize_kernel_config(merged)
		if not silent:
			print(f"{LOG_PREFIX} Applied INT8 Triton config from {source}: {_FIXED_KERNEL_CONFIG}")
	except Exception as e:
		if not silent:
			print(f"{LOG_PREFIX} Failed to apply kernel config from {source}: {e}")
	return dict(_FIXED_KERNEL_CONFIG)


def format_kernel_config_env_lines(config: dict) -> list[str]:
	cfg = _sanitize_kernel_config(config)
	return [
		f"INT8_TRITON_BLOCK_M={cfg['BLOCK_M']}",
		f"INT8_TRITON_BLOCK_N={cfg['BLOCK_N']}",
		f"INT8_TRITON_BLOCK_K={cfg['BLOCK_K']}",
		f"INT8_TRITON_GROUP_SIZE_M={cfg['GROUP_SIZE_M']}",
		f"INT8_TRITON_NUM_WARPS={cfg['num_warps']}",
		f"INT8_TRITON_NUM_STAGES={cfg['num_stages']}",
	]


def get_candidate_kernel_configs(extra_candidates: list[dict] | None = None, include_current: bool = True) -> list[dict]:
	candidates = []
	seen = set()

	def _add(cfg):
		try:
			sanitized = _sanitize_kernel_config(cfg)
		except Exception:
			return
		fingerprint = tuple(sanitized[key] for key in _KERNEL_CONFIG_KEYS)
		if fingerprint in seen:
			return
		seen.add(fingerprint)
		candidates.append(sanitized)

	if include_current:
		_add(_FIXED_KERNEL_CONFIG)

	for cfg in _AUTOTUNE_CONFIGS:
		kwargs = getattr(cfg, "kwargs", None)
		if kwargs is None:
			continue
		_add({
			"BLOCK_M": kwargs.get("BLOCK_M", _FIXED_KERNEL_CONFIG["BLOCK_M"]),
			"BLOCK_N": kwargs.get("BLOCK_N", _FIXED_KERNEL_CONFIG["BLOCK_N"]),
			"BLOCK_K": kwargs.get("BLOCK_K", _FIXED_KERNEL_CONFIG["BLOCK_K"]),
			"GROUP_SIZE_M": kwargs.get("GROUP_SIZE_M", _FIXED_KERNEL_CONFIG["GROUP_SIZE_M"]),
			"num_warps": getattr(cfg, "num_warps", _FIXED_KERNEL_CONFIG["num_warps"]),
			"num_stages": getattr(cfg, "num_stages", _FIXED_KERNEL_CONFIG["num_stages"]),
		})

	if extra_candidates:
		for cfg in extra_candidates:
			if isinstance(cfg, dict):
				_add(cfg)

	return candidates


@torch.no_grad()
def microbench_fixed_kernel_configs(
	m: int = 2048,
	k: int = 4096,
	n: int = 4096,
	warmup: int = 2,
	iterations: int = 6,
	include_scalar: bool = False,
	extra_candidates: list[dict] | None = None,
):
	if not is_fixed_kernel_mode():
		raise RuntimeError("INT8_TRITON_AUTOTUNE=1 is active; fixed kernel config microbench is unavailable.")

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for Triton kernel microbench.")

	m = max(64, int(m))
	k = max(64, int(k))
	n = max(64, int(n))
	warmup = max(1, int(warmup))
	iterations = max(2, int(iterations))

	original_config = get_fixed_kernel_config()
	candidates = get_candidate_kernel_configs(extra_candidates=extra_candidates, include_current=True)
	if not candidates:
		raise RuntimeError("No Triton kernel configs available for benchmarking.")

	device = torch.device("cuda")
	x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
	weight = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
	w_scale_row = torch.rand((n, 1), device=device, dtype=torch.float32).mul_(0.02).add_(0.005)
	w_scale_scalar = torch.tensor([0.01], device=device, dtype=torch.float32)
	bias = torch.randn((n,), device=device, dtype=torch.float32)

	results = []
	try:
		for cfg in candidates:
			set_fixed_kernel_config(cfg, source="microbench", silent=True)

			for _ in range(warmup):
				_ = triton_int8_linear_per_row(x, weight, w_scale_row, bias=bias, compute_dtype=torch.bfloat16)
				if include_scalar:
					_ = triton_int8_linear(x, weight, w_scale_scalar, bias=bias, compute_dtype=torch.bfloat16)
			torch.cuda.synchronize()

			start = time.perf_counter()
			for _ in range(iterations):
				_ = triton_int8_linear_per_row(x, weight, w_scale_row, bias=bias, compute_dtype=torch.bfloat16)
				if include_scalar:
					_ = triton_int8_linear(x, weight, w_scale_scalar, bias=bias, compute_dtype=torch.bfloat16)
			torch.cuda.synchronize()

			elapsed_ms = (time.perf_counter() - start) * 1000.0
			avg_ms = elapsed_ms / iterations
			results.append({
				"config": dict(cfg),
				"avg_ms": avg_ms,
			})
	finally:
		set_fixed_kernel_config(original_config, source="restore", silent=True)

	results.sort(key=lambda item: item["avg_ms"])
	best = results[0]["config"]
	return best, results

if _ENABLE_TRITON_AUTOTUNE:
	print(f"{LOG_PREFIX} Triton autotune is enabled (INT8_TRITON_AUTOTUNE=1).")
else:
	print(
		f"{LOG_PREFIX} Triton autotune is disabled; using fixed INT8 kernel config "
		f"{_FIXED_KERNEL_CONFIG}."
	)


def _kernel_strategy_decorator():
	if _ENABLE_TRITON_AUTOTUNE:
		return triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"], cache_results=True)

	def _identity(kernel):
		return kernel

	return _identity


_KERNEL_STRATEGY_DECORATOR = _kernel_strategy_decorator()

# =============================================================================
# Kernel 1: Fused Row-wise Quantization (FP16/BF16 -> INT8 + Scale)
# =============================================================================


@triton.jit
def _quantize_rowwise_kernel(
    x_ptr,  # Input pointer (FP16/BF16)
    y_ptr,  # Output pointer (INT8)
    s_ptr,  # Scale pointer (FP32)
    n_elements,  # Number of columns
    BLOCK_SIZE: tl.constexpr,
):
	"""Tiled row-wise INT8 quant: supports K larger than one power-of-2 block (e.g. 10240)."""
	row_idx = tl.program_id(0)
	x_row_ptr = x_ptr + row_idx * n_elements
	y_row_ptr = y_ptr + row_idx * n_elements

	# Pass 1: row absmax across tiles
	max_val = 0.0
	for start in tl.range(0, n_elements, BLOCK_SIZE):
		offsets = start + tl.arange(0, BLOCK_SIZE)
		mask = offsets < n_elements
		x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
		max_val = tl.maximum(max_val, tl.max(tl.abs(x), axis=0))

	scale = tl.maximum(max_val / 127.0, 1e-30)

	# Pass 2: quantize + store
	for start in tl.range(0, n_elements, BLOCK_SIZE):
		offsets = start + tl.arange(0, BLOCK_SIZE)
		mask = offsets < n_elements
		x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
		q_f = x / scale
		q_i = libdevice.rint(q_f).to(tl.int32)
		q_i = tl.clamp(q_i, -128.0, 127.0)
		tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)

	tl.store(s_ptr + row_idx, scale.to(tl.float32))


def _torch_quantize_rowwise(x: torch.Tensor):
	"""Host/GPU torch fallback when Triton quant cannot run."""
	abs_max = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
	scale = abs_max / 127.0
	q = torch.clamp(torch.round(x.float() / scale), -128, 127).to(torch.int8)
	return q, scale.to(dtype=torch.float32)


def triton_quantize_rowwise(x: torch.Tensor):
	"""
    Input: [Batch, Dim] (float16/bfloat16/float32)
    Output: [Batch, Dim] (int8), [Batch, 1] (float32)

    Uses a tiled Triton kernel so wide activations (K > 8192, e.g. 10240) work.
    Falls back to torch row-wise quant on Triton failure (GEMM path can still use Triton).
    """
	if not x.is_contiguous():
		x = x.contiguous()

	rows, cols = x.shape
	if TRITON_ROWWISE_QUANT_MAX_COLS > 0 and cols > TRITON_ROWWISE_QUANT_MAX_COLS:
		return _torch_quantize_rowwise(x)

	y = torch.empty_like(x, dtype=torch.int8)
	s = torch.empty((rows, 1), device=x.device, dtype=torch.float32)

	# Fixed tile size: one power-of-2 block, looped over columns.
	BLOCK_SIZE = triton.next_power_of_2(min(cols, TRITON_ROWWISE_QUANT_TILE))
	if BLOCK_SIZE < 128:
		BLOCK_SIZE = 128

	grid = (rows, )
	try:
		_quantize_rowwise_kernel[grid](x, y, s, cols, BLOCK_SIZE=BLOCK_SIZE)
		return y, s
	except Exception as e:
		print(f"{LOG_PREFIX} tiled rowwise quant failed (K={cols}); torch fallback: {e}")
		return _torch_quantize_rowwise(x)


# =============================================================================
# Kernel 2: INT8 GEMM + Fused Dequantization Epilogue
# =============================================================================


@_KERNEL_STRATEGY_DECORATOR
@triton.jit
def _int8_matmul_dequant_kernel(
        # Pointers
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        bias_ptr,
        # Matrix Dimensions
        M,
        N,
        K,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_MN_TAIL: tl.constexpr,
        LEGACY_UNSAFE: tl.constexpr):
	"""
    Computes: C = ((A * B) * (scale_a * scale_b)) + bias
    A: [M, K] int8
    B: [N, K] int8 (Transposed physically or logically via strides)
    """
	pid = tl.program_id(axis=0)
	num_pid_m = tl.cdiv(M, BLOCK_M)
	num_pid_n = tl.cdiv(N, BLOCK_N)
	num_pid_in_group = GROUP_SIZE_M * num_pid_n
	group_id = pid // num_pid_in_group
	first_pid_m = group_id * GROUP_SIZE_M
	group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
	pid_m = first_pid_m + (pid % group_size_m)
	pid_n = (pid % num_pid_in_group) // group_size_m

	# 1. Prepare Pointers for A and B
	# A block pointer: [BLOCK_M, BLOCK_K]
	if LEGACY_UNSAFE:
		offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
		offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
	else:
		offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
		offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_k = tl.arange(0, BLOCK_K)

	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
	b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

	# 2. Main Loop (Accumulate in Int32)
	accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

	for k in range(0, tl.cdiv(K, BLOCK_K)):
		# Load chunks
		k_mask = offs_k < K - k * BLOCK_K
		if LEGACY_UNSAFE:
			a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
			b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
		elif HAS_MN_TAIL:
			a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
			b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
		else:
			a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
			b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

		# Matrix Multiply (Int8 inputs -> Int32 accum)
		accumulator += tl.dot(a, b)

		# Advance pointers
		a_ptrs += BLOCK_K * stride_ak
		b_ptrs += BLOCK_K * stride_bk

	# 3. Fused Epilogue (Dequantize & Bias)

	# Load dynamic scales
	# A Scale is per-row [M, 1]
	if LEGACY_UNSAFE:
		scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]
	elif HAS_MN_TAIL:
		scale_a = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=0.0)  # Vector [BLOCK_M]
	else:
		scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]

	# B Scale is scalar or tensor.
	scale_b = tl.load(b_scale_ptr)

	# Convert Accumulator to Float
	c = accumulator.to(tl.float32)

	# Combine scales: scale_a (broadcast columns) * scale_b
	total_scale = scale_a[:, None] * scale_b

	c = c * total_scale

	# Add Bias if present
	if HAS_BIAS:
		if LEGACY_UNSAFE:
			bias = tl.load(bias_ptr + offs_bn)  # Vector [BLOCK_N]
		elif HAS_MN_TAIL:
			bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)  # Vector [BLOCK_N]
		else:
			bias = tl.load(bias_ptr + offs_bn)  # Vector [BLOCK_N]
		c = c + bias[None, :]

	# 4. Store Result (Cast to output dtype, usually FP16)
	c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
	if LEGACY_UNSAFE:
		c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
	elif HAS_MN_TAIL:
		c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
	else:
		c_mask = True

	# We write as fp16 or bf16 implicitly by the pointer type, but explicit cast is safer
	tl.store(c_ptrs, c, mask=c_mask)


# =============================================================================
# Python Wrapper
# =============================================================================


def triton_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale, bias=None, compute_dtype=torch.float16, weight_is_prepacked: bool = False, legacy_unsafe: bool = False):
	"""
    Fused pipeline for W8A8 Linear Layer.
    """
	# 1. Flatten inputs if 3D [Batch, Tokens, Dim] -> [Batch*Tokens, Dim]
	x_shape_orig = x.shape
	x_2d = x.reshape(-1, x_shape_orig[-1])

	M, K = x_2d.shape
	N = weight.shape[1] if weight_is_prepacked else weight.shape[0]

	# 2. Kernel 1: Dynamic Activation Quantization
	#    (This is much faster than Python-loop based axiswise quant)
	x_int8, x_scale = triton_quantize_rowwise(x_2d)

	# 3. Allocate Output
	output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

	# 4. Prepare Scales for Kernel
	# Ensure weight_scale is a tensor on device
	if not isinstance(weight_scale, torch.Tensor):
		weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
	elif weight_scale.numel() == 1:
		weight_scale = weight_scale.reshape(1)

	# 5. Kernel 2: Fused GEMM + Dequant
	grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

	# Check if we have bias
	has_bias = bias is not None
	bias_ptr = bias if has_bias else x  # Dummy pointer if None
	launch_kwargs = {"HAS_BIAS": has_bias, "HAS_MN_TAIL": True, "LEGACY_UNSAFE": bool(legacy_unsafe)}
	if not _ENABLE_TRITON_AUTOTUNE:
		has_mn_tail = (M % _FIXED_KERNEL_CONFIG["BLOCK_M"]) != 0 or (N % _FIXED_KERNEL_CONFIG["BLOCK_N"]) != 0
		launch_kwargs.update({
			"BLOCK_M": _FIXED_KERNEL_CONFIG["BLOCK_M"],
			"BLOCK_N": _FIXED_KERNEL_CONFIG["BLOCK_N"],
			"BLOCK_K": _FIXED_KERNEL_CONFIG["BLOCK_K"],
			"GROUP_SIZE_M": _FIXED_KERNEL_CONFIG["GROUP_SIZE_M"],
			"HAS_MN_TAIL": has_mn_tail,
			"LEGACY_UNSAFE": bool(legacy_unsafe),
			"num_warps": _FIXED_KERNEL_CONFIG["num_warps"],
			"num_stages": _FIXED_KERNEL_CONFIG["num_stages"],
		})

	if weight_is_prepacked:
		stride_bk = weight.stride(0)
		stride_bn = weight.stride(1)
	else:
		# PyTorch Linear weights are [Out, In] (N, K). The kernel expects B as
		# [K, N], so ordinary weights are read through transposed strides.
		stride_bk = weight.stride(1)
		stride_bn = weight.stride(0)

	_int8_matmul_dequant_kernel[grid](
	    # Pointers
	    a_ptr=x_int8,
	    b_ptr=weight,
	    c_ptr=output,
	    a_scale_ptr=x_scale,
	    b_scale_ptr=weight_scale,
	    bias_ptr=bias_ptr,
	    # Shapes
	    M=M,
	    N=N,
	    K=K,
	    # Strides
	    stride_am=x_int8.stride(0),
	    stride_ak=x_int8.stride(1),
	    stride_bk=stride_bk,
	    stride_bn=stride_bn,
	    stride_cm=output.stride(0),
	    stride_cn=output.stride(1),
	    # Meta
	    **launch_kwargs)

	# 6. Reshape output
	return output.reshape(x_shape_orig[:-1] + (N, ))


# =============================================================================
# Kernel 3: INT8 GEMM + Fused Dequant with Per-Row Weight Scales
# =============================================================================


@_KERNEL_STRATEGY_DECORATOR
@triton.jit
def _int8_matmul_dequant_per_row_kernel(
		# Pointers
		a_ptr,
		b_ptr,
		c_ptr,
		a_scale_ptr,
		b_scale_ptr,
		bias_ptr,
		# Matrix Dimensions
		M,
		N,
		K,
		# Strides
		stride_am,
		stride_ak,
		stride_bk,
		stride_bn,
		stride_cm,
		stride_cn,
		# Meta-parameters
		BLOCK_M: tl.constexpr,
		BLOCK_N: tl.constexpr,
		BLOCK_K: tl.constexpr,
		GROUP_SIZE_M: tl.constexpr,
		HAS_BIAS: tl.constexpr,
		HAS_MN_TAIL: tl.constexpr,
		LEGACY_UNSAFE: tl.constexpr):
	"""
	Computes: C = ((A * B) * (scale_a[:, None] * scale_b[None, :])) + bias
	A: [M, K] int8, scale_a: [M, 1] per-row activation scales
	B: [N, K] int8, scale_b: [N, 1] per-row weight scales
	"""
	pid = tl.program_id(axis=0)
	num_pid_m = tl.cdiv(M, BLOCK_M)
	num_pid_n = tl.cdiv(N, BLOCK_N)
	num_pid_in_group = GROUP_SIZE_M * num_pid_n
	group_id = pid // num_pid_in_group
	first_pid_m = group_id * GROUP_SIZE_M
	group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
	pid_m = first_pid_m + (pid % group_size_m)
	pid_n = (pid % num_pid_in_group) // group_size_m

	# 1. Prepare Pointers for A and B
	if LEGACY_UNSAFE:
		offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
		offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
	else:
		offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
		offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_k = tl.arange(0, BLOCK_K)

	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
	b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

	# 2. Main Loop (Accumulate in Int32)
	accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

	for k in range(0, tl.cdiv(K, BLOCK_K)):
		k_mask = offs_k < K - k * BLOCK_K
		if LEGACY_UNSAFE:
			a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
			b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
		elif HAS_MN_TAIL:
			a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
			b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
		else:
			a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
			b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
		accumulator += tl.dot(a, b)
		a_ptrs += BLOCK_K * stride_ak
		b_ptrs += BLOCK_K * stride_bk

	# 3. Fused Epilogue (Dequantize & Bias)
	if LEGACY_UNSAFE:
		scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]
		scale_b = tl.load(b_scale_ptr + offs_bn)  # Vector [BLOCK_N]
	elif HAS_MN_TAIL:
		scale_a = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=0.0)  # Vector [BLOCK_M]
		scale_b = tl.load(b_scale_ptr + offs_bn, mask=offs_bn < N, other=0.0)  # Vector [BLOCK_N]
	else:
		scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]
		scale_b = tl.load(b_scale_ptr + offs_bn)  # Vector [BLOCK_N]

	c = accumulator.to(tl.float32)
	total_scale = scale_a[:, None] * scale_b[None, :]
	c = c * total_scale

	if HAS_BIAS:
		if LEGACY_UNSAFE:
			bias = tl.load(bias_ptr + offs_bn)
		elif HAS_MN_TAIL:
			bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
		else:
			bias = tl.load(bias_ptr + offs_bn)
		c = c + bias[None, :]

	# 4. Store Result
	c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
	if LEGACY_UNSAFE:
		c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
	elif HAS_MN_TAIL:
		c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
	else:
		c_mask = True
	tl.store(c_ptrs, c, mask=c_mask)


# =============================================================================
# Python Wrapper (Per-Row Weight Scales)
# =============================================================================


def triton_int8_linear_per_row(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias=None, compute_dtype=torch.float16, weight_is_prepacked: bool = False, legacy_unsafe: bool = False):
	"""
	Fused pipeline for W8A8 Linear Layer with per-row weight quantization.
	weight_scale: [N, 1] per-row scales
	"""
	# 1. Flatten inputs if 3D
	x_shape_orig = x.shape
	x_2d = x.reshape(-1, x_shape_orig[-1])

	M, K = x_2d.shape
	N = weight.shape[1] if weight_is_prepacked else weight.shape[0]

	# 2. Dynamic Activation Quantization
	x_int8, x_scale = triton_quantize_rowwise(x_2d)

	# 3. Allocate Output
	output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

	# 4. Prepare weight scales - flatten [N, 1] -> [N] for kernel
	ws = weight_scale.reshape(N).contiguous()

	# 5. Fused GEMM + Per-Row Dequant
	grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )

	has_bias = bias is not None
	bias_ptr = bias if has_bias else x  # Dummy pointer if None
	launch_kwargs = {"HAS_BIAS": has_bias, "HAS_MN_TAIL": True, "LEGACY_UNSAFE": bool(legacy_unsafe)}
	if not _ENABLE_TRITON_AUTOTUNE:
		has_mn_tail = (M % _FIXED_KERNEL_CONFIG["BLOCK_M"]) != 0 or (N % _FIXED_KERNEL_CONFIG["BLOCK_N"]) != 0
		launch_kwargs.update({
			"BLOCK_M": _FIXED_KERNEL_CONFIG["BLOCK_M"],
			"BLOCK_N": _FIXED_KERNEL_CONFIG["BLOCK_N"],
			"BLOCK_K": _FIXED_KERNEL_CONFIG["BLOCK_K"],
			"GROUP_SIZE_M": _FIXED_KERNEL_CONFIG["GROUP_SIZE_M"],
			"HAS_MN_TAIL": has_mn_tail,
			"LEGACY_UNSAFE": bool(legacy_unsafe),
			"num_warps": _FIXED_KERNEL_CONFIG["num_warps"],
			"num_stages": _FIXED_KERNEL_CONFIG["num_stages"],
		})

	_int8_matmul_dequant_per_row_kernel[grid](
		a_ptr=x_int8,
		b_ptr=weight,
		c_ptr=output,
		a_scale_ptr=x_scale,
		b_scale_ptr=ws,
		bias_ptr=bias_ptr,
		M=M,
		N=N,
		K=K,
		stride_am=x_int8.stride(0),
		stride_ak=x_int8.stride(1),
		stride_bk=weight.stride(0) if weight_is_prepacked else weight.stride(1),
		stride_bn=weight.stride(1) if weight_is_prepacked else weight.stride(0),
		stride_cm=output.stride(0),
		stride_cn=output.stride(1),
		**launch_kwargs)

	# 6. Reshape output
	return output.reshape(x_shape_orig[:-1] + (N, ))
```

### 3.5 `patches/comfy_quant_int8.py` — Triton helpers + Linear wrap (full contiguous block)

```python

# Plan B: Triton INT8 Linear accelerate (default ON until a loader stamps OFF)
_HSWQ_TRITON_ACCELERATE_DEFAULT = True
_TRITON_LINEAR_FAIL_LOGGED = False
_TRITON_LINEAR_OK_LOGGED = False
_INT8_TRITON_TINY_M = 16

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


def _iter_stamp_modules(model):
    """Yield unique nn.Modules under ModelPatcher / diffusion_model."""
    seen = set()
    roots = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        roots.append(inner)
        dm = getattr(inner, "diffusion_model", None)
        if dm is not None:
            roots.append(dm)
    for root in roots:
        modules_fn = getattr(root, "modules", None)
        if not callable(modules_fn):
            continue
        for mod in modules_fn():
            mid = id(mod)
            if mid in seen:
                continue
            seen.add(mid)
            yield mod


def _extract_int8_plain_tensors(weight):
    """Return (weight_q int8, weight_scale) or None."""
    try:
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

        if getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout":
            return TensorWiseINT8Layout.get_plain_tensors(weight)
    except Exception:
        pass
    qdata = getattr(weight, "_qdata", None)
    params = getattr(weight, "_params", None)
    if qdata is None or params is None:
        return None
    scale = getattr(params, "scale", None)
    if scale is None:
        return None
    return qdata, scale


def _try_triton_int8_linear(module, input_tensor, weight, bias):
    """Plan B fused Triton path. Returns output tensor or None to fall through."""
    global _TRITON_LINEAR_FAIL_LOGGED, _TRITON_LINEAR_OK_LOGGED

    if getattr(module, "quant_format", None) != "int8_tensorwise":
        return None
    accelerate = getattr(module, "_hswq_triton_accelerate", _HSWQ_TRITON_ACCELERATE_DEFAULT)
    if not accelerate:
        return None

    import torch
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return None
    if not isinstance(weight, QuantizedTensor):
        return None
    layout_cls = getattr(weight, "_layout_cls", None)
    if layout_cls not in (None, "TensorWiseINT8Layout"):
        if getattr(module, "layout_type", None) != "TensorWiseINT8Layout":
            return None
    elif layout_cls is None and getattr(module, "layout_type", None) not in (
        None,
        "TensorWiseINT8Layout",
    ):
        return None

    params = getattr(weight, "_params", None)
    if params is not None:
        if getattr(params, "convrot", False):
            return None
        if getattr(params, "transposed", False):
            return None

    try:
        from .int8_triton_kernels import (
            cuda_sm_ok_for_triton_int8,
            fused_int8_linear,
            is_triton_int8_available,
        )
    except ImportError:
        return None
    if not is_triton_int8_available():
        return None

    x = input_tensor
    if isinstance(x, QuantizedTensor):
        x = x.dequantize()
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return None

    # Plan success criterion: SM >= 8.0. Optional SM 7.5 via env.
    if not cuda_sm_ok_for_triton_int8((8, 0)):
        allow_sm75 = os.environ.get("HSWQ_TRITON_INT8_ALLOW_SM75", "0") == "1"
        if not (allow_sm75 and cuda_sm_ok_for_triton_int8((7, 5))):
            return None

    plain = _extract_int8_plain_tensors(weight)
    if plain is None:
        return None
    weight_q, weight_scale = plain

    x_2d_rows = x.reshape(-1, x.shape[-1]).shape[0]
    if x_2d_rows <= _INT8_TRITON_TINY_M:
        w_f = weight.dequantize().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w_f, bias)

    try:
        out = fused_int8_linear(
            x.contiguous(),
            weight_q.contiguous(),
            weight_scale,
            bias,
            compute_dtype=x.dtype,
        )
    except Exception as exc:
        if not _TRITON_LINEAR_FAIL_LOGGED:
            _TRITON_LINEAR_FAIL_LOGGED = True
            logger.warning(
                "[HSWQ INT8 Triton] fused Linear failed once — using F.linear fallback: %s",
                exc,
            )
        return None

    if not _TRITON_LINEAR_OK_LOGGED:
        _TRITON_LINEAR_OK_LOGGED = True
        _console("[HSWQ INT8 Triton] fused W8A8 Linear path active (Plan B)")
    return out


def _patch_linear_triton_forward(linear_cls) -> bool:
    """Wrap MixedPrecisionOps.Linear._forward for Plan B Triton INT8."""
    if linear_cls is None:
        return False
    if getattr(linear_cls, "_hswq_triton_linear_patched", False):
        return True
    orig_forward = getattr(linear_cls, "_forward", None)
    if orig_forward is None or not callable(orig_forward):
        return False

    def _forward(self, input, weight, bias):
        accelerated = _try_triton_int8_linear(self, input, weight, bias)
        if accelerated is not None:
            return accelerated
        return orig_forward(self, input, weight, bias)

    linear_cls._forward = _forward
    linear_cls._hswq_triton_linear_patched = True
    return True


```

### 3.6 `patches/comfy_quant_int8.py` — ops patch v2 (Linear fuse on MixedPrecisionOps)

```python
        ops_module._load_quantized_module = _load_quantized_module_patched

    # Also normalize Embedding's direct json.loads path by wrapping Embedding._load_from_state_dict
    # is covered if convert_old_quants + file markers are normalized; keep load wrapper as safety.

    original_mp = getattr(ops_module, "mixed_precision_ops", None)
    if original_mp is None or not callable(original_mp):
        return False
    _OPS_PATCH_VER = 2
    true_orig = getattr(original_mp, "_hswq_orig_mixed_precision_ops", original_mp)
    if (
        getattr(original_mp, "_hswq_int8_ops_ver", 0) >= _OPS_PATCH_VER
        and getattr(original_mp, "_hswq_int8_conv_patched", False)
    ):
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
        result = true_orig(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        # Inject Quantized Conv2d only during INT8 load scope (or explicit
        # int8_tensorwise quant_config). FP MixedPrecisionOps must keep upstream
        # Conv2d — detect_layer_quantization() is {"mixed_ops": True} for both.
        if _should_inject_int8_conv(quant_config):
            result.Conv2d = _make_quantized_conv2d(ops_module, result, disabled)
        # Plan B: fuse Triton INT8 Linear on this MixedPrecisionOps.Linear class.
        _patch_linear_triton_forward(getattr(result, "Linear", None))
        return result

    mixed_precision_ops_force_conv._hswq_orig_mixed_precision_ops = true_orig
    mixed_precision_ops_force_conv._hswq_int8_conv_patched = True
    mixed_precision_ops_force_conv._hswq_int8_ops_ver = _OPS_PATCH_VER
    ops_module.mixed_precision_ops = mixed_precision_ops_force_conv
    return True


def _patch_lowvram_patch_float_intermediate() -> bool:
    """Fix LowVramPatch intermediate_dtype for comfy_quant QuantizedTensor only.

    Upstream LowVramPatch passes intermediate_dtype=weight.dtype. When the
    weight is still a QuantizedTensor (int8 storage), LoRA matmul casts to
    int8 and either errors or silently produces a no-op delta — same bug as
    BobJohnson24/ComfyUI-INT8-Fast#76.

    Must NOT divert bare ``torch.int8`` tensors. Nunchaku SVDQ / Lumina2 use
    int8 storage; grabbing them here corrupts fused CUDA (Abort in
    ``_forward_silu_gating``) even when VRAM handoff already freed GPU memory.
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
    _LV_VER = 3
    if original is None or getattr(original, "_hswq_int8_lora_dtype_ver", 0) >= _LV_VER:
        return getattr(original, "_hswq_int8_lora_dtype", False)
    true_orig = getattr(original, "_hswq_orig_lowvram_call", original)

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

    __call__._hswq_int8_lora_dtype = True
    __call__._hswq_int8_lora_dtype_ver = _LV_VER
    __call__._hswq_orig_lowvram_call = true_orig
```

### 3.7 `patches/comfy_quant_int8.py` — stamp + loaders + dispatch

```python
def _stamp_triton_accelerate(model, triton_accelerate):
    """Persist the Triton accelerate toggle on MODEL + INT8 Linear modules."""
    global _HSWQ_TRITON_ACCELERATE_DEFAULT
    enabled = bool(triton_accelerate)
    _HSWQ_TRITON_ACCELERATE_DEFAULT = enabled
    if model is None:
        return
    opts = getattr(model, "model_options", None)
    if not isinstance(opts, dict):
        opts = {}
        try:
            model.model_options = opts
        except Exception:
            pass
    if isinstance(opts, dict):
        opts["hswq_triton_accelerate"] = enabled
    stamped = 0
    for mod in _iter_stamp_modules(model):
        if getattr(mod, "quant_format", None) == "int8_tensorwise":
            try:
                mod._hswq_triton_accelerate = enabled
                stamped += 1
            except Exception:
                pass
    logging.info(
        "[HSWQ INT8] Triton accelerate toggle: %s (stamped %d int8_tensorwise modules)",
        "ON" if enabled else "OFF",
        stamped,
    )


def load_unet_hswq_weight_dtype(unet_name, weight_dtype, triton_accelerate=True):
    import logging
    import torch
    import folder_paths
    import comfy.sd

    # INT8 Conv2d + comfy_quant decode patches only when loading INT8.
    # Do not apply for FP8 / default weight_dtype (those stay on original path).
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(unet_path)

    if is_int8:
        apply_comfy_quant_int8_patches()
        model_options = {"hswq_triton_accelerate": bool(triton_accelerate)}


def load_unet_hswq_weight_dtype(unet_name, weight_dtype, triton_accelerate=True):
    import logging
    import torch
    import folder_paths
    import comfy.sd

    # INT8 Conv2d + comfy_quant decode patches only when loading INT8.
    # Do not apply for FP8 / default weight_dtype (those stay on original path).
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(unet_path)

    if is_int8:
        apply_comfy_quant_int8_patches()
        model_options = {"hswq_triton_accelerate": bool(triton_accelerate)}
        reset_int8_lora_log_counters()
        logging.info("[HSWQ INT8] Loading UNet via MixedPrecisionOps (int8_tensorwise / comfy_quant)")
        print(f"[HSWQ INT8] Loading UNet: {unet_name}", flush=True)
        with _int8_quant_conv_scope():
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        _stamp_triton_accelerate(model, triton_accelerate)
        summarize_int8_lora_capability(model)
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

    return (model,)


def load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device=None, triton_accelerate=True):
    import sys
    import torch
    import folder_paths
    import comfy.sd

    pkg = sys.modules[__name__.rsplit(".", 2)[0]]
    get_current_device = pkg.get_current_device
    set_current_device = pkg.set_current_device
    sdxl_logger = pkg.sdxl_logger

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        # INT8 Conv2d + comfy_quant decode only when checkpoint is INT8.
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        # Auto-detect native comfy_quant INT8; do not force float8 dtype over int8 weights.
        is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(ckpt_path)



def load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device=None, triton_accelerate=True):
    import sys
    import torch
    import folder_paths
    import comfy.sd

    pkg = sys.modules[__name__.rsplit(".", 2)[0]]
    get_current_device = pkg.get_current_device
    set_current_device = pkg.set_current_device
    sdxl_logger = pkg.sdxl_logger

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        # INT8 Conv2d + comfy_quant decode only when checkpoint is INT8.
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        # Auto-detect native comfy_quant INT8; do not force float8 dtype over int8 weights.
        is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(ckpt_path)

        model_options = {}
        if is_int8:
            apply_comfy_quant_int8_patches()
            reset_int8_lora_log_counters()
            model_options["hswq_triton_accelerate"] = bool(triton_accelerate)
            sdxl_logger.info(
                "[SDXL INT8] Loading checkpoint via MixedPrecisionOps "
                "(int8_tensorwise / comfy_quant): %s",
                ckpt_name,
            )
            with _int8_quant_conv_scope():
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=False,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options=model_options,
                )
            model, clip, _v = out[:3]
            _stamp_triton_accelerate(model, triton_accelerate)
            summarize_int8_lora_capability(model)
            return (model, clip)

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


def install_int8_option_dispatch(node_class_mappings) -> bool:
    if not isinstance(node_class_mappings, dict):
        return False

    # Do NOT apply INT8 patches at node registration / import.
    # Patches install only inside load_unet_hswq_weight_dtype /
    # load_checkpoint_sdxl_hswq_weight_dtype when INT8 is actually loaded.

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is not None:
        _orig_load_unet = unet_cls.load_unet

        def load_unet(self, unet_name, weight_dtype, triton_accelerate=True):
            # Explicit FP8 choices stay on the original FP loader body — never INT8 helper.
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_unet(self, unet_name, weight_dtype, triton_accelerate=triton_accelerate)
            if weight_dtype == "int8_tensorwise":
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype, triton_accelerate=triton_accelerate)
            # default: auto-detect INT8 checkpoints only; otherwise original FP path.
            import folder_paths

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            if checkpoint_looks_like_comfy_quant_int8(unet_path):
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype, triton_accelerate=triton_accelerate)


def install_int8_option_dispatch(node_class_mappings) -> bool:
    if not isinstance(node_class_mappings, dict):
        return False

    # Do NOT apply INT8 patches at node registration / import.
    # Patches install only inside load_unet_hswq_weight_dtype /
    # load_checkpoint_sdxl_hswq_weight_dtype when INT8 is actually loaded.

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is not None:
        _orig_load_unet = unet_cls.load_unet

        def load_unet(self, unet_name, weight_dtype, triton_accelerate=True):
            # Explicit FP8 choices stay on the original FP loader body — never INT8 helper.
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_unet(self, unet_name, weight_dtype, triton_accelerate=triton_accelerate)
            if weight_dtype == "int8_tensorwise":
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype, triton_accelerate=triton_accelerate)
            # default: auto-detect INT8 checkpoints only; otherwise original FP path.
            import folder_paths

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            if checkpoint_looks_like_comfy_quant_int8(unet_path):
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype, triton_accelerate=triton_accelerate)
            return _orig_load_unet(self, unet_name, weight_dtype, triton_accelerate=triton_accelerate)

        unet_cls.load_unet = load_unet

    sdxl_cls = node_class_mappings.get("NunchakuUssoewwinCheckpointLoaderSDXL")
    if sdxl_cls is not None:
        _orig_load_checkpoint = sdxl_cls.load_checkpoint

        def load_checkpoint(self, ckpt_name, weight_dtype, triton_accelerate=True, device=None):
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_checkpoint(
                    self, ckpt_name, weight_dtype, triton_accelerate=triton_accelerate, device=device
                )
            if weight_dtype == "int8_tensorwise":
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device, triton_accelerate=triton_accelerate
                )
            import folder_paths

            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if checkpoint_looks_like_comfy_quant_int8(ckpt_path):
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device, triton_accelerate=triton_accelerate
                )
            return _orig_load_checkpoint(
                self, ckpt_name, weight_dtype, triton_accelerate=triton_accelerate, device=device
            )

        sdxl_cls.load_checkpoint = load_checkpoint

    return True
```

### 3.8 `__init__.py` — SDXL loader (full Triton-related class body)

`install_int8_option_dispatch` later wraps `load_checkpoint` so INT8 loads call
`load_checkpoint_sdxl_hswq_weight_dtype(..., triton_accelerate=...)`. The class body
below is the registered surface that declares the UI field and accepts the kwarg.

```python
# From __init__.py — class NunchakuUssoewwinCheckpointLoaderSDXL (Triton-related region)

                req = {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "SDXL checkpoint to load MODEL and CLIP from (same as standard Load Checkpoint)."}),
                    "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "int8_tensorwise"],),
                    "triton_accelerate": ("BOOLEAN", {
                        "default": True,
                        "label": "Triton accelerate",
                        "tooltip": "When ON and weight_dtype is int8_tensorwise (or an INT8 checkpoint is auto-detected), use Triton fused INT8 Linear if Triton is installed. OFF forces eager/_int_mm.",
                    }),
                }
                opt = {"device": (devices, {"default": default_dev})}
                return {"required": req, "optional": opt}

            RETURN_TYPES = ("MODEL", "CLIP")
            OUTPUT_TOOLTIPS = ("The UNet diffusion model from checkpoint.", "The CLIP model from the SDXL checkpoint.")
            FUNCTION = "load_checkpoint"
            CATEGORY = "loaders"
            TITLE = "HSWQ Checkpoint Loader (SDXL)"

            def load_checkpoint(self, ckpt_name, weight_dtype, triton_accelerate=True, device=None):
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
```

### 3.9 `hswq/zimage_fp8_e4m3_unet.py` — UNet loader (full Triton-related class body)

Same pattern: UI field + signature on the class; INT8 path is provided by the
dispatch wrapper that calls `load_unet_hswq_weight_dtype(..., triton_accelerate=...)`.

```python
# From hswq/zimage_fp8_e4m3_unet.py — class HSWQFP8E4M3UNetLoader (Triton-related region)

class HSWQFP8E4M3UNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "int8_tensorwise"],),
                              "triton_accelerate": ("BOOLEAN", {
                                  "default": True,
                                  "label": "Triton accelerate",
                                  "tooltip": "When ON and weight_dtype is int8_tensorwise (or an INT8 checkpoint is auto-detected), use Triton fused INT8 Linear if Triton is installed. OFF forces eager/_int_mm.",
                              }),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "advanced/loaders"
    TITLE = "HSWQ FP8 E4M3/INT8 UNet Loader"

    def load_unet(self, unet_name, weight_dtype, triton_accelerate=True):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        return (model,)
```

---

## 4. Meaning of each piece (confirmed)

### 4.1 Soft-import wrapper (`int8_triton_kernels.py`)

| Symbol | Confirmed meaning |
|--------|-------------------|
| Soft `try/except` import of `_int8_triton_kernels_impl` | Graph load does not crash if Triton is absent |
| `is_triton_int8_available()` | One-shot warning + boolean for callers |
| `cuda_sm_ok_for_triton_int8` | Gate: SM ≥ 8.0 (Ampere+) |
| `fused_int8_linear` | Dispatches scalar vs per-row weight scale kernels |

### 4.2 Kernel implementation (`_int8_triton_kernels_impl.py`)

| Piece | Confirmed meaning |
|-------|-------------------|
| Fixed config / optional `INT8_TRITON_AUTOTUNE=1` | Default fixed tiles for stable public latency; autotune opt-in |
| `_quantize_rowwise_kernel` + `tl.range` tiles | Row absmax then quant without one huge power-of-2 block → **K=10240 works** |
| `_torch_quantize_rowwise` | Soft fallback for quant; GEMM can still be Triton |
| `_int8_matmul_dequant_kernel` | `C = (A@B) * (scale_a * scale_b) + bias` with int32 accumulate |
| `_int8_matmul_dequant_per_row_kernel` | Same with per-output-row weight scales (HSWQ tensorwise layout) |
| `triton_int8_linear` / `_per_row` | Flatten tokens → quant act → launch GEMM → reshape |

Environment knobs: `INT8_TRITON_BLOCK_*`, `INT8_TRITON_ROWWISE_QUANT_TILE`, `INT8_TRITON_ROWWISE_QUANT_MAX_COLS` (0 = unlimited).

### 4.3 `comfy_quant_int8.py` wiring

| Piece | Confirmed meaning |
|-------|-------------------|
| `_try_triton_int8_linear` | All safety gates in one place; returns `None` to keep original path |
| Skip `convrot` / `transposed` | Those layouts are not the plain W8A8 GEMM this kernel expects |
| `_INT8_TRITON_TINY_M = 16` | Tiny batches use `F.linear` |
| `_patch_linear_triton_forward` | Wraps only MixedPrecisionOps `Linear._forward` — not global F.linear |
| Ops patch **v2** | Re-applies even if older v1 Conv-only patch was already installed |
| `_stamp_triton_accelerate` | Persists UI toggle onto `model_options` and each `int8_tensorwise` module |
| Loader / dispatch kwargs | BOOLEAN flows from node → load helper → stamp |
| `install_int8_option_dispatch` | Replaces `load_unet` / `load_checkpoint` at registration; INT8 (or auto-detected INT8) calls the helpers that stamp the toggle; FP8 stays on the original class body |

### 4.4 Loader UI + JS

| Piece | Confirmed meaning |
|-------|-------------------|
| BOOLEAN `triton_accelerate` default True | Default attempts fused path when Triton READY |
| Class body accepts `triton_accelerate` but FP path ignores stamp | Stamp happens only inside INT8 helpers after dispatch |
| JS forces `widget.type = "toggle"` | Same UX pattern as other HSWQ toggles |
| Tooltip | Documents ON vs OFF behavior for end users |

### 4.5 `install.py` Triton stage

| Piece | Confirmed meaning |
|-------|-------------------|
| After requirements | Triton stage runs on every Manager install/update of this node |
| Windows `triton-windows` | Windows installs `triton-windows<3.7` after uninstalling stock `triton` |
| Linux `triton` | Linux installs standard `triton` when not already importable |
| READY / UNAVAILABLE logs | Operators can verify without reading Python exceptions |
| Install failure does not abort node install | INT8 still loads; fused path stays off until `import triton` succeeds |

### 4.6 Runtime evidence (confirmed console lines)

| Log line (literal) | Confirmed meaning |
|--------------------|-------------------|
| `Triton accelerate toggle: ON (stamped N …)` | Stamp succeeded on N INT8 Linears |
| `fused W8A8 Linear path active (Plan B)` | First successful fused Linear launch (string is exact as printed by code) |
| `only supports <= 8192 … got 10240` | Does **not** appear with the shipped tiled quant |
| SA3 / FA-2 restore lines | Attention backends are independent of Linear GEMM |

### 4.7 Coexistence (confirmed)

INT8 Triton Linear acceleration is orthogonal to Flash-Attention 2 / SageAttention 2 / SageAttention 3. Those patch attention; this path accelerates **QuantizedTensor INT8 Linear GEMM** only.

---

## 5. Operator checklist

1. Install / re-run `install.py` → look for `INT8 Triton speed path: READY`.
2. Use `HSWQFP8E4M3UNetLoader` or SDXL HSWQ checkpoint loader with INT8 weights.
3. Leave **Triton accelerate** ON (default).
4. Sample once; confirm the literal log `fused W8A8 Linear path active (Plan B)`.
5. Toggle OFF → expect eager/_int_mm only (fused path gated off).

---

## 6. References

- Design record: `md/INT8_TRITON_W8A8_PUBLIC_ACCELERATION_PLAN.md`
- Kernel recipe lineage: ComfyUI-INT8-Fast `int8_fused_kernel.py` / forge kitchen INT8 linear family
- Ship commit: `bf8a8a6`

---

End of guide.
