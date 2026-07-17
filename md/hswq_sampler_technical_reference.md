# HSWQ Sampler Complete Technical Reference

## 1. Purpose of Creation
This custom node (HSWQ Sampler) is an independent implementation designed to **fully support** the specialized samplers and schedulers provided by RES4LYF (`custom_nodes/RES4LYF`), while also **ensuring extensibility** for the future ecosystem of quantized models, including "HSWQ".

In the Forge environment, RES4LYF deploys a complete list of samplers (including over 100 Runge-Kutta samplers derived from `rk_beta`). However, in a vanilla ComfyUI environment, many of these samplers become unselectable from standard `KSampler` nodes due to the effects of `importlib.reload()` during initialization and the lack of dynamic wrapper function generation logic.

This implementation fully replicates and supplements the "dynamic generation and re-registration of all RK samplers" logic found in Forge on ComfyUI. Its purpose is to ensure that all samplers are reliably captured and made available regardless of the load order or module name (`RES4LYF` vs. `custom_nodes.RES4LYF`).

---

## 2. Added / Modified Files (③)

| # | File | Change type | Role of the change |
|---|------|-------------|--------------------|
| 1 | `nodes/hswq_sampler.py` | **Added (new file)** | The entire HSWQ Sampler node: RES4LYF module discovery, Forge-equivalent RK wrapper generation, re-injection into ComfyUI core, and the `HSWQSampler` node class. |
| 2 | `__init__.py` | **Modified** | Registers the node into ComfyUI by importing `HSWQSampler` and inserting it into `NODE_CLASS_MAPPINGS`. The display name is then derived automatically from the class `TITLE` via the existing `NODE_DISPLAY_NAME_MAPPINGS` construction. |

There are **no** other added or modified files for this node. `nodes/hswq_sampler.py` is fully self-contained (it only depends on ComfyUI core modules `comfy.samplers`, `comfy.k_diffusion.sampling`, and `nodes`), and the only integration point in the package is the registration block in `__init__.py` shown in section 3.2.

---

## 3. Full Source Code of Additions / Modifications (④ — nothing omitted)

### 3.1 Full source of the new file `nodes/hswq_sampler.py`

```python
"""
HSWQ Sampler — A node fully equivalent to the standard ComfyUI KSampler.
If RES4LYF (custom_nodes/RES4LYF) is loaded,
it automatically adds all of its samplers / schedulers.

## Reason for bridging the gap with Forge
Forge's modules/RES4LYF/beta/__init__.py dynamically generates wrappers
for rk_sampler_beta.sample_rk_beta for all entries in RK_SAMPLER_NAMES_BETA_NO_FOLDERS
and adds them to extra_samplers.
The ComfyUI version of beta/__init__.py does not have this logic.
This node supplements that missing difference.
"""
import sys
import logging

import comfy.samplers
import comfy.k_diffusion.sampling as _k_diff
import nodes as _nodes

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# RES4LYF Module Discovery
# ────────────────────────────────────────────────

def _find_res4lyf_mod():
    """Find the RES4LYF module containing extra_samplers from sys.modules."""
    for cand in ("RES4LYF", "custom_nodes.RES4LYF"):
        m = sys.modules.get(cand)
        if m is not None and hasattr(m, "extra_samplers"):
            return m
    for name, m in list(sys.modules.items()):
        if m is not None and "RES4LYF" in name and hasattr(m, "extra_samplers"):
            return m
    return None


def _find_rk_sampler_beta_mod():
    """Find the module where sample_rk_beta can be retrieved for comfy.k_diffusion.sampling."""
    # RES4LYF.beta.rk_sampler_beta can be registered under multiple names
    for cand in (
        "RES4LYF.beta.rk_sampler_beta",
        "custom_nodes.RES4LYF.beta.rk_sampler_beta",
        "beta.rk_sampler_beta",
    ):
        m = sys.modules.get(cand)
        if m is not None and hasattr(m, "sample_rk_beta"):
            return m
    # Fallback: scan submodules of the RES4LYF module
    for name, m in list(sys.modules.items()):
        if m is not None and "rk_sampler_beta" in name and hasattr(m, "sample_rk_beta"):
            return m
    return None


def _find_rk_coefficients_mod():
    """Find the module containing RK_SAMPLER_NAMES_BETA_NO_FOLDERS."""
    for cand in (
        "RES4LYF.beta.rk_coefficients_beta",
        "custom_nodes.RES4LYF.beta.rk_coefficients_beta",
        "beta.rk_coefficients_beta",
    ):
        m = sys.modules.get(cand)
        if m is not None and hasattr(m, "RK_SAMPLER_NAMES_BETA_NO_FOLDERS"):
            return m
    for name, m in list(sys.modules.items()):
        if m is not None and "rk_coefficients_beta" in name and hasattr(m, "RK_SAMPLER_NAMES_BETA_NO_FOLDERS"):
            return m
    return None


# ────────────────────────────────────────────────
# Forge Compatibility: Generate and register wrappers for all rk_types
# ────────────────────────────────────────────────

# Do not create ODE versions for implicit samplers (same condition as Forge)
_IMPLICIT_KEYWORDS = (
    "gauss-legendre", "radau", "lobatto",
    "irk_exp_diag", "kraaijevanger", "qin_zhang",
    "pareschi", "crouzeix",
)


def _build_rk_extra_samplers(rk_mod, names) -> dict:
    """
    Identical logic to Forge's beta/__init__.py L92-L119.
    Generates sample_fn / sample_ode_fn closures for all entries in
    RK_SAMPLER_NAMES_BETA_NO_FOLDERS.
    """
    result = {}

    for sampler_name in names:
        if sampler_name == "none":
            continue

        def make_fn(rk_type):
            def sample_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
                return rk_mod.sample_rk_beta(
                    model, x, sigmas, None, extra_args, callback, disable,
                    rk_type=rk_type,
                )
            sample_fn.__name__ = f"sample_{rk_type}"
            return sample_fn

        result[sampler_name] = make_fn(sampler_name)

        # ODE versions (excluding implicit types)
        if not any(kw in sampler_name for kw in _IMPLICIT_KEYWORDS):
            ode_name = f"{sampler_name}_ode"

            def make_ode_fn(rk_type):
                def sample_ode_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
                    return rk_mod.sample_rk_beta(
                        model, x, sigmas, None, extra_args, callback, disable,
                        rk_type=rk_type, eta=0.0, eta_substep=0.0,
                    )
                sample_ode_fn.__name__ = f"sample_{rk_type}_ode"
                return sample_ode_fn

            result[ode_name] = make_ode_fn(sampler_name)

    # generic rk_beta
    result["rk_beta"] = rk_mod.sample_rk_beta

    return result


def _ensure_all_registered(extra: dict) -> None:
    """
    Registers all entries in extra_samplers to KSampler.SAMPLERS and
    comfy.k_diffusion.sampling.
    """
    samplers_list = comfy.samplers.KSampler.SAMPLERS
    insert_after = "uni_pc_bh2"
    try:
        insert_idx = samplers_list.index(insert_after)
    except ValueError:
        insert_idx = len(samplers_list) - 1

    added = 0
    for name, fn in extra.items():
        # Add to KSampler.SAMPLERS
        if name not in samplers_list:
            samplers_list.insert(insert_idx + 1, name)
            insert_idx += 1
            added += 1

        # Inject function into comfy.k_diffusion.sampling (supplements missing functions from reload)
        attr = f"sample_{name}"
        if not hasattr(_k_diff, attr):
            setattr(_k_diff, attr, fn)

    if added:
        logger.info("[HSWQSampler] Registered %d RES4LYF samplers into KSampler.SAMPLERS", added)


# ────────────────────────────────────────────────
# INPUT_TYPES Helpers
# ────────────────────────────────────────────────

def _get_samplers() -> list:
    res4lyf  = _find_res4lyf_mod()
    rk_mod   = _find_rk_sampler_beta_mod()
    coef_mod = _find_rk_coefficients_mod()

    if res4lyf is not None and rk_mod is not None and coef_mod is not None:
        names = getattr(coef_mod, "RK_SAMPLER_NAMES_BETA_NO_FOLDERS", [])
        # Generate wrappers for all rk_types identically to Forge
        rk_extra = _build_rk_extra_samplers(rk_mod, names)
        # Merge with existing extra_samplers
        extra = dict(getattr(res4lyf, "extra_samplers", {}))
        extra.update(rk_extra)
        _ensure_all_registered(extra)
    elif res4lyf is not None:
        extra = getattr(res4lyf, "extra_samplers", {})
        _ensure_all_registered(extra)

    return list(comfy.samplers.KSampler.SAMPLERS)


def _get_schedulers() -> list:
    handlers: dict = getattr(comfy.samplers, "SCHEDULER_HANDLERS", {})
    names: list = list(comfy.samplers.KSampler.SCHEDULERS)
    for name in handlers:
        if name not in names:
            names.append(name)
    return names


# ────────────────────────────────────────────────
# Node Main Class
# ────────────────────────────────────────────────

class HSWQSampler:
    @classmethod
    def INPUT_TYPES(cls):
        samplers   = _get_samplers()
        schedulers = _get_schedulers()
        logger.debug(
            "[HSWQSampler] INPUT_TYPES: %d samplers, %d schedulers",
            len(samplers), len(schedulers),
        )
        return {
            "required": {
                "model":        ("MODEL",),
                "seed":         ("INT",   {"default": 0,   "min": 0,   "max": 0xffffffffffffffff}),
                "steps":        ("INT",   {"default": 20,  "min": 1,   "max": 10000}),
                "cfg":          ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (samplers,),
                "scheduler":    (schedulers,),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "HSWQ Sampler"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0):
        return _nodes.common_ksampler(
            model, seed, steps, cfg,
            sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise,
        )
```

### 3.2 Full source of the modification in `__init__.py`

Registration is done through the standard ComfyUI mechanism: `NODE_CLASS_MAPPINGS` maps a node id to its class, and `NODE_DISPLAY_NAME_MAPPINGS` maps that id to its UI display label. The block below is the exact registration added for this node.

```python
try:
    from .nodes.hswq_sampler import HSWQSampler
    NODE_CLASS_MAPPINGS["HSWQSampler"] = HSWQSampler
    logger.info("Registered HSWQ Sampler")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ Sampler not registered: %s", e)
```

No separate line is needed for the display name. The display-name map is built once, after all nodes are registered, from each class's `TITLE` attribute:

```python
NODE_DISPLAY_NAME_MAPPINGS = {k: getattr(v, "TITLE", k) for k, v in NODE_CLASS_MAPPINGS.items()}
```

Because `HSWQSampler` defines `TITLE = "HSWQ Sampler"`, this single comprehension automatically yields `NODE_DISPLAY_NAME_MAPPINGS["HSWQSampler"] = "HSWQ Sampler"` — no explicit override is required (unlike some other nodes in this file that need one).

---

## 4. Meaning of Each Change (⑤)

### 4.1 Meaning of `nodes/hswq_sampler.py`

#### 4.1.1 Module Discovery Independent of Load Order and Environment Differences (`_find_res4lyf_mod` / `_find_rk_sampler_beta_mod` / `_find_rk_coefficients_mod`)
The load order of ComfyUI custom nodes is undetermined (it depends on directory names and the OS file system). If `HSWQSampler` is loaded before `RES4LYF`, a simple top-level `import RES4LYF` would raise an error and break the node.

To avoid this, discovery is deferred to `INPUT_TYPES`, which runs just before the node UI is rendered (when `/object_info` is requested — i.e. after every node has finished loading). At that point `sys.modules` is scanned dynamically. Each finder first tries the known canonical names, then falls back to a partial-match scan, so the fluctuation between `"RES4LYF"` and `"custom_nodes.RES4LYF"` (which depends on how the user installed RES4LYF) is handled transparently. Three separate finders are needed because the three required symbols — `extra_samplers`, `sample_rk_beta`, and `RK_SAMPLER_NAMES_BETA_NO_FOLDERS` — live in three different submodules.

#### 4.1.2 Replication of Forge's Dynamic Wrapper Generation (`_build_rk_extra_samplers` + `_IMPLICIT_KEYWORDS`)
RES4LYF defines a large list of Runge-Kutta sampler names (`RK_SAMPLER_NAMES_BETA_NO_FOLDERS`) internally, but vanilla ComfyUI has no process to register these as individually selectable samplers. Forge's `modules/RES4LYF/beta/__init__.py` loops through this list and dynamically generates closures that call `sample_rk_beta` with the corresponding `rk_type`, and also generates ODE (Ordinary Differential Equation) variants.

`_build_rk_extra_samplers` reproduces that logic exactly:
- For every name it builds a `sample_fn` closure (via `make_fn`) that forwards to `rk_mod.sample_rk_beta(..., rk_type=rk_type)`.
- For every **non-implicit** name it also builds an `_ode` variant (via `make_ode_fn`) that pins `eta=0.0, eta_substep=0.0`. Implicit families listed in `_IMPLICIT_KEYWORDS` (gauss-legendre, radau, lobatto, etc.) are excluded from ODE generation — the same condition Forge uses.
- `make_fn` / `make_ode_fn` are used deliberately so that `rk_type` is captured per-iteration; defining the closures inline in the loop without them would let every closure share the last loop value.
- Finally the generic `rk_beta` entry is added.

The result is that the complete Forge sampler family is materialised in memory, ready to be registered.

#### 4.1.3 Reliable Re-injection of Functions into ComfyUI Core (`_ensure_all_registered`)
RES4LYF's initialization sometimes calls `importlib.reload(k_diffusion_sampling)`. This wipes the function references of custom samplers that were previously added, so a sampler can appear in the UI yet fail at run time because its function object no longer exists in the core module.

`_ensure_all_registered` guards against this by enforcing **both** halves of registration for every sampler:
1. **UI selectability** — the name is inserted into `comfy.samplers.KSampler.SAMPLERS`, placed right after `uni_pc_bh2` (with a safe fallback to the list end if that anchor is missing).
2. **Runtime execution** — the function is injected into `comfy.k_diffusion.sampling` under the `sample_<name>` attribute via `setattr`, but only if it is not already present, so a healthy core is never overwritten.

Because both are applied on every `INPUT_TYPES` call, the sampler set stays consistent regardless of RES4LYF's internal reload timing.

#### 4.1.4 INPUT_TYPES Orchestration (`_get_samplers` / `_get_schedulers`)
`_get_samplers` ties the pieces together: if all three RES4LYF submodules are found, it builds the Forge-equivalent wrappers, merges them with RES4LYF's existing `extra_samplers`, and registers everything; if only the top-level module is found, it registers whatever `extra_samplers` already exposes. In either case it returns the up-to-date `KSampler.SAMPLERS` list. `_get_schedulers` merges the modern `SCHEDULER_HANDLERS` keys with the classic `KSampler.SCHEDULERS` list so custom schedulers also remain selectable.

#### 4.1.5 The Node Class as a Thin, Extensible Wrapper (`HSWQSampler`)
The node exposes exactly the same inputs as the standard KSampler and delegates execution to `nodes.common_ksampler`, so behaviour is identical to the built-in sampler. Keeping it a thin wrapper is intentional: quantized inference such as HSWQ may later need sampler-side hooks (injecting quantization scales, switching per-step compute precision, etc.), and those can be intercepted inside this `sample` method without modifying the ComfyUI core.

### 4.2 Meaning of the `__init__.py` change
The `try/except` block performs the actual node registration. The import is wrapped so that an environment missing an optional dependency degrades gracefully — instead of aborting the whole package load, a failed import is logged at debug level and every other node still registers. Populating `NODE_CLASS_MAPPINGS["HSWQSampler"]` is what makes ComfyUI expose the node; the subsequent `NODE_DISPLAY_NAME_MAPPINGS` comprehension then derives the human-readable label from the class `TITLE`, which is why no per-node display-name line is required for this node.
