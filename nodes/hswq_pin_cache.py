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
            try:
                mm.unpin_memory(pin)
            except Exception:
                pass
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
                    if torch.cuda.cudart().cudaHostUnregister(pin.data_ptr()) != 0:
                        mm.discard_cuda_async_error()
                    else:
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


@contextmanager
def hswq_pin_cache_scope():
    """Context manager: PinCache on only for Batched Detailer."""
    ok = activate()
    try:
        yield ok
    finally:
        deactivate()
