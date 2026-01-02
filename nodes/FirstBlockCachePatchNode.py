import logging
import time
import functools
import types

import torch
import comfy
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch, set_model_patch_replace, \
    is_nunchaku_model, is_nunchaku_sdxl_model, is_nunchaku_zimage_model

logger = logging.getLogger(__name__)

fb_cache_key_attrs = "fb_cache_attr"
fb_cache_model_temp = "nunchaku_fb_cache"

def get_fb_cache_global_cache(transformer_options, timesteps):
    diffusion_model = transformer_options.get(PatchKeys.running_net_model)
    if hasattr(diffusion_model, fb_cache_model_temp):
        tea_cache = getattr(diffusion_model, fb_cache_model_temp, {})
        transformer_options[fb_cache_key_attrs] = tea_cache

    attrs = transformer_options.get(fb_cache_key_attrs, {})
    attrs['step_i'] = timesteps[0].detach().cpu().item()
    
    # Initialize statistics if not present
    if 'cache_stats' not in attrs:
        attrs['cache_stats'] = {
            'total_steps': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'steps_in_range': 0,
            'steps_out_range': 0,
            'first_block_calc_count': 0,
            'first_block_cache_count': 0,
            'start_time': time.time()
        }
        logger.info(f"[FB Cache] Initialized cache_stats at step {attrs['step_i']:.2f}")

# For Nunchaku SDXL models (UNet2DConditionModel based)
def fb_cache_enter_sdxl(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    logger.info(f"[FB Cache] enter_sdxl called at step {timesteps[0].detach().cpu().item():.2f}")
    get_fb_cache_global_cache(transformer_options, timesteps)
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask

# For Nunchaku Z-Image models (DiT based)
def fb_cache_enter_zimage(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    logger.info(f"[FB Cache] enter_zimage called at step {timesteps[0].detach().cpu().item():.2f}")
    get_fb_cache_global_cache(transformer_options, timesteps)
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask

def are_two_tensors_similar(t1, t2, *, threshold):
    if t1.shape != t2.shape:
        return False
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = mean_diff / mean_t1
    return diff.item() < threshold

def _patch_unet_for_cache(unet_model, fb_cache_attrs, residual_diff_threshold):
    """
    Patch UNet's forward method and transformer_blocks to implement First Block Cache for Nunchaku SDXL.
    """
    # First, patch the forward method to initialize cache context with timestep
    if not hasattr(unet_model, '_fb_cache_original_forward'):
        original_forward = unet_model.forward
        
        @functools.wraps(original_forward)
        def cached_unet_forward(self, sample, timestep, *args, **kwargs):
            # Initialize cache context with timestep
            cache_attrs = getattr(self, fb_cache_model_temp, {})
            if not cache_attrs:
                cache_attrs = fb_cache_attrs.copy()
                setattr(self, fb_cache_model_temp, cache_attrs)
            
            # Update step_i from timestep
            if isinstance(timestep, torch.Tensor):
                if timestep.dim() > 0:
                    step_i = timestep[0].detach().cpu().item()
                else:
                    step_i = timestep.detach().cpu().item()
            else:
                step_i = float(timestep)
            
            cache_attrs['step_i'] = step_i
            
            # Initialize cache_stats if not present
            if 'cache_stats' not in cache_attrs:
                cache_attrs['cache_stats'] = {
                    'total_steps': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'steps_in_range': 0,
                    'steps_out_range': 0,
                    'first_block_calc_count': 0,
                    'first_block_cache_count': 0,
                    'start_time': time.time()
                }
                logger.info(f"[FB Cache] Initialized cache_stats at step {step_i:.2f}")

            # Call original forward (do NOT compute previous_residual here; UNet output type/value is sampler/model dependent)
            return original_forward(sample, timestep, *args, **kwargs)
        
        unet_model.forward = types.MethodType(cached_unet_forward, unet_model)
        unet_model._fb_cache_original_forward = original_forward
        logger.info("[FB Cache] Patched UNet forward method")
    
    # Then patch transformer_blocks
    _patch_unet_transformer_blocks_for_cache(unet_model, fb_cache_attrs, residual_diff_threshold)

def _patch_unet_transformer_blocks_for_cache(unet_model, fb_cache_attrs, residual_diff_threshold):
    """
    Patch UNet's transformer_blocks to implement First Block Cache for Nunchaku SDXL.
    
    This function patches the first transformer_block in the first down_block's first attention
    to cache its output and reuse it when similar.
    """
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
    
    # Track which block is the first one to execute
    first_block_found = False
    
    def _patch_transformer_block(transformer_block, is_first_block):
        """Patch a single transformer_block's forward method."""
        if not hasattr(transformer_block, '_fb_cache_original_forward'):
            original_forward = transformer_block.forward
            
            @functools.wraps(original_forward)
            def cached_forward(self, hidden_states, *args, **kwargs):
                # Get cache context from model
                cache_attrs = getattr(unet_model, fb_cache_model_temp, {})
                if not cache_attrs:
                    cache_attrs = fb_cache_attrs.copy()
                    setattr(unet_model, fb_cache_model_temp, cache_attrs)

                # If sampler is known unsafe for block-skipping cache, disable cache for this run
                if cache_attrs.get("fb_cache_disable_for_sampler", False):
                    cache_attrs["should_calc"] = True
                    return original_forward(hidden_states, *args, **kwargs)
                
                stats = cache_attrs.get('cache_stats', {})
                step_i = cache_attrs.get('step_i', 0)
                timestep_start = cache_attrs.get('timestep_start', float('inf'))
                timestep_end = cache_attrs.get('timestep_end', float('-inf'))
                in_step = timestep_end <= step_i <= timestep_start
                
                if not in_step:
                    stats['steps_out_range'] = stats.get('steps_out_range', 0) + 1
                    return original_forward(hidden_states, *args, **kwargs)
                
                stats['steps_in_range'] = stats.get('steps_in_range', 0) + 1
                
                # Only cache the first block
                if is_first_block:
                    # Heun / multi-stage samplers can call the model multiple times with the same timestep.
                    # Mixing cache across same-step evaluations can break the image, so disable cache within the same step.
                    last_step_i = cache_attrs.get('fb_cache_last_step_i')
                    if last_step_i is not None and step_i == last_step_i:
                        # Force calculation and do not touch previous_first_block_residual (keep it as "previous step")
                        cache_attrs['should_calc'] = True
                        calc_start = time.time()
                        output = original_forward(hidden_states, *args, **kwargs)
                        calc_time = time.time() - calc_start
                        stats['cache_misses'] = stats.get('cache_misses', 0) + 1
                        stats['first_block_calc_count'] = stats.get('first_block_calc_count', 0) + 1
                        stats['total_steps'] = stats.get('total_steps', 0) + 1
                        logger.info(f"[FB Cache] Step {step_i:.2f}: FORCE CALC (same timestep repeated, calc: {calc_time*1000:.2f}ms)")
                        return output

                    # Calculate first block (always calculate, then check similarity)
                    calc_start = time.time()
                    output = original_forward(hidden_states, *args, **kwargs)
                    calc_time = time.time() - calc_start
                    
                    # Check cache by comparing output with previous output
                    previous_first_block_residual = cache_attrs.get('previous_first_block_residual')
                    should_calc = True  # Default to True
                    
                    if previous_first_block_residual is not None:
                        similarity_check_start = time.time()
                        should_calc = not are_two_tensors_similar(
                            previous_first_block_residual,
                            output,
                            threshold=cache_attrs.get('rel_diff_threshold', residual_diff_threshold)
                        )
                        similarity_check_time = time.time() - similarity_check_start
                        
                        if not should_calc:
                            # Cache hit - use previous residual
                            stats['cache_hits'] = stats.get('cache_hits', 0) + 1
                            stats['first_block_cache_count'] = stats.get('first_block_cache_count', 0) + 1
                            logger.info(f"[FB Cache] Step {step_i:.2f}: CACHE HIT (similarity: {similarity_check_time*1000:.2f}ms, saved: {calc_time*1000:.2f}ms)")
                        else:
                            # Cache miss
                            stats['cache_misses'] = stats.get('cache_misses', 0) + 1
                            stats['first_block_calc_count'] = stats.get('first_block_calc_count', 0) + 1
                            logger.info(f"[FB Cache] Step {step_i:.2f}: CACHE MISS (similarity: {similarity_check_time*1000:.2f}ms, calc: {calc_time*1000:.2f}ms)")
                            cache_attrs['previous_first_block_residual'] = output.clone()
                    else:
                        # First time, no cache
                        stats['cache_misses'] = stats.get('cache_misses', 0) + 1
                        stats['first_block_calc_count'] = stats.get('first_block_calc_count', 0) + 1
                        logger.info(f"[FB Cache] Step {step_i:.2f}: CACHE MISS (no previous residual, calc: {calc_time*1000:.2f}ms)")
                        cache_attrs['previous_first_block_residual'] = output.clone()
                    
                    cache_attrs['should_calc'] = should_calc
                    cache_attrs['fb_cache_last_step_i'] = step_i
                    stats['total_steps'] = stats.get('total_steps', 0) + 1
                    
                    return output
                else:
                    # Not first block, use cached result if available
                    should_calc = cache_attrs.get('should_calc', True)
                    if should_calc:
                        return original_forward(hidden_states, *args, **kwargs)
                    else:
                        # Skip calculation, use cached result
                        return hidden_states
            
            transformer_block.forward = types.MethodType(cached_forward, transformer_block)
            transformer_block._fb_cache_original_forward = original_forward
            transformer_block._fb_cache_patched = True
            return True
        return False
    
    # Patch the first transformer_block in the first down_block's first attention
    patched_count = 0
    
    # Find and patch the first transformer_block in the first down_block that has attention
    # Note: First down_block may not have attention, so we need to find the first one with attention
    for block_idx, down_block in enumerate(unet_model.down_blocks):
        if hasattr(down_block, 'attentions') and len(down_block.attentions) > 0:
            first_attn = down_block.attentions[0]
            if isinstance(first_attn, Transformer2DModel) and hasattr(first_attn, 'transformer_blocks'):
                if len(first_attn.transformer_blocks) > 0:
                    first_tb = first_attn.transformer_blocks[0]
                    if _patch_transformer_block(first_tb, is_first_block=True):
                        patched_count += 1
                        first_block_found = True
                        logger.info(f"[FB Cache] Patched first transformer_block: down_{block_idx}_attn_0_tb_0")
                        break  # Only patch the first one found
    
    # Patch other blocks (not first) to respect should_calc flag
    # Find the first block index to skip it
    first_block_block_idx = None
    for block_idx, down_block in enumerate(unet_model.down_blocks):
        if hasattr(down_block, 'attentions') and len(down_block.attentions) > 0:
            first_attn = down_block.attentions[0]
            if isinstance(first_attn, Transformer2DModel) and hasattr(first_attn, 'transformer_blocks'):
                if len(first_attn.transformer_blocks) > 0:
                    first_block_block_idx = block_idx
                    break
    
    for block_idx, down_block in enumerate(unet_model.down_blocks):
        if hasattr(down_block, 'attentions'):
            for attn_idx, attn in enumerate(down_block.attentions):
                if isinstance(attn, Transformer2DModel) and hasattr(attn, 'transformer_blocks'):
                    for tb_idx, transformer_block in enumerate(attn.transformer_blocks):
                        # Skip the first block we already patched
                        if not (first_block_found and block_idx == first_block_block_idx and attn_idx == 0 and tb_idx == 0):
                            if not hasattr(transformer_block, '_fb_cache_patched'):
                                if _patch_transformer_block(transformer_block, is_first_block=False):
                                    patched_count += 1
    
    # Patch mid_block
    if hasattr(unet_model, 'mid_block') and hasattr(unet_model.mid_block, 'attentions'):
        for attn_idx, attn in enumerate(unet_model.mid_block.attentions):
            if isinstance(attn, Transformer2DModel) and hasattr(attn, 'transformer_blocks'):
                for tb_idx, transformer_block in enumerate(attn.transformer_blocks):
                    if not hasattr(transformer_block, '_fb_cache_patched'):
                        if _patch_transformer_block(transformer_block, is_first_block=False):
                            patched_count += 1
    
    # Patch up_blocks
    for block_idx, up_block in enumerate(unet_model.up_blocks):
        if hasattr(up_block, 'attentions'):
            for attn_idx, attn in enumerate(up_block.attentions):
                if isinstance(attn, Transformer2DModel) and hasattr(attn, 'transformer_blocks'):
                    for tb_idx, transformer_block in enumerate(attn.transformer_blocks):
                        if not hasattr(transformer_block, '_fb_cache_patched'):
                            if _patch_transformer_block(transformer_block, is_first_block=False):
                                patched_count += 1
    
    logger.info(f"[FB Cache] Patched {patched_count} transformer_blocks for UNet cache (first_block_found={first_block_found})")

def fb_cache_patch_double_block_with_control_replace(original_args, wrapper_options):
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    stats = attrs.get('cache_stats', {})
    step_i = attrs['step_i']
    timestep_start = attrs['timestep_start']
    timestep_end = attrs['timestep_end']
    in_step = timestep_end <= step_i <= timestep_start
    
    if not in_step:
        attrs['should_calc'] = True
        stats['steps_out_range'] = stats.get('steps_out_range', 0) + 1
        return wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)

    stats['steps_in_range'] = stats.get('steps_in_range', 0) + 1
    block_i = original_args['i']
    txt = original_args['txt']
    
    if block_i == 0:
        # Compare with the first double block output from the previous sampling.
        # If the absolute mean difference is less than threshold, cache can be used.
        calc_start = time.time()
        img, txt = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)
        calc_time = time.time() - calc_start

        previous_first_block_residual = attrs.get('previous_first_block_residual')
        if previous_first_block_residual is not None:
            similarity_check_start = time.time()
            should_calc = not are_two_tensors_similar(previous_first_block_residual, img, threshold=attrs['rel_diff_threshold'])
            similarity_check_time = time.time() - similarity_check_start
            
            if not should_calc:
                # Cache hit
                stats['cache_hits'] = stats.get('cache_hits', 0) + 1
                stats['first_block_cache_count'] = stats.get('first_block_cache_count', 0) + 1
                logger.debug(f"[FB Cache] Step {step_i:.2f}: CACHE HIT (similarity check: {similarity_check_time*1000:.2f}ms, saved calc: {calc_time*1000:.2f}ms)")
            else:
                # Cache miss
                stats['cache_misses'] = stats.get('cache_misses', 0) + 1
                stats['first_block_calc_count'] = stats.get('first_block_calc_count', 0) + 1
                logger.debug(f"[FB Cache] Step {step_i:.2f}: CACHE MISS (similarity check: {similarity_check_time*1000:.2f}ms, calc: {calc_time*1000:.2f}ms)")
        else:
            # Need to calculate, i.e., do not use cache
            should_calc = True
            stats['cache_misses'] = stats.get('cache_misses', 0) + 1
            stats['first_block_calc_count'] = stats.get('first_block_calc_count', 0) + 1
            logger.debug(f"[FB Cache] Step {step_i:.2f}: CACHE MISS (no previous residual, calc: {calc_time*1000:.2f}ms)")

        if should_calc:
            attrs['previous_first_block_residual'] = img.clone()
        else:
            # Previous non-cached sampling value
            previous_residual = attrs.get('previous_residual')
            if previous_residual is not None:
                img += previous_residual

        attrs['should_calc'] = should_calc
        attrs['ori_img'] = None
        stats['total_steps'] = stats.get('total_steps', 0) + 1
    else:
        img = original_args['img']
        should_calc = attrs['should_calc']
        if should_calc:
            if attrs['ori_img'] is None:
                attrs['ori_img'] = original_args['img'].clone()
            if block_i > 0:
                img, txt = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)

    del attrs, transformer_options
    return img, txt

def fb_cache_patch_blocks_transition_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)
    return img

def fb_cache_patch_single_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def fb_cache_patch_blocks_after_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args)
    return img

def fb_cache_patch_final_transition_after(img, txt, transformer_options):
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        if attrs.get('ori_img') is not None:
            attrs['previous_residual'] = img - attrs['ori_img']
    return img

def fb_cache_patch_dit_exit(img, transformer_options):
    tea_cache = transformer_options.get(fb_cache_key_attrs, {})
    diffusion_model = transformer_options.get(PatchKeys.running_net_model)
    if diffusion_model is not None:
        setattr(diffusion_model, fb_cache_model_temp, tea_cache)
        # Debug: log if cache_stats exists
        if 'cache_stats' in tea_cache:
            stats = tea_cache.get('cache_stats', {})
            logger.info(f"[FB Cache] dit_exit called, cache_stats saved: total_steps={stats.get('total_steps', 0)}, hits={stats.get('cache_hits', 0)}, misses={stats.get('cache_misses', 0)}")
        else:
            logger.info(f"[FB Cache] dit_exit called, but cache_stats not found in tea_cache (keys: {list(tea_cache.keys())})")
    else:
        logger.warning("[FB Cache] dit_exit called, but running_net_model is None")
    return img

def fb_cache_prepare_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                  callback=None, disable_pbar=False, seed=None, latent_shapes=None, **kwargs):
    cfg_guider = wrapper_executor.class_obj
    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    
    # Initialize start time for statistics
    start_time = time.time()
    # Detect sampler that is known to call the model multiple times per step (multi-stage),
    # where skipping blocks based on cache can easily break images (e.g. HeunPP2).
    sampler_name = None
    try:
        # Try multiple ways to get the actual sampler name
        if callable(sampler):
            sampler_name = getattr(sampler, "__name__", None)
            if not sampler_name or "sampler" in sampler_name.lower():
                # Try to get from __qualname__ or __module__
                sampler_name = getattr(sampler, "__qualname__", None) or getattr(sampler, "__module__", None) or sampler_name
        else:
            sampler_name = getattr(sampler, "sampler_name", None) or getattr(sampler, "name", None) or type(sampler).__name__
        
        # If we still don't have a good name, try to inspect the sampler object
        if not sampler_name or "ksampler" in str(sampler_name).lower():
            # Try to get all attributes and find sampler-related ones
            attrs = dir(sampler)
            for attr in attrs:
                if "heun" in attr.lower() or "sampler" in attr.lower():
                    val = getattr(sampler, attr, None)
                    if val and (callable(val) or isinstance(val, str)):
                        if callable(val):
                            name = getattr(val, "__name__", None)
                            if name and "heun" in name.lower():
                                sampler_name = name
                                break
                        elif isinstance(val, str) and "heun" in val.lower():
                            sampler_name = val
                            break
    except Exception as e:
        sampler_name = type(sampler).__name__
        logger.debug(f"[FB Cache] Failed to get sampler name: {e}")

    sampler_name_l = (str(sampler_name) if sampler_name is not None else "").lower()
    # Also check the string representation of the sampler object itself
    sampler_str = str(sampler).lower()
    sampler_repr = repr(sampler).lower()
    
    # Debug: log sampler details for troubleshooting
    logger.debug(f"[FB Cache] Sampler detection: name={sampler_name}, type={type(sampler).__name__}, str={sampler_str[:100]}")
    
    disable_cache_for_sampler = (
        ("heunpp2" in sampler_name_l) or ("heun++" in sampler_name_l) or ("heunpp" in sampler_name_l) or ("heun" in sampler_name_l) or
        ("heunpp2" in sampler_str) or ("heun++" in sampler_str) or ("heunpp" in sampler_str) or ("heun" in sampler_str) or
        ("heunpp2" in sampler_repr) or ("heun++" in sampler_repr) or ("heunpp" in sampler_repr) or ("heun" in sampler_repr)
    )

    # Store per-run flags on the model object so patched blocks can read it
    if not hasattr(diffusion_model, fb_cache_model_temp):
        setattr(diffusion_model, fb_cache_model_temp, {})
    cache_attrs = getattr(diffusion_model, fb_cache_model_temp, {})
    cache_attrs["fb_cache_sampler_name"] = str(sampler_name)
    cache_attrs["fb_cache_disable_for_sampler"] = bool(disable_cache_for_sampler)

    logger.info(
        f"[FB Cache] prepare_wrapper started, diffusion_model: {type(diffusion_model).__name__}, "
        f"sampler={sampler_name}, disable_cache_for_sampler={disable_cache_for_sampler}"
    )
    
    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed, latent_shapes=latent_shapes, **kwargs)
    finally:
        # Get cache statistics from diffusion_model (saved by fb_cache_patch_dit_exit)
        cache_stats = None
        has_cache_attr = hasattr(diffusion_model, fb_cache_model_temp)
        logger.info(f"[FB Cache] prepare_wrapper finally block, hasattr({fb_cache_model_temp}): {has_cache_attr}")
        if has_cache_attr:
            cache_attrs = getattr(diffusion_model, fb_cache_model_temp, {})
            logger.info(f"[FB Cache] cache_attrs keys: {list(cache_attrs.keys())}")
            cache_stats = cache_attrs.get('cache_stats', {})
            logger.info(f"[FB Cache] cache_stats found: {cache_stats is not None}, keys: {list(cache_stats.keys()) if cache_stats else 'None'}")
        
        # Log cache statistics after execution
        total_time = time.time() - start_time
        if cache_stats:
            total_steps = cache_stats.get('total_steps', 0)
            cache_hits = cache_stats.get('cache_hits', 0)
            cache_misses = cache_stats.get('cache_misses', 0)
            steps_in_range = cache_stats.get('steps_in_range', 0)
            steps_out_range = cache_stats.get('steps_out_range', 0)
            first_block_calc = cache_stats.get('first_block_calc_count', 0)
            first_block_cache = cache_stats.get('first_block_cache_count', 0)
            
            logger.info(
                f"[FB Cache Stats] Total steps: {total_steps}, "
                f"In range: {steps_in_range}, Out range: {steps_out_range}, "
                f"Cache hits: {cache_hits}, Cache misses: {cache_misses}, "
                f"First block calc: {first_block_calc}, First block cache: {first_block_cache}, "
                f"Total time: {total_time:.2f}s"
            )
            
            if cache_hits + cache_misses > 0:
                hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100
                logger.info(
                    f"[FB Cache Stats] Hit rate: {hit_rate:.1f}% "
                    f"(hits: {cache_hits}, misses: {cache_misses})"
                )
                
                # Estimate speedup
                if hit_rate > 0:
                    # Rough estimate: each cache hit saves ~1 first block calculation
                    # Assuming first block is ~10-20% of total computation
                    estimated_speedup = 1.0 + (hit_rate / 100.0) * 0.15
                    logger.info(f"[FB Cache Stats] Estimated speedup: {estimated_speedup:.2f}x (based on {hit_rate:.1f}% hit rate)")
            else:
                logger.info("[FB Cache Stats] No cache hits/misses recorded (cache may not be active in this step range)")
        else:
            logger.info(f"[FB Cache Stats] No cache statistics found (cache may not be initialized or active)")
        
        if hasattr(diffusion_model, fb_cache_model_temp):
            delattr(diffusion_model, fb_cache_model_temp)

    return out

class NunchakuUssoewwinApplyFirstBlockCachePatchAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "residual_diff_threshold": ("FLOAT",
                                            {
                                                "default": 0.00,
                                                "min": 0.0,
                                                "max": 1.0,
                                                "step": 0.01,
                                                "tooltip": "Nunchaku SDXL: 0 (original), 0.12 (1.8x speedup).\n"
                                                           "Nunchaku Z-Image: 0 (original), 0.1 (1.6x speedup)."
                                            }),
                "start_at": ("FLOAT",
                             {
                                 "default": 0.0,
                                 "step": 0.01,
                                 "max": 1.0,
                                 "min": 0.0
                             }
                             ),
                "end_at": ("FLOAT",
                           {
                               "default": 1.0,
                               "step": 0.01,
                               "max": 1.0,
                               "min": 0.0
                           })
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch_advanced"
    CATEGORY = "nunchaku_unofficial/speed"
    TITLE = "Nunchaku Apply First Block Cache Patch Advanced"
    DESCRIPTION = ("Apply the First Block Cache patch to accelerate Nunchaku models. Use it together with nodes that have the suffix ForwardOverrider."
                   "\nThis is effective only for Nunchaku SDXL and Nunchaku Z-Image models.")

    def apply_patch_advanced(self, model, residual_diff_threshold, start_at=0.0, end_at=1.0):

        model = model.clone()
        patch_key = "fb_cache_wrapper"
        if residual_diff_threshold == 0 or len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) > 0:
            logger.info(f"[FB Cache] Patch not applied: residual_diff_threshold={residual_diff_threshold}, wrapper_exists={len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) > 0}")
            return (model,)

        diffusion_model = model.get_model_object('diffusion_model')
        if not is_nunchaku_model(diffusion_model):
            logger.warning("First Block Cache patch is not applied because the model is not a Nunchaku model.")
            return (model,)
        
        logger.info(f"[FB Cache] Applying patch to {type(diffusion_model).__name__}, residual_diff_threshold={residual_diff_threshold}, start_at={start_at}, end_at={end_at}")

        fb_cache_attrs = add_model_patch_option(model, fb_cache_key_attrs)

        fb_cache_attrs['rel_diff_threshold'] = residual_diff_threshold
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_at)
        sigma_end = model_sampling.percent_to_sigma(end_at)
        # Convert to tensor if needed (percent_to_sigma may return float)
        if not isinstance(sigma_start, torch.Tensor):
            sigma_start = torch.tensor(sigma_start, dtype=torch.float32)
        if not isinstance(sigma_end, torch.Tensor):
            sigma_end = torch.tensor(sigma_end, dtype=torch.float32)
        fb_cache_attrs['timestep_start'] = model_sampling.timestep(sigma_start)
        fb_cache_attrs['timestep_end'] = model_sampling.timestep(sigma_end)

        # Use appropriate patch method based on Nunchaku model type
        if is_nunchaku_zimage_model(diffusion_model):
            # Z-Image is DiT-based, use DiT patches
            logger.info(f"[FB Cache] Setting patch for Z-Image model: {PatchKeys.options_key}, {PatchKeys.dit_enter}")
            set_model_patch(model, PatchKeys.options_key, fb_cache_enter_zimage, PatchKeys.dit_enter)
            set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_double_block_with_control_replace, PatchKeys.dit_double_block_with_control_replace)
            set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_blocks_transition_replace, PatchKeys.dit_blocks_transition_replace)
            set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_single_blocks_replace, PatchKeys.dit_single_blocks_replace)
            set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_blocks_after_replace, PatchKeys.dit_blocks_after_transition_replace)
            set_model_patch(model, PatchKeys.options_key, fb_cache_patch_final_transition_after, PatchKeys.dit_final_layer_before)
            set_model_patch(model, PatchKeys.options_key, fb_cache_patch_dit_exit, PatchKeys.dit_exit)
        elif is_nunchaku_sdxl_model(diffusion_model):
            # SDXL is UNet-based, need to patch UNet's forward and transformer_blocks directly
            logger.info(f"[FB Cache] Patching UNet forward and transformer_blocks for SDXL model")
            _patch_unet_for_cache(diffusion_model, fb_cache_attrs, residual_diff_threshold)
        else:
            logger.warning(f"[FB Cache] Unknown Nunchaku model type: {type(diffusion_model).__name__}")

        # Just add it once when connecting in series
        logger.info(f"[FB Cache] Adding wrapper: {patch_key}")
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   patch_key,
                                   fb_cache_prepare_wrapper
                                   )
        logger.info("[FB Cache] Patch applied successfully")
        return (model, )

