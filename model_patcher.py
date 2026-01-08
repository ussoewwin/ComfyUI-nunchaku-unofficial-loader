"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""

import time
from comfy.model_patcher import ModelPatcher


class NunchakuModelPatcher(ModelPatcher):
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_count = 0
        self._detach_count = 0

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """
        Load the diffusion model onto the specified device.
        """
        self._load_count += 1
        start = time.perf_counter()
        
        with self.use_ejected():
            diffusion_model = self.model.diffusion_model
            if hasattr(diffusion_model, 'to_safely'):
                diffusion_model.to_safely(device_to)
            else:
                diffusion_model.to(device_to)
        
        elapsed = (time.perf_counter() - start) * 1000
        if self._load_count <= 3 or self._load_count % 10 == 0:
            print(f"[NunchakuModelPatcher] load #{self._load_count} to {device_to}: {elapsed:.1f}ms")
        self.current_device = device_to

    def detach(self, unpatch_all: bool = True):
        """
        Detach the model and move it to the offload device.
        """
        self._detach_count += 1
        start = time.perf_counter()
        
        self.eject_model()
        diffusion_model = self.model.diffusion_model
        if hasattr(diffusion_model, 'to_safely'):
            diffusion_model.to_safely(self.offload_device)
        else:
            diffusion_model.to(self.offload_device)
        
        elapsed = (time.perf_counter() - start) * 1000
        if self._detach_count <= 3 or self._detach_count % 10 == 0:
            print(f"[NunchakuModelPatcher] detach #{self._detach_count} to {self.offload_device}: {elapsed:.1f}ms")
        self.current_device = self.offload_device
