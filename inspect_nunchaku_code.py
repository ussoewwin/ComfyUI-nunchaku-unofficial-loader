import inspect
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

print("="*80)
print("INSPECTING: NunchakuSDXLUNet2DConditionModel.forward")
print("="*80)

try:
    src = inspect.getsource(NunchakuSDXLUNet2DConditionModel.forward)
    print(src)
except Exception as e:
    print(f"Error getting source: {e}")

print("="*80)
print("INSPECTING: NunchakuSDXLUNet2DConditionModel (Class Doc/Init)")
print("="*80)
try:
    src = inspect.getsource(NunchakuSDXLUNet2DConditionModel)
    # Print first 50 lines to check init/imports
    print("\n".join(src.splitlines()[:100]))
except Exception as e:
    print(f"Error getting source: {e}")
