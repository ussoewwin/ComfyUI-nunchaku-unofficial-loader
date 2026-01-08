"""Check if to_safely exists"""
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
print("to_safely exists:", hasattr(NunchakuSDXLUNet2DConditionModel, 'to_safely'))
