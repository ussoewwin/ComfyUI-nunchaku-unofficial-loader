import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image, PngImagePlugin


class NunchakuSaveImage:
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
                "format": (["PNG", "JPG"], {"default": "PNG", "tooltip": "Output image format."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."}),
                "quality (JPG only)": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "JPEG quality (1-100). Only used when format is JPG; ignored for PNG."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory as PNG or JPG."
    TITLE = "HSWQ Save Image"

    def save_images(self, images, format, filename_prefix="ComfyUI", **kwargs):
        quality = kwargs.get("quality (JPG only)", 95)
        prompt = kwargs.get("prompt", None)
        extra_pnginfo = kwargs.get("extra_pnginfo", None)
        format = format.upper()
        if format not in ("PNG", "JPG"):
            raise ValueError(f"Unsupported format: {format}")

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = None
            if format == "PNG":
                metadata = PngImagePlugin.PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            ext = ".png" if format == "PNG" else ".jpg"
            file = f"{filename_with_batch_num}_{counter:05}{ext}"
            full_path = os.path.join(full_output_folder, file)

            if format == "PNG":
                img.save(full_path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                img.save(full_path, quality=quality, optimize=True)

            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}}
