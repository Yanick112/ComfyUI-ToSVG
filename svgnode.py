import vtracer
import os
import time
import folder_paths
import numpy as np
from PIL import Image
from typing import List, Tuple
import torch
from io import BytesIO
import fitz
import random
import folder_paths

from PIL import Image
from nodes import SaveImage

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ConvertRasterToVectorColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "color_precision": ("INT", {"default": 6, "min": 0, "max": 10, "step": 1}),
                "layer_difference": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180, "step": 1}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 70, "step": 1}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180, "step": 1}),
                "path_precision": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert_to_svg"

    CATEGORY = "💎TOSVG"

    def convert_to_svg(self, image, hierarchical, mode, filter_speckle, color_precision, layer_difference, corner_threshold,
                       length_threshold, max_iterations, splice_threshold, path_precision):
        
        svg_strings = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)
            
            if _image.mode != 'RGBA':
                alpha = Image.new('L', _image.size, 255)
                _image.putalpha(alpha)

            pixels = list(_image.getdata())
            size = _image.size

            svg_str = vtracer.convert_pixels_to_svg(
                pixels,
                size=size,
                colormode="color",
                hierarchical=hierarchical,
                mode=mode,
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                layer_difference=layer_difference,
                corner_threshold=corner_threshold,
                length_threshold=length_threshold,
                max_iterations=max_iterations,
                splice_threshold=splice_threshold,
                path_precision=path_precision,
            )
            
            svg_strings.append(svg_str)

        return (svg_strings,)

class ConvertRasterToVectorBW:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180, "step": 1}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert_to_svg"

    CATEGORY = "💎TOSVG"

    def convert_to_svg(self, image, mode, filter_speckle, corner_threshold, length_threshold, splice_threshold):
        
        svg_strings = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)
            
            if _image.mode != 'RGBA':
                alpha = Image.new('L', _image.size, 255)
                _image.putalpha(alpha)

            pixels = list(_image.getdata())
            size = _image.size

            svg_str = vtracer.convert_pixels_to_svg(
                pixels,
                size=size,
                colormode="binary",
                mode=mode,
                filter_speckle=filter_speckle,
                corner_threshold=corner_threshold,
                length_threshold=length_threshold,
                splice_threshold=splice_threshold,
            )
            
            svg_strings.append(svg_str)

        return (svg_strings,)


class ConvertVectorToRaster:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_strings": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_svg_to_image"
    CATEGORY = "💎TOSVG"

    def convert_svg_to_image(self, svg_strings):

        doc = fitz.open(stream=svg_strings.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        pix = page.get_pixmap()

        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        return (pil2tensor(pil_image),)
    
    
class SaveSVG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_strings": ("STRING", {"forceInput": True}),              
                "filename_prefix": ("STRING", {"default": "ComfyUI_SVG"}),
            },
            "optional": {
                "append_timestamp": ("BOOLEAN", {"default": True}),
                "custom_output_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = "💎TOSVG"
    DESCRIPTION = "Save SVG data to a file."
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_svg_file"

    def generate_unique_filename(self, prefix, timestamp=False):
        if timestamp:
            timestamp_str = time.strftime("%Y%m%d%H%M%S")
            return f"{prefix}_{timestamp_str}.svg"
        else:
            return f"{prefix}.svg"

    def save_svg_file(self, svg_strings, filename_prefix="ComfyUI_SVG", append_timestamp=True, custom_output_path=""):
        
        output_path = custom_output_path if custom_output_path else self.output_dir
        os.makedirs(output_path, exist_ok=True)
        
        unique_filename = self.generate_unique_filename(f"{filename_prefix}", append_timestamp)
        final_filepath = os.path.join(output_path, unique_filename)
            
            
        with open(final_filepath, "w") as svg_file:
            svg_file.write(svg_strings)
            
            
        ui_info = {"ui": {"saved_svg": unique_filename, "path": final_filepath}}

        return ui_info




class SVGPreview(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svg_strings": ("STRING", {"forceInput": True})
            }
        }

    FUNCTION = "svg_preview"
    CATEGORY = "💎TOSVG"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(5))
        self.compress_level = 4

    def svg_preview(self, svg_strings):
        doc = fitz.open(stream=svg_strings.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        pix = page.get_pixmap()

        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        preview = pil2tensor(pil_image)

        return self.save_images(preview, "PointPreview")
