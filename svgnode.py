import vtracer
import os
import time
import folder_paths
import numpy as np
import torch
import fitz
import random
import folder_paths

from io import BytesIO
from PIL import Image
from comfy_extras.nodes_images import SVG
from nodes import SaveImage

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class TS_ImageToSVGStringColor:
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

class TS_ImageToSVGStringBW:
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


class TS_SVGStringToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_svg_to_image"
    CATEGORY = "💎TOSVG"

    def convert_svg_to_image(self, SVG_String):

        doc = fitz.open(stream=SVG_String.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        pix = page.get_pixmap()

        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        return (pil2tensor(pil_image),)
    
    
class TS_SaveSVGString:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True}),              
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

    def save_svg_file(self, SVG_String, filename_prefix="ComfyUI_SVG", append_timestamp=True, custom_output_path=""):
        
        output_path = custom_output_path if custom_output_path else self.output_dir
        os.makedirs(output_path, exist_ok=True)
        
        unique_filename = self.generate_unique_filename(f"{filename_prefix}", append_timestamp)
        final_filepath = os.path.join(output_path, unique_filename)
            
            
        with open(final_filepath, "w") as svg_file:
            svg_file.write(SVG_String)
            
            
        ui_info = {"ui": {"saved_svg": unique_filename, "path": final_filepath}}

        return ui_info




class TS_SVGStringPreview(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True})
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

    def svg_preview(self, SVG_String):
        doc = fitz.open(stream=SVG_String.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        pix = page.get_pixmap()

        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        preview = pil2tensor(pil_image)

        return self.save_images(preview, "PointPreview")

class TS_SVGStringToSVGBytesIO:
    """
    将字符串类型的SVG转换为ComfyUI的SVG类型（BytesIO列表）
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("SVG",)
    FUNCTION = "convert_string_to_svg"
    CATEGORY = "💎TOSVG"

    def convert_string_to_svg(self, SVG_String):
        # 将字符串转换为BytesIO对象
        svg_bytes = BytesIO(SVG_String.encode('utf-8'))
        # 返回ComfyUI的SVG类型
        return (SVG([svg_bytes]),)

class TS_SVGBytesIOToString:
    """
    将ComfyUI的SVG类型（BytesIO列表）转换为字符串类型
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_BytesIO": ("SVG", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_svg_to_string"
    CATEGORY = "💎TOSVG"

    def convert_svg_to_string(self, SVG_BytesIO):
        # 获取第一个BytesIO对象的内容
        if not SVG_BytesIO.data:
            return ("",)
        
        # 读取BytesIO对象的内容并解码为字符串
        svg_bytes = SVG_BytesIO.data[0].getvalue()
        svg_string = svg_bytes.decode('utf-8')
        
        return (svg_string,)


NODE_CLASS_MAPPINGS = {
    "TS_ImageToSVGStringColor": TS_ImageToSVGStringColor,
    "TS_ImageToSVGStringBW": TS_ImageToSVGStringBW,
    "TS_SVGStringToImage": TS_SVGStringToImage,
    "TS_SaveSVGString": TS_SaveSVGString,
    "TS_SVGStringPreview": TS_SVGStringPreview,
    "TS_SVGStringToSVGBytesIO": TS_SVGStringToSVGBytesIO,
    "TS_SVGBytesIOToString": TS_SVGBytesIOToString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageToSVGStringColor": "Image to SVG String Color",
    "TS_ImageToSVGStringBW": "Image to SVG String BW",
    "TS_SVGStringToImage": "SVG String to Image",
    "TS_SaveSVGString": "Save SVG String",
    "TS_SVGStringPreview": "SVG String Preview",
    "TS_SVGStringToSVGBytesIO": "SVG String to SVG BytesIO",
    "TS_SVGBytesIOToString": "SVG BytesIO to SVG String",
}