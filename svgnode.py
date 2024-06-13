import vtracer
import os
import time
import folder_paths
import numpy as np
from PIL import Image
from typing import List, Tuple
import torch

def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class ConvertRasterToVector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormode": (["color", "binary"], {"default": "color"}),
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 100}),
                "color_precision": ("INT", {"default": 6, "min": 0, "max": 10}),
                "layer_difference": ("INT", {"default": 16, "min": 0, "max": 256}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 70}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180}),
                "path_precision": ("INT", {"default": 3, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "convert_to_svg"

    CATEGORY = "💎TOSVG"

    def convert_to_svg(self, image, colormode, hierarchical, mode, filter_speckle, color_precision, layer_difference, corner_threshold, length_threshold, max_iterations, splice_threshold, path_precision):
        
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
                colormode=colormode,
                hierarchical=hierarchical,
                mode=mode,
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                layer_difference=layer_difference,
                corner_threshold=corner_threshold,
                length_threshold=length_threshold,
                max_iterations=max_iterations,
                splice_threshold=splice_threshold,
                path_precision=path_precision
            )

            
            svg_strings.append(svg_str)

        return (svg_strings,)  





class SaveSVG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_strings": ("LIST", {"forceInput": True}),              
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
        
        ui_info_list = []  

        for index, svg_string in enumerate(svg_strings):
            
            unique_filename = self.generate_unique_filename(f"{filename_prefix}_{index}", append_timestamp)
            final_filepath = os.path.join(output_path, unique_filename)
            
            
            with open(final_filepath, "w") as svg_file:
                svg_file.write(svg_string)
            
            
            ui_info = {"ui": {"saved_svg": unique_filename, "path": final_filepath}}
            ui_info_list.append(ui_info)

        return ui_info_list


