import vtracer
import os
import time
import folder_paths
import numpy as np
import torch
import fitz
import random
import folder_paths
import potrace
from io import BytesIO
from PIL import Image
from comfy_extras.nodes_images import SVG
from nodes import SaveImage
import re
import xml.etree.ElementTree as ET

def tensor2pil(image):
    """Tensor转PIL图像"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """PIL图像转Tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class TS_ImageQuantize:
    """
    图像量化：通过减少图像中的颜色数量来优化矢量转换过程。
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1}),
                "dither": (["Clear", "Smooth"], {"default": "Clear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize_image"
    CATEGORY = "💎TOSVG/Tools"

    def quantize_image(self, image, colors, dither):
        """
        执行图像量化处理。
        """
        quantized_images = []
        
        dither_method = Image.Dither.NONE
        if dither == "Smooth":
            dither_method = Image.Dither.FLOYDSTEINBERG

        for i in image:
            pil_image = tensor2pil(torch.unsqueeze(i, 0))

            quantized_pil = pil_image.convert('RGB').quantize(colors=colors, dither=dither_method)
            
            quantized_pil_rgb = quantized_pil.convert('RGB')
            
            tensor_image = pil2tensor(quantized_pil_rgb)
            quantized_images.append(tensor_image.squeeze(0))

        if not quantized_images:
            return (image,)

        return (torch.stack(quantized_images),)

class TS_ImageToSVGStringColor_Vtracer:
    """图像转彩色SVG字符串"""
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

    CATEGORY = "💎TOSVG/Convert"

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

class TS_ImageToSVGStringBW_Vtracer:
    """图像转黑白SVG字符串"""
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

    CATEGORY = "💎TOSVG/Convert"

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
    """SVG字符串转图像"""  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_svg_to_image"
    CATEGORY = "💎TOSVG/Convert"

    def convert_svg_to_image(self, SVG_String):

        doc = fitz.open(stream=SVG_String.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        pix = page.get_pixmap()

        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        return (pil2tensor(pil_image),)
    
    
class TS_SaveSVGString:
    """保存SVG字符串到文件"""
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

    CATEGORY = "💎TOSVG/Tools"
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
    """SVG字符串预览"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True})
            }
        }

    FUNCTION = "svg_preview"
    CATEGORY = "💎TOSVG/Tools"
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
    """SVG字符串转BytesIO"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("SVG",)
    FUNCTION = "convert_string_to_svg"
    CATEGORY = "💎TOSVG/Tools"

    def convert_string_to_svg(self, SVG_String):
        svg_bytes = BytesIO(SVG_String.encode('utf-8'))
        return (SVG([svg_bytes]),)

class TS_SVGBytesIOToString:
    """BytesIO转SVG字符串"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_BytesIO": ("SVG", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_svg_to_string"
    CATEGORY = "💎TOSVG/Tools"

    def convert_svg_to_string(self, SVG_BytesIO):
        if not SVG_BytesIO.data:
            return ("",)
        
        svg_bytes = SVG_BytesIO.data[0].getvalue()
        svg_string = svg_bytes.decode('utf-8')
        
        return (svg_string,)


class TS_SVGPathSimplify:
    """SVG路径简化"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True}),
                "tolerance": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "preserve_curves": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "simplify_svg_paths"
    CATEGORY = "💎TOSVG/Tools"

    def douglas_peucker(self, points, tolerance):
        """Douglas-Peucker算法简化路径点"""
        if len(points) <= 2:
            return points
            
        # 找到距离起点和终点连线最远的点
        max_distance = 0
        max_index = 0
        
        start = points[0]
        end = points[-1]
        
        if abs(start[0] - end[0]) < 0.01 and abs(start[1] - end[1]) < 0.01:
            for i in range(1, len(points) - 1):
                distance = ((points[i][0] - start[0]) ** 2 + (points[i][1] - start[1]) ** 2) ** 0.5
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
        else:
            for i in range(1, len(points) - 1):
                distance = self.point_to_line_distance(points[i], start, end)
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
        
        if max_distance > tolerance and max_index > 0:
            left_part = self.douglas_peucker(points[:max_index + 1], tolerance)  
            right_part = self.douglas_peucker(points[max_index:], tolerance)
            return left_part[:-1] + right_part
        else:
            return [start, end]

    def point_to_line_distance(self, point, line_start, line_end):
        """计算点到直线距离"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        if x1 == x2 and y1 == y2:
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        
        return numerator / denominator if denominator > 0 else 0

    def parse_path_commands(self, path_data):
        """解析SVG路径命令"""
        commands = re.findall(r'[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*', path_data)
        return commands

    def extract_points_from_path(self, path_data):
        """提取路径中的坐标点"""
        points = []
        commands = self.parse_path_commands(path_data)
        current_pos = (0, 0)
        
        for command in commands:
            cmd_type = command[0]
            params = re.findall(r'-?\d*\.?\d+', command[1:])
            params = [float(p) for p in params]
            
            if cmd_type in 'Mm':
                if len(params) >= 2:
                    if cmd_type == 'M':
                        current_pos = (params[0], params[1])
                    else:  
                        current_pos = (current_pos[0] + params[0], current_pos[1] + params[1])
                    points.append(current_pos)
                    
            elif cmd_type in 'LlHhVv':
                if cmd_type in 'Ll':
                    for i in range(0, len(params), 2):
                        if i + 1 < len(params):
                            if cmd_type == 'L':
                                current_pos = (params[i], params[i + 1])
                            else:
                                current_pos = (current_pos[0] + params[i], current_pos[1] + params[i + 1])
                            points.append(current_pos)
                elif cmd_type in 'Hh':
                    for param in params:
                        if cmd_type == 'H':
                            current_pos = (param, current_pos[1])
                        else:
                            current_pos = (current_pos[0] + param, current_pos[1])
                        points.append(current_pos)
                elif cmd_type in 'Vv':
                    for param in params:
                        if cmd_type == 'V':
                            current_pos = (current_pos[0], param)
                        else:
                            current_pos = (current_pos[0], current_pos[1] + param)
                        points.append(current_pos)
                        
            elif cmd_type in 'CcSsQqTt':
                if cmd_type in 'Cc':
                    for i in range(0, len(params), 6):
                        if i + 5 < len(params):
                            if cmd_type == 'C':
                                control1 = (params[i], params[i + 1])
                                control2 = (params[i + 2], params[i + 3])
                                end_point = (params[i + 4], params[i + 5])
                                points.extend([control1, control2, end_point])
                                current_pos = end_point
                            else:
                                control1 = (current_pos[0] + params[i], current_pos[1] + params[i + 1])
                                control2 = (current_pos[0] + params[i + 2], current_pos[1] + params[i + 3])
                                end_point = (current_pos[0] + params[i + 4], current_pos[1] + params[i + 5])
                                points.extend([control1, control2, end_point])
                                current_pos = end_point
                elif cmd_type in 'Ss':
                    for i in range(0, len(params), 4):
                        if i + 3 < len(params):
                            if cmd_type == 'S':
                                control2 = (params[i], params[i + 1])
                                end_point = (params[i + 2], params[i + 3])
                                points.extend([control2, end_point])
                                current_pos = end_point
                            else:
                                control2 = (current_pos[0] + params[i], current_pos[1] + params[i + 1])
                                end_point = (current_pos[0] + params[i + 2], current_pos[1] + params[i + 3])
                                points.extend([control2, end_point])
                                current_pos = end_point
                elif cmd_type in 'Qq':
                    for i in range(0, len(params), 4):
                        if i + 3 < len(params):
                            if cmd_type == 'Q':
                                control = (params[i], params[i + 1])
                                end_point = (params[i + 2], params[i + 3])
                                points.extend([control, end_point])
                                current_pos = end_point
                            else:
                                control = (current_pos[0] + params[i], current_pos[1] + params[i + 1])
                                end_point = (current_pos[0] + params[i + 2], current_pos[1] + params[i + 3])
                                points.extend([control, end_point])
                                current_pos = end_point
                elif cmd_type in 'Tt':
                    for i in range(0, len(params), 2):
                        if i + 1 < len(params):
                            if cmd_type == 'T':
                                end_point = (params[i], params[i + 1])
                            else:
                                end_point = (current_pos[0] + params[i], current_pos[1] + params[i + 1])
                            points.append(end_point)
                            current_pos = end_point
        
        return points

    def simplify_path_data(self, path_data, tolerance, preserve_curves, stats):
        """简化路径数据"""
        original_points = self.extract_points_from_path(path_data)
        stats['original_points'] += len(original_points)
        
        if len(original_points) < 3:
            stats['simplified_points'] += len(original_points)
            return path_data
        
        if not preserve_curves:
            filtered_points = [original_points[0]]
            for i in range(1, len(original_points)):
                if (abs(original_points[i][0] - filtered_points[-1][0]) > 0.1 or 
                    abs(original_points[i][1] - filtered_points[-1][1]) > 0.1):
                    filtered_points.append(original_points[i])
            original_points = filtered_points
        
        simplified_points = self.douglas_peucker(original_points, tolerance)
        stats['simplified_points'] += len(simplified_points)
        
        if preserve_curves and len(simplified_points) >= len(original_points) * 0.9:
            stats['simplified_points'] = stats['simplified_points'] - len(simplified_points) + len(original_points)
            return path_data
        
        if len(simplified_points) < 2:
            return path_data
            
        path_parts = [f"M{simplified_points[0][0]:.1f},{simplified_points[0][1]:.1f}"]
        
        for i in range(1, len(simplified_points)):
            x, y = simplified_points[i]
            path_parts.append(f"L{x:.1f},{y:.1f}")
        
        if path_data.strip().endswith('Z') or path_data.strip().endswith('z'):
            path_parts.append('Z')
        
        return ' '.join(path_parts)

    def simplify_svg_paths(self, SVG_String, tolerance, preserve_curves):
        """简化SVG路径"""
        effective_tolerance = tolerance
        
        stats = {
            'original_points': 0,
            'simplified_points': 0,
            'paths_processed': 0,
            'original_size': len(SVG_String),
            'simplified_size': 0
        }
        
        try:
            root = ET.fromstring(SVG_String)
            
            paths = root.findall('.//{http://www.w3.org/2000/svg}path')
            if not paths:
                paths = root.findall('.//path')
            
            for path in paths:
                if 'd' in path.attrib:
                    original_data = path.attrib['d']
                    simplified_data = self.simplify_path_data(original_data, effective_tolerance, preserve_curves, stats)
                    path.attrib['d'] = simplified_data
                    stats['paths_processed'] += 1
            
            ET.register_namespace('', 'http://www.w3.org/2000/svg')
            simplified_svg = ET.tostring(root, encoding='unicode')
            stats['simplified_size'] = len(simplified_svg)
            
        except Exception as e:
            simplified_svg = self.simplify_svg_regex(SVG_String, effective_tolerance, preserve_curves, stats)
            stats['simplified_size'] = len(simplified_svg)
        
        reduction_ratio = (stats['original_points'] - stats['simplified_points']) / max(stats['original_points'], 1) * 100
        print(f"SVG Path Simplified: {reduction_ratio:.1f}% points reduced")
        
        return (simplified_svg,)

    def simplify_svg_regex(self, SVG_String, tolerance, preserve_curves, stats):
        """
        使用正则表达式方法简化SVG路径（备选方案）
        """
        def replace_path(match):
            path_data = match.group(1)
            simplified = self.simplify_path_data(path_data, tolerance, preserve_curves, stats)
            stats['paths_processed'] += 1
            return f'd="{simplified}"'
        
        simplified_svg = re.sub(r'd="([^"]*)"', replace_path, SVG_String)
        
        return simplified_svg


class TS_ImageToSVGStringBW_Potracer:
    """Potracer矢量化为SVG"""
    turnpolicy_map = {
        "minority": potrace.POTRACE_TURNPOLICY_MINORITY,
        "black": potrace.POTRACE_TURNPOLICY_BLACK,
        "white": potrace.POTRACE_TURNPOLICY_WHITE,
        "left": potrace.POTRACE_TURNPOLICY_LEFT,
        "right": potrace.POTRACE_TURNPOLICY_RIGHT,
        "majority": potrace.POTRACE_TURNPOLICY_MAJORITY,
    }

    @classmethod
    def INPUT_TYPES(cls):
        policy_options = list(cls.turnpolicy_map.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"default": 128, "min": 0, "max": 255}),
            },
            "optional": {
                "input_foreground": (["White on Black", "Black on White"], {"default": "Black on White"}),
                "turnpolicy": (policy_options, {"default": "minority"}),
                "turdsize": ("INT", {"default": 2, "min": 0}),
                "corner_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.34, "step": 0.01}),
                "zero_sharp_corners": ("BOOLEAN", {"default": False}),
                "opttolerance": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "optimize_curve": ("BOOLEAN", {"default": True}),
                "foreground_color": ("STRING", {"widget": "color", "default": "#000000"}),
                "background_color": ("STRING", {"widget": "color", "default": "#FFFFFF"}),
                "stroke_color": ("STRING", {"widget": "color", "default": "#FF0000"}),
                "stroke_width": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "vectorize"
    CATEGORY = "💎TOSVG/Convert"

    def vectorize(self, image, threshold, turnpolicy, turdsize, corner_threshold, opttolerance,
                  input_foreground="Black on White", optimize_curve=True,
                  zero_sharp_corners=False,
                  foreground_color="#000000", background_color="#FFFFFF",
                  stroke_color="#FF0000", stroke_width=0.0):
        
        image_np = image.cpu().numpy()
        batch_svg_strings = []

        for i, single_image_np in enumerate(image_np):
            orig_width_temp, orig_height_temp = (single_image_np.shape[1], single_image_np.shape[0]) if single_image_np.ndim >= 2 else (100,100)
            svg_data_for_current_image = f'<svg width="{orig_width_temp}" height="{orig_height_temp}"><desc>Error: Processing failed before SVG generation for image {i}</desc></svg>'

            try:
                pil_img = Image.fromarray((single_image_np * 255).astype(np.uint8))
                orig_width, orig_height = pil_img.size

                if orig_width <= 0 or orig_height <= 0:
                    error_svg = f'<svg width="1" height="1"><desc>Error: Invalid image dimensions for image {i}</desc></svg>'
                    batch_svg_strings.append(error_svg)
                    continue

                threshold_norm = threshold / 255.0
                if single_image_np.ndim == 3:
                    binary_np = single_image_np[:, :, 0] < threshold_norm if single_image_np.shape[2] > 1 else single_image_np[:,:,0] < threshold_norm
                elif single_image_np.ndim == 2:
                    binary_np = single_image_np < threshold_norm
                else:
                    error_svg = f'<svg width="{orig_width}" height="{orig_height}"><desc>Error: Unexpected image dimensions for image {i}</desc></svg>'
                    batch_svg_strings.append(error_svg)
                    continue

                if input_foreground == "Black on White":
                    binary_np = ~binary_np

                if np.all(binary_np) or not np.any(binary_np):
                    skipped_svg = f'<svg width="{orig_width}" height="{orig_height}"><desc>Potracer: Skipped blank image {i}</desc></svg>'
                    batch_svg_strings.append(skipped_svg)
                    continue

                turdsize_int = int(turdsize) if turdsize is not None else 0
                policy_arg = self.turnpolicy_map.get(turnpolicy, turnpolicy)
                alphamax_value_to_use = 1.34 if zero_sharp_corners else corner_threshold
                scale = 1.0

                bm = potrace.Bitmap(binary_np)
                plist = bm.trace(
                    turdsize=turdsize_int,
                    turnpolicy=policy_arg,
                    alphamax=alphamax_value_to_use,
                    opticurve=optimize_curve,
                    opttolerance=opttolerance
                )

                scaled_width = max(1, round(orig_width * scale))
                scaled_height = max(1, round(orig_height * scale))
                svg_header = f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{scaled_width}" height="{scaled_height}" viewBox="0 0 {scaled_width} {scaled_height}">'
                svg_footer = "</svg>"
                background_rect = ""
                bg_color_lower = background_color.lower()

                if bg_color_lower != "none" and bg_color_lower != "":
                    background_rect = f'<rect width="100%" height="100%" fill="{background_color}"/>'

                scaled_stroke_width = stroke_width * scale
                stroke_attr = f'stroke="{stroke_color}" stroke-width="{scaled_stroke_width}"' if scaled_stroke_width > 0 and stroke_color.lower() != "none" else 'stroke="none"'
                fill_attr = f'fill="{foreground_color}"' if foreground_color.lower() != "none" else 'fill="none"'
                if fill_attr == 'fill="none"' and stroke_attr == 'stroke="none"':
                    fill_attr = 'fill="black"'

                all_paths_svg_parts = []
                if plist:
                    fill_rule_to_use = "evenodd"
                    for curve in plist:
                        if not (hasattr(curve, 'start_point') and hasattr(curve.start_point, 'x') and hasattr(curve.start_point, 'y')):
                            continue
                        fs = curve.start_point
                        all_paths_svg_parts.append(f"M{fs.x * scale:.2f},{fs.y * scale:.2f}")

                        if not hasattr(curve, 'segments'):
                            continue
                        for segment in curve.segments:
                            valid_segment = True
                            if not (hasattr(segment, 'is_corner') and hasattr(segment, 'end_point') and hasattr(segment.end_point, 'x') and hasattr(segment.end_point, 'y')):
                                valid_segment = False

                            if valid_segment and segment.is_corner:
                                if not (hasattr(segment, 'c') and hasattr(segment.c, 'x') and hasattr(segment.c, 'y')):
                                    valid_segment = False
                                else:
                                    c_x = segment.c.x * scale
                                    c_y = segment.c.y * scale
                                    ep_x = segment.end_point.x * scale
                                    ep_y = segment.end_point.y * scale
                                    all_paths_svg_parts.append(f"L{c_x:.2f},{c_y:.2f}L{ep_x:.2f},{ep_y:.2f}")
                            elif valid_segment:
                                if not (hasattr(segment, 'c1') and hasattr(segment.c1, 'x') and hasattr(segment.c1, 'y') and \
                                        hasattr(segment, 'c2') and hasattr(segment.c2, 'x') and hasattr(segment.c2, 'y')):
                                    valid_segment = False
                                else:
                                    c1_x = segment.c1.x * scale; c1_y = segment.c1.y * scale
                                    c2_x = segment.c2.x * scale; c2_y = segment.c2.y * scale
                                    ep_x = segment.end_point.x * scale; ep_y = segment.end_point.y * scale
                                    all_paths_svg_parts.append(f"C{c1_x:.2f},{c1_y:.2f} {c2_x:.2f},{c2_y:.2f} {ep_x:.2f},{ep_y:.2f}")
                        all_paths_svg_parts.append("Z")

                    if all_paths_svg_parts:
                        path_d_attribute = "".join(all_paths_svg_parts)
                        path_element = f'<path {stroke_attr} {fill_attr} fill-rule="{fill_rule_to_use}" d="{path_d_attribute}"/>'
                        svg_data_for_current_image = svg_header + background_rect + path_element + svg_footer
                    else:
                        svg_data_for_current_image = f'{svg_header}<desc>Potracer: Path data generation failed for image {i}</desc>{svg_footer}'
                else:
                    svg_data_for_current_image = f'{svg_header}<desc>Potracer: No paths found for image {i}</desc>{svg_footer}'

                batch_svg_strings.append(svg_data_for_current_image)

            except Exception as e:
                error_svg_content = f'<svg width="100" height="100"><desc>Error processing image {i}: {type(e).__name__} - {str(e).replace("<", "&lt;").replace(">", "&gt;")}</desc></svg>'
                batch_svg_strings.append(error_svg_content)
                continue

        output_string_joined = "\n".join(batch_svg_strings)

        return (output_string_joined,)



NODE_CLASS_MAPPINGS = {
    "TS_ImageQuantize": TS_ImageQuantize,
    "TS_ImageToSVGStringColor_Vtracer": TS_ImageToSVGStringColor_Vtracer,
    "TS_ImageToSVGStringBW_Vtracer": TS_ImageToSVGStringBW_Vtracer,
    "TS_SVGStringToImage": TS_SVGStringToImage,
    "TS_SaveSVGString": TS_SaveSVGString,
    "TS_SVGStringPreview": TS_SVGStringPreview,
    "TS_SVGStringToSVGBytesIO": TS_SVGStringToSVGBytesIO,
    "TS_SVGBytesIOToString": TS_SVGBytesIOToString,
    "TS_SVGPathSimplify": TS_SVGPathSimplify,
    "TS_ImageToSVGStringBW_Potracer": TS_ImageToSVGStringBW_Potracer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageQuantize": "Image Quantize",
    "TS_ImageToSVGStringColor_Vtracer": "Image to SVG String Color_Vtracer",
    "TS_ImageToSVGStringBW_Vtracer": "Image to SVG String BW_Vtracer",
    "TS_SVGStringToImage": "SVG String to Image",
    "TS_SaveSVGString": "Save SVG String",
    "TS_SVGStringPreview": "SVG String Preview",
    "TS_SVGStringToSVGBytesIO": "SVG String to SVG BytesIO",
    "TS_SVGBytesIOToString": "SVG BytesIO to SVG String",
    "TS_SVGPathSimplify": "SVG String Path Simplify",
    "TS_ImageToSVGStringBW_Potracer": "Image to SVG String BW_Potracer",
}