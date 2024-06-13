from .svgnode import *

NODE_CLASS_MAPPINGS = {
    "ConvertRasterToVector": ConvertRasterToVector,
    "SaveSVG": SaveSVG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertRasterToVector": "Raster to Vector (SVG)",
    "SaveSVG": "Save SVG"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
