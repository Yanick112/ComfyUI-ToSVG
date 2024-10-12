from .svgnode import *

NODE_CLASS_MAPPINGS = {
    "ConvertRasterToVectorColor": ConvertRasterToVectorColor,
    "ConvertRasterToVectorBW": ConvertRasterToVectorBW,
    "ConvertVectorToRaster": ConvertVectorToRaster,
    "SaveSVG": SaveSVG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertRasterToVectorColor": "Raster to Vector (SVG)Color",
    "ConvertRasterToVectorBW": "Raster to Vector (SVG)BW",
    "ConvertVectorToRaster": "Vector to Raster (SVG)",
    "SaveSVG": "Save SVG"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
