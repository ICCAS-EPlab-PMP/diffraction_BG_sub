#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io/__init__.py — I/O module
输入输出模块
"""

from BGsub.io.image_io import (
    load_image_file,
    load_h5_stack,
    find_h5_transmissions,
    TIFF_EXTS,
    H5_EXTS,
)
from BGsub.io.curve_io import (
    load_curve_file,
    save_curve_file,
    save_curve_with_background,
    detect_curve_format,
    is_1d_curve_file,
    CURVE_EXTS,
)

__all__ = [
    "load_image_file",
    "load_h5_stack",
    "find_h5_transmissions",
    "TIFF_EXTS",
    "H5_EXTS",
    "load_curve_file",
    "save_curve_file",
    "save_curve_with_background",
    "detect_curve_format",
    "is_1d_curve_file",
    "CURVE_EXTS",
]
