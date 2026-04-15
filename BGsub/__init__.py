#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGsub — Background Subtraction Suite
背景扣除套件

A standalone desktop application for X-ray diffraction background subtraction.
支持两种模式:
  - 有参考背景扣除 (with reference background)
  - 无参考数学拟合 (mathematical fitting without reference)
"""

__version__ = "0.1.0"
__author__ = "Tianyi Ma; with Minimax 2.7 help"

from BGsub.core.bg_fitter import BgFitter
from BGsub.core.curve_processor import CurveProcessor, CurveProcessorConfig
from BGsub.io.image_io import load_image_file, load_h5_stack
from BGsub.io.curve_io import load_curve_file

__all__ = [
    "__version__",
    "BgFitter",
    "CurveProcessor",
    "CurveProcessorConfig",
    "load_image_file",
    "load_h5_stack",
    "load_curve_file",
]
