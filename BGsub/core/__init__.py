#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/__init__.py — Core module
核心模块
"""

from BGsub.core.bg_fitter import BgFitter, RollingBallMethod
from BGsub.core.bg_math import (
    apply_mask,
    rolling_ball_background_subtraction,
    auto_detect_mask_thresholds,
)
from BGsub.core.curve_data import Curve1D, CurveMetadata, ProcessMode
from BGsub.core.curve_processor import (
    CurveProcessor,
    CurveProcessorConfig,
    morphological_background_1d,
    polynomial_background_1d,
    rolling_ball_background_1d,
)
from BGsub.core.task_pipeline import (
    PipelineEngine,
    PipelineContext,
    TaskItem,
    TaskResult,
    TaskStatus,
)

__all__ = [
    "BgFitter",
    "RollingBallMethod",
    "apply_mask",
    "rolling_ball_background_subtraction",
    "auto_detect_mask_thresholds",
    "Curve1D",
    "CurveMetadata",
    "ProcessMode",
    "CurveProcessor",
    "CurveProcessorConfig",
    "morphological_background_1d",
    "polynomial_background_1d",
    "rolling_ball_background_1d",
    "PipelineEngine",
    "PipelineContext",
    "TaskItem",
    "TaskResult",
    "TaskStatus",
]