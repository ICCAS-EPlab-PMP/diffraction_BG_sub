#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standalone_1d — Standalone 1D curve processing GUI (PySide6 + matplotlib)
独立一维曲线处理 GUI（PySide6 + matplotlib）

This sub-package provides a silx-free 1D curve processing application
that reuses the shared core from BGsub.core and BGsub.io.
本子包提供不依赖 silx 的一维曲线处理应用，
复用 BGsub.core 和 BGsub.io 中的共享核心。

Shared-core boundary / 共享核心边界:
    from BGsub.core.curve_data import Curve1D, CurveMetadata, ProcessMode
    from BGsub.core.curve_processor import CurveProcessor, CurveProcessorConfig
    from BGsub.core.task_pipeline import PipelineEngine, PipelineContext, TaskItem
    from BGsub.io.curve_io import (
        load_curve_file, save_curve_file,
        is_1d_curve_file,
        CURVE_EXTS,
    )

No silx imports anywhere in this package.
本包内没有任何 silx 导入。
"""
