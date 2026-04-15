#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_widget.py — CompareImages wrapper widget
CompareImages 包装组件
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from silx.gui.plot.CompareImages import CompareImages

    HAS_SILX = True
except ImportError:
    CompareImages = None
    HAS_SILX = False


def _viz_mode(name: str):
    """Safely fetch silx visualization enum. 安全获取 silx 可视化枚举。"""
    if not HAS_SILX or CompareImages is None:
        return None
    viz_cls = getattr(CompareImages, "VisualizationMode", None)
    return getattr(viz_cls, name, None) if viz_cls is not None else None


class BgCompareWidget:
    """
    Lightweight wrapper around silx CompareImages.
    silx CompareImages 的轻量包装。
    """

    def __init__(self, parent=None):
        self._widget = CompareImages(parent) if HAS_SILX and CompareImages is not None else None
        self._original: Optional[np.ndarray] = None
        self._processed: Optional[np.ndarray] = None

    @property
    def widget(self):
        """Return inner silx widget when available. 返回内部 silx 控件。"""
        return self._widget

    def set_images(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        lock_aspect: bool = True,
    ) -> None:
        self._original = original
        self._processed = processed
        if original.shape != processed.shape:
            raise ValueError(f"Image shapes must match: {original.shape} vs {processed.shape}")
        if self._widget is None:
            return
        self._widget.setData(original, processed)
        if lock_aspect:
            plot = self._widget.getPlot()
            if plot is not None and hasattr(plot, "setKeepDataAspectRatio"):
                plot.setKeepDataAspectRatio(True)

    def set_visualization_mode(self, mode: str) -> None:
        enum_val = {
            "slide": _viz_mode("SLIDE"),
            "blend": _viz_mode("BLEND"),
            "difference": _viz_mode("DIFFERENCE"),
        }.get(mode)
        if self._widget is not None and enum_val is not None:
            self._widget.setVisualizationMode(enum_val)

    def get_visualization_mode_name(self) -> Optional[str]:
        if self._widget is None:
            return None
        current = self._widget.getVisualizationMode()
        for name in ("slide", "blend", "difference"):
            if current is _viz_mode(name.upper() if name != "difference" else "DIFFERENCE"):
                return name
        return None

    def get_original(self) -> Optional[np.ndarray]:
        return self._original

    def get_processed(self) -> Optional[np.ndarray]:
        return self._processed

    def get_image_shapes(self) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        s1 = self._original.shape if self._original is not None else None
        s2 = self._processed.shape if self._processed is not None else None
        return s1, s2

    @staticmethod
    def available_modes() -> Tuple[str, ...]:
        return ("slide", "blend", "difference")

    def reset_view(self) -> None:
        if self._widget is not None:
            self._widget.resetZoom()

    def clear_images(self) -> None:
        self._original = None
        self._processed = None
