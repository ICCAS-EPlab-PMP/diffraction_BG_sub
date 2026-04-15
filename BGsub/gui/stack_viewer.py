#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stack_viewer.py — StackView wrapper for multi-frame viewing
多帧查看器包装

Provides a convenient wrapper around silx's StackView for browsing
through image stacks (3D volumes).
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np

try:
    from silx.gui.plot.StackView import StackViewMainWindow
    HAS_SILX = True
except ImportError:
    HAS_SILX = False


class BgStackViewer(StackViewMainWindow if HAS_SILX else object):
    """
    Wrapper around silx StackViewMainWindow for BGsub.

    Provides convenient methods for loading and browsing image stacks.
    """

    def __init__(self, parent=None):
        if HAS_SILX:
            super().__init__(parent)

        self._stack_data: Optional[np.ndarray] = None
        self._current_frame: int = 0

    def set_stack(self, data: np.ndarray, perspective: int = 0) -> None:
        """
        Set the image stack to browse.
        设置要浏览的图像堆叠。

        Parameters
        ----------
        data : np.ndarray
            3D array of shape (N, H, W) or (H, W, N) depending on perspective
            三维数组，形状为 (N, H, W) 或根据视角不同
        perspective : int
            Which dimension to use as frame index (0, 1, or 2)
            用作帧索引的维度
        """
        if not HAS_SILX:
            return

        if data.ndim != 3:
            raise ValueError(f"Stack data must be 3D, got {data.ndim}D")

        self._stack_data = data
        super().setStack(data, perspective=perspective)

    def get_current_frame(self) -> int:
        """
        Get the current frame number.
        获取当前帧编号。

        Returns
        -------
        int
            Current frame index / 当前帧索引
        """
        if HAS_SILX:
            return super().getFrameNumber()
        return self._current_frame

    def set_frame(self, index: int) -> None:
        """
        Set the current frame to display.
        设置当前显示的帧。

        Parameters
        ----------
        index : int
            Frame index / 帧索引
        """
        if HAS_SILX:
            super().setFrameNumber(index)
        self._current_frame = index

    def get_stack_size(self) -> Optional[List[int]]:
        """
        Get the dimensions of the loaded stack.
        获取加载的堆叠维度。

        Returns
        -------
        List[int] or None
            [n_frames, height, width] / [帧数, 高度, 宽度]
        """
        if self._stack_data is not None:
            return list(self._stack_data.shape)
        return None

    def get_frame_at_index(self, index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the stack.
        获取堆叠中指定帧。

        Parameters
        ----------
        index : int
            Frame index / 帧索引

        Returns
        -------
        np.ndarray or None
            2D frame data / 2D 帧数据
        """
        if self._stack_data is not None and 0 <= index < self._stack_data.shape[0]:
            return self._stack_data[index]
        return None

    def set_colormap(self, name: str = "viridis", vmin=None, vmax=None) -> None:
        """
        Set the colormap for the stack view.
        设置堆叠查看的色图。

        Parameters
        ----------
        name : str
            Colormap name / 色图名称
        vmin : float, optional
            Minimum value / 最小值
        vmax : float, optional
            Maximum value / 最大值
        """
        if not HAS_SILX:
            return

        kwargs = {"colormap": name}
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
        super().setColormap(**kwargs)

    def scale_colormap_to_stack(self) -> None:
        """Scale colormap range to current stack data."""
        if HAS_SILX:
            super().scaleColormapRangeToStack()
