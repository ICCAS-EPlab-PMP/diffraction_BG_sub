#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_canvas.py — Matplotlib canvas widget for PySide6
Matplotlib 画布组件，替代 silx.gui.plot.Plot1D
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout


class CurveCanvas(QWidget):
    """
    Matplotlib-based 1D curve plot widget for PySide6.
    基于 Matplotlib 的一维曲线绘图组件。
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._fig = Figure(tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._init_axes()

    def _init_axes(self) -> None:
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("Intensity")
        self._ax.set_title("1D Curve Preview")

    def clear(self) -> None:
        self._ax.clear()
        self._init_axes()
        self._canvas.draw_idle()

    def plot_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str = "",
        color: str = "#1565C0",
        linestyle: str = "-",
    ) -> None:
        self._ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=1.0)

    def finalize(self) -> None:
        if self._ax.get_legend_handles_labels()[1]:
            self._ax.legend(fontsize=8)
        self._canvas.draw_idle()

    def reset_zoom(self) -> None:
        self._ax.relim()
        self._ax.autoscale_view()
        self._canvas.draw_idle()

    def set_labels(self, x_label: str = "x", y_label: str = "Intensity", title: str = "") -> None:
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)
        if title:
            self._ax.set_title(title)

    def save_figure(self, path: str) -> bool:
        try:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            return True
        except Exception:
            return False


class MiniCurveCanvas(QWidget):
    """
    Lightweight compact matplotlib canvas without toolbar.
    轻量级紧凑型 matplotlib 画布（无工具栏），适用于面板内嵌预览。

    Fixed height (~160 px), small fonts/ticks, thin border feel.
    固定高度（约 160 px），小号字体/刻度，带细边框效果。
    """

    # 固定画布高度 / Fixed canvas height
    _FIXED_HEIGHT = 160

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(self._FIXED_HEIGHT)

        # 紧凑布局 / Compact layout
        self._fig = Figure(tight_layout=True, dpi=90)
        self._fig.patch.set_facecolor("#FAFAFA")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        layout.addWidget(self._canvas)

        # 紧凑边框 / Subtle border
        self.setStyleSheet(
            "MiniCurveCanvas {"
            "  border: 1px solid #D0D0D0;"
            "  border-radius: 3px;"
            "  background: #FAFAFA;"
            "}"
        )

        self._init_axes()

    def _init_axes(self) -> None:
        """Set compact default axis style. 设置紧凑默认坐标轴样式。"""
        self._ax.set_xlabel("x", fontsize=8)
        self._ax.set_ylabel("Intensity", fontsize=8)
        self._ax.set_title("", fontsize=9)
        self._ax.tick_params(labelsize=7)
        self._ax.grid(True, linewidth=0.3, alpha=0.5)

    def clear(self) -> None:
        """Clear axes and redraw. 清空坐标轴并重绘。"""
        self._ax.clear()
        self._init_axes()
        self._canvas.draw_idle()

    def plot_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str = "",
        color: str = "#1565C0",
        linestyle: str = "-",
    ) -> None:
        """Plot a single curve with compact styling. 绘制单条曲线（紧凑样式）。"""
        self._ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=1.2)

    def plot_fill(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: str = "#1565C0",
        alpha: float = 0.12,
    ) -> None:
        """Add a faint fill under the curve. 在曲线下方添加淡色填充。"""
        self._ax.fill_between(x, y, alpha=alpha, color=color)

    def set_labels(self, x_label: str = "x", y_label: str = "Intensity", title: str = "") -> None:
        """Set axis labels with compact font sizes. 设置坐标轴标签（小号字体）。"""
        self._ax.set_xlabel(x_label, fontsize=8)
        self._ax.set_ylabel(y_label, fontsize=8)
        if title:
            self._ax.set_title(title, fontsize=9)

    def finalize(self) -> None:
        """Apply legend (if any) and redraw. 添加图例（如有）并重绘。"""
        handles, labels = self._ax.get_legend_handles_labels()
        if labels:
            self._ax.legend(fontsize=7, loc="best")
        self._ax.tick_params(labelsize=7)
        self._canvas.draw_idle()
