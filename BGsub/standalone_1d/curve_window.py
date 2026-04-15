#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curve_window.py — Standalone 1D curve processing main window (PySide6)
独立一维曲线处理主窗口（PySide6）

This window is the dedicated 1D application UI. It uses PySide6 +
matplotlib and delegates all processing to the shared core.
该窗口是独立 1D 程序的专用界面，使用 PySide6 + matplotlib，
并将所有处理委托给共享核心。

Shared-core imports / 共享核心导入:
    BGsub.core.curve_data      → ProcessMode, Curve1D, CurveMetadata
    BGsub.core.curve_processor → CurveProcessor, CurveProcessorConfig
    BGsub.core.task_pipeline   → PipelineEngine, PipelineContext, TaskItem
    BGsub.io.curve_io          → load/save, format detection
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QGroupBox,
    QFormLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QStackedWidget,
    QSplitter,
)

from BGsub.core.curve_data import Curve1D, CurveMetadata, ProcessMode
from BGsub.core.curve_processor import CurveProcessor, CurveProcessorConfig
from BGsub.core.task_pipeline import (
    PipelineEngine,
    PipelineContext,
    TaskItem,
    TaskResult,
    TaskStatus,
)
from BGsub.io.curve_io import (
    CURVE_EXTS,
    CurveColumnSpec,
    inspect_curve_layout,
    is_1d_curve_file,
    load_curve_file,
    load_curve_collection,
    save_curve_collection,
    save_curve_file,
)

from BGsub.standalone_1d.plot_canvas import CurveCanvas
from BGsub.standalone_1d.mode_panels import (
    Morph1DPanel,
    Fit1DPanel,
    TBGSubtractPanel,
    PhysFitPanel,
    MODE_PANEL_MAP,
)
from BGsub.utils.common import build_transmission_map, IonMatchResult


class LogWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(100)

    def append_log(self, text: str, level: str = "info") -> None:
        color_map = {
            "info": "#1565C0",
            "ok": "#2E7D32",
            "warn": "#EF6C00",
            "err": "#C62828",
            "dim": "#616161",
        }
        color = color_map.get(level, color_map["info"])
        self.append(f'<span style="color:{color};">{text}</span>')


class CurveWindow(QMainWindow):
    """
    Standalone 1D curve processing window.
    独立一维曲线处理窗口。

    Three processing modes:
        1. MORPH_1D       — 1D curve → morphological bg estimate / 形态学背景估计
        2. FIT_1D         — 1D curve → polynomial fit bg estimate / 拟合背景估计
        3. T_BG_SUBTRACT  — sample/(T/100)-background / 透过率修正参考扣除
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BGsub 1D — 独立一维曲线处理 / Standalone 1D Curve Processor")
        self.resize(1100, 750)

        self._mode = ProcessMode.T_BG_SUBTRACT
        self._paths: List[str] = []
        self._last_results: List[TaskResult] = []
        self._current_curve: Optional[Curve1D] = None
        self._is_running = False
        self._separate_background_path: Optional[str] = None
        self._ion_paths: List[str] = []

        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction / UI 构建
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)

        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self._build_mode_group())
        left_layout.addWidget(self._build_batch_group())
        left_layout.addWidget(self._build_run_group())
        left_layout.addStretch()
        left_scroll.setWidget(left_content)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_plot_group())
        right_layout.addWidget(self._build_log_group(), stretch=0)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        root_layout.addWidget(splitter)
        self._update_output_target_ui()
        self._refresh_background_combo()

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("① 输入选择 / Input Selection")
        layout = QVBoxLayout(group)

        btn_row = QHBoxLayout()
        self._btn_pick_files = QPushButton("选择文件")
        self._btn_pick_folder = QPushButton("选择文件夹")
        self._btn_remove_sel = QPushButton("删除选中")
        self._btn_clear = QPushButton("清空")
        self._btn_pick_files.clicked.connect(self._pick_files)
        self._btn_pick_folder.clicked.connect(self._pick_folder)
        self._btn_remove_sel.clicked.connect(self._remove_selected)
        self._btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(self._btn_pick_files)
        btn_row.addWidget(self._btn_pick_folder)
        btn_row.addWidget(self._btn_remove_sel)
        btn_row.addWidget(self._btn_clear)
        self._count_label = QLabel("0 个文件")
        btn_row.addWidget(self._count_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._file_list.itemDoubleClicked.connect(self._preview_selected)
        layout.addWidget(self._file_list)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("背景文件 / Background:"))
        self._background_combo = QComboBox()
        self._background_combo.currentIndexChanged.connect(lambda _: self._refresh_t_mode_context())
        bg_row.addWidget(self._background_combo, 1)
        self._btn_pick_background = QPushButton("加载独立背景...")
        self._btn_pick_background.clicked.connect(self._pick_background_file)
        bg_row.addWidget(self._btn_pick_background)
        layout.addLayout(bg_row)

        parse_form = QFormLayout()
        self._parse_mode_combo = QComboBox()
        self._parse_mode_combo.addItems(
            [
                "单曲线 XY / Single XY",
                "XYXY 多曲线 / XYXY",
                "XYYY 多曲线 / XYYY",
            ]
        )
        self._parse_mode_combo.currentIndexChanged.connect(self._refresh_t_mode_context)
        parse_form.addRow("列结构 / Column layout", self._parse_mode_combo)

        self._skip_header_spin = QSpinBox()
        self._skip_header_spin.setRange(0, 200)
        self._skip_header_spin.setValue(0)
        parse_form.addRow("跳过头部 / Skip header", self._skip_header_spin)

        self._delimiter_combo = QComboBox()
        self._delimiter_combo.addItems(
            ["自动空白 / Auto whitespace", "逗号 / Comma", "Tab", "空格 / Space"]
        )
        parse_form.addRow("分隔符 / Delimiter", self._delimiter_combo)

        self._parse_hint_label = QLabel("默认按两列 XY 读取；多列文件可切换为 XYXY 或 XYYY。")
        self._parse_hint_label.setWordWrap(True)
        parse_form.addRow("提示 / Hint", self._parse_hint_label)
        layout.addLayout(parse_form)

        return group

    def _build_mode_group(self) -> QGroupBox:
        group = QGroupBox("② 处理模式 / Processing Mode")
        layout = QVBoxLayout(group)

        mode_row = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        mode_labels = [
            ("T-背景扣除 / T-BG Subtract", ProcessMode.T_BG_SUBTRACT),
            ("形态学处理", ProcessMode.MORPH_1D),
            ("拟合处理", ProcessMode.FIT_1D),
            ("物理拟合", ProcessMode.PHYS_FIT),
        ]
        for idx, (text, mode) in enumerate(mode_labels):
            btn = QRadioButton(text)
            btn.setChecked(mode == ProcessMode.T_BG_SUBTRACT)
            self._mode_group.addButton(btn, idx)
            mode_row.addWidget(btn)
        mode_row.addStretch()
        self._mode_group.idToggled.connect(self._on_mode_toggled)
        layout.addLayout(mode_row)

        self._mode_stack = QStackedWidget()
        self._mode_panels = {
            ProcessMode.MORPH_1D: Morph1DPanel(),
            ProcessMode.FIT_1D: Fit1DPanel(),
            ProcessMode.T_BG_SUBTRACT: TBGSubtractPanel(),
            ProcessMode.PHYS_FIT: PhysFitPanel(),
        }
        mode_order = [
            ProcessMode.T_BG_SUBTRACT,
            ProcessMode.MORPH_1D,
            ProcessMode.FIT_1D,
            ProcessMode.PHYS_FIT,
        ]
        self._mode_index_map = {}
        for i, m in enumerate(mode_order):
            self._mode_stack.addWidget(self._mode_panels[m])
            self._mode_index_map[m] = i

        t_panel = self._mode_panels[ProcessMode.T_BG_SUBTRACT]
        t_panel.pick_ion_files_btn.clicked.connect(self._pick_ion_files)
        t_panel.pick_ion_folder_btn.clicked.connect(self._pick_ion_folder)
        t_panel.clear_ion_btn.clicked.connect(self._clear_ion_files)
        layout.addWidget(self._mode_stack)

        return group

    def _build_batch_group(self) -> QGroupBox:
        group = QGroupBox("③ 批处理 / Batch")
        layout = QFormLayout(group)

        self._memory_friendly = QCheckBox("逐文件流式处理 / Stream per file")
        self._memory_friendly.setChecked(True)

        self._preview_limit = QSpinBox()
        self._preview_limit.setRange(1, 1000)
        self._preview_limit.setValue(10)

        self._output_fmt_combo = QComboBox()
        self._output_fmt_combo.addItems(["xy", "csv", "txt", "gr", "npy", "h5"])

        self._output_mode_combo = QComboBox()
        self._output_mode_combo.addItems(["逐一文件 / Per-file", "合并单文件 / Merged"])
        self._output_mode_combo.currentIndexChanged.connect(self._update_output_target_ui)

        self._export_raw = QCheckBox("导出原始曲线")
        self._export_raw.setChecked(True)
        self._export_bg = QCheckBox("导出背景")
        self._export_sub = QCheckBox("导出扣除后曲线")
        self._export_sub.setChecked(True)

        self._save_results = QCheckBox("处理后自动保存 / Auto save")

        out_row = QHBoxLayout()
        self._output_target = QLineEdit()
        browse_out = QPushButton("选择目录")
        browse_out.clicked.connect(self._pick_output_target)
        self._browse_output_btn = browse_out
        out_row.addWidget(self._output_target)
        out_row.addWidget(browse_out)

        layout.addRow(self._memory_friendly)
        layout.addRow("预览数量 / Preview limit", self._preview_limit)
        layout.addRow("输出格式 / Output format", self._output_fmt_combo)
        layout.addRow("输出模式 / Output mode", self._output_mode_combo)
        layout.addRow(self._export_raw)
        layout.addRow(self._export_bg)
        layout.addRow(self._export_sub)
        layout.addRow(self._save_results)
        layout.addRow("输出目标 / Output target", out_row)

        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("④ 执行 / Run")
        layout = QVBoxLayout(group)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("开始处理")
        self._run_btn.clicked.connect(self._run)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setEnabled(False)
        self._progress = QProgressBar()
        self._status = QLabel("就绪 / Ready")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._stop_btn)
        run_row.addWidget(self._progress, 1)
        run_row.addWidget(self._status)
        layout.addLayout(run_row)

        self._result_table = QTableWidget()
        self._result_table.setColumnCount(4)
        self._result_table.setHorizontalHeaderLabels(["文件名", "状态", "质量", "耗时"])
        self._result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._result_table.setAlternatingRowColors(True)
        self._result_table.setMaximumHeight(150)
        self._result_table.itemDoubleClicked.connect(self._preview_result)
        layout.addWidget(self._result_table)

        return group

    def _build_plot_group(self) -> QGroupBox:
        group = QGroupBox("⑤ 曲线预览 / Curve Preview")
        layout = QVBoxLayout(group)

        toolbar = QHBoxLayout()
        self._plot_reset_btn = QPushButton("重置视图")
        self._plot_reset_btn.clicked.connect(lambda: self._canvas.reset_zoom())
        self._plot_export_btn = QPushButton("导出图像")
        self._plot_export_btn.clicked.connect(self._export_plot)

        self._show_raw = QCheckBox("原始")
        self._show_raw.setChecked(True)
        self._show_raw.toggled.connect(self._update_plot)
        self._show_bg = QCheckBox("背景")
        self._show_bg.setChecked(True)
        self._show_bg.toggled.connect(self._update_plot)
        self._show_sub = QCheckBox("扣除后")
        self._show_sub.setChecked(True)
        self._show_sub.toggled.connect(self._update_plot)

        toolbar.addWidget(self._plot_reset_btn)
        toolbar.addWidget(self._plot_export_btn)
        toolbar.addWidget(self._show_raw)
        toolbar.addWidget(self._show_bg)
        toolbar.addWidget(self._show_sub)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._canvas = CurveCanvas()
        layout.addWidget(self._canvas)

        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("日志 / Log")
        layout = QVBoxLayout(group)
        self._log = LogWidget()
        layout.addWidget(self._log)
        return group

    # ------------------------------------------------------------------
    # Slot helpers / 槽函数辅助
    # ------------------------------------------------------------------

    def _on_mode_toggled(self, button_id: int, checked: bool) -> None:
        if not checked:
            return
        mode_order = [
            ProcessMode.T_BG_SUBTRACT,
            ProcessMode.MORPH_1D,
            ProcessMode.FIT_1D,
            ProcessMode.PHYS_FIT,
        ]
        self._mode = mode_order[button_id]
        self._mode_stack.setCurrentIndex(button_id)
        self._refresh_background_combo()

    def _is_merged_output_mode(self) -> bool:
        return self._output_mode_combo.currentIndex() == 1

    def _parse_mode_key(self) -> str:
        index = self._parse_mode_combo.currentIndex()
        if index == 1:
            return "xyxy"
        if index == 2:
            return "xyyy"
        return "single"

    def _delimiter_value(self) -> Optional[str]:
        index = self._delimiter_combo.currentIndex()
        if index == 1:
            return "comma"
        if index == 2:
            return "tab"
        if index == 3:
            return "space"
        return None

    def _update_output_target_ui(self) -> None:
        if self._is_merged_output_mode():
            self._browse_output_btn.setText("选择文件")
            if not self._output_target.text().strip():
                suffix = self._output_fmt_combo.currentText().lower()
                self._output_target.setText(os.path.join(os.getcwd(), f"merged_curves.{suffix}"))
        else:
            self._browse_output_btn.setText("选择目录")

    def _pick_output_target(self) -> None:
        if self._is_merged_output_mode():
            fmt = self._output_fmt_combo.currentText().lower()
            path, _ = QFileDialog.getSaveFileName(
                self,
                "选择合并输出文件",
                "",
                f"Output (*.{fmt});;All Files (*)",
            )
            if path:
                self._output_target.setText(path)
            return

        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self._output_target.setText(path)

    def _pick_files(self) -> None:
        patterns = "曲线 (*.txt *.csv *.dat *.xy *.gr);;All Files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", patterns)
        if paths:
            self._add_paths(paths)

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder:
            return
        found = []
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in CURVE_EXTS:
                    found.append(os.path.join(root, name))
        self._add_paths(found)

    def _add_paths(self, paths: List[str]) -> None:
        for path in paths:
            if is_1d_curve_file(path) and path not in self._paths:
                self._paths.append(path)
        self._file_list.clear()
        self._file_list.addItems([os.path.basename(p) for p in self._paths])
        self._count_label.setText(f"{len(self._paths)} 个文件")
        self._refresh_background_combo()
        self._refresh_t_mode_context()
        self._log.append_log(f"已加载 {len(self._paths)} 个文件", "ok")

    def _sample_paths_for_t_mode(self) -> List[str]:
        background_path = self._background_combo.currentData()
        return [path for path in self._paths if path != background_path]

    def _column_specs_for_path(self, path: str) -> List[CurveColumnSpec]:
        layout = inspect_curve_layout(
            path=path,
            parse_mode=self._parse_mode_key(),
            skip_header=self._skip_header_spin.value(),
            delimiter=self._delimiter_value(),
        )
        return list(layout["curve_specs"])

    def _refresh_t_mode_context(self) -> None:
        parse_mode = self._parse_mode_key()
        if parse_mode == "xyxy":
            self._parse_hint_label.setText(
                "当前按 XYXY 读取：第 1/2 列、第 3/4 列……分别作为独立曲线。"
            )
        elif parse_mode == "xyyy":
            self._parse_hint_label.setText(
                "当前按 XYYY 读取：第 1 列作为 x，其余每列作为独立 y 曲线。"
            )
        else:
            self._parse_hint_label.setText("默认按两列 XY 读取；多列文件可切换为 XYXY 或 XYYY。")
        if self._mode != ProcessMode.T_BG_SUBTRACT:
            return
        t_panel = self._mode_panels[ProcessMode.T_BG_SUBTRACT]
        t_panel.set_sample_paths(self._sample_paths_for_t_mode())
        t_panel.ion_label.setText(
            f"已选 {len(self._ion_paths)} 个 / {len(self._ion_paths)} selected"
            if self._ion_paths
            else "未选择 / Not selected"
        )

    def _refresh_background_combo(self) -> None:
        current_path = self._background_combo.currentData()
        self._background_combo.clear()
        is_t_mode = self._mode == ProcessMode.T_BG_SUBTRACT
        self._background_combo.setEnabled(is_t_mode)
        self._btn_pick_background.setEnabled(is_t_mode)
        if not is_t_mode:
            return

        self._background_combo.addItem("请选择背景曲线 / Select background curve", None)
        for path in self._paths:
            self._background_combo.addItem(os.path.basename(path), path)

        if self._separate_background_path:
            self._background_combo.addItem(
                f"独立背景: {os.path.basename(self._separate_background_path)}",
                self._separate_background_path,
            )

        if current_path is not None:
            current_index = self._background_combo.findData(current_path)
            if current_index >= 0:
                self._background_combo.setCurrentIndex(current_index)
                return

        if self._separate_background_path:
            separate_index = self._background_combo.findData(self._separate_background_path)
            if separate_index >= 0:
                self._background_combo.setCurrentIndex(separate_index)

        self._refresh_t_mode_context()

    def _pick_background_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择独立背景曲线",
            "",
            "曲线 (*.txt *.csv *.dat *.xy *.gr);;All Files (*)",
        )
        if not path:
            return
        if not is_1d_curve_file(path):
            QMessageBox.warning(self, "提示", "所选文件不是支持的 1D 曲线格式")
            return
        self._separate_background_path = path
        self._refresh_background_combo()
        idx = self._background_combo.findData(path)
        if idx >= 0:
            self._background_combo.setCurrentIndex(idx)
        self._refresh_t_mode_context()
        self._log.append_log(f"已加载独立背景: {os.path.basename(path)}", "ok")

    def _pick_ion_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择电离室文件",
            "",
            "电离室 (*.ionchamber *.txt);;All Files (*)",
        )
        if not paths:
            return
        for path in paths:
            if path not in self._ion_paths:
                self._ion_paths.append(path)
        self._refresh_t_mode_context()
        self._log.append_log(f"已加载 {len(self._ion_paths)} 个电离室文件", "ok")

    def _pick_ion_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择电离室文件夹")
        if not folder:
            return
        found = []
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                low_name = name.lower()
                if low_name.endswith(".ionchamber") or low_name.endswith(".txt"):
                    path = os.path.join(root, name)
                    if path not in self._ion_paths:
                        self._ion_paths.append(path)
                        found.append(path)
        self._refresh_t_mode_context()
        self._log.append_log(f"导入电离室文件夹: 新增 {len(found)} 个文件", "ok")

    def _clear_ion_files(self) -> None:
        self._ion_paths.clear()
        if self._mode == ProcessMode.T_BG_SUBTRACT:
            self._mode_panels[ProcessMode.T_BG_SUBTRACT].set_ion_summary(
                "运行时自动匹配样品/背景电离室并计算逐文件 T。"
            )
        self._refresh_t_mode_context()
        self._log.append_log("已清空电离室文件列表", "dim")

    def _remove_selected(self) -> None:
        """Remove selected files from the list. 删除列表中选中的文件。"""
        if self._is_running:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "提示", "处理正在运行中，请先停止")
            return
        selected = self._file_list.selectedItems()
        if not selected:
            return
        rows = sorted([self._file_list.row(item) for item in selected], reverse=True)
        for row in rows:
            if 0 <= row < len(self._paths):
                removed = self._paths.pop(row)
                self._log.append_log(f"已移除: {os.path.basename(removed)}", "dim")
        self._file_list.clear()
        self._file_list.addItems([os.path.basename(p) for p in self._paths])
        self._count_label.setText(f"{len(self._paths)} 个文件")
        self._refresh_background_combo()
        self._refresh_t_mode_context()

    def _clear(self) -> None:
        if self._is_running:
            QMessageBox.warning(self, "提示", "处理正在运行中，请先停止")
            return
        self._paths.clear()
        self._last_results.clear()
        self._current_curve = None
        self._separate_background_path = None
        self._ion_paths.clear()
        self._file_list.clear()
        self._background_combo.clear()
        self._count_label.setText("0 个文件")
        self._result_table.setRowCount(0)
        self._canvas.clear()
        self._mode_panels[ProcessMode.T_BG_SUBTRACT].set_ion_summary(
            "运行时自动匹配样品/背景电离室并计算逐文件 T。"
        )
        self._log.append_log("已清空输入", "dim")

    def _preview_selected(self, item) -> None:
        row = self._file_list.row(item)
        if row < 0 or row >= len(self._paths):
            return
        path = self._paths[row]
        curves = load_curve_collection(
            path,
            parse_mode=self._parse_mode_key(),
            skip_header=self._skip_header_spin.value(),
            delimiter=self._delimiter_value(),
        )
        if curves:
            preview_curve = curves[0]
            self._show_curve(preview_curve, os.path.basename(path))
            self._log.append_log(
                f"预览: {os.path.basename(path)} | 检测到 {len(curves)} 条逻辑曲线", "info"
            )

    # ------------------------------------------------------------------
    # Config / task building / 配置与任务构建
    # ------------------------------------------------------------------

    def _make_config(self) -> CurveProcessorConfig:
        config = CurveProcessorConfig()
        panel = self._mode_panels[self._mode]
        panel.apply_to_config(config)
        return config

    def _resolve_transmission_settings(
        self,
        background_path: Optional[str],
    ) -> Tuple[List[str], Optional[float], Optional[Dict[str, float]]]:
        if self._mode != ProcessMode.T_BG_SUBTRACT:
            return list(self._paths), None, None

        t_panel = self._mode_panels[ProcessMode.T_BG_SUBTRACT]
        sample_paths = self._sample_paths_for_t_mode()
        if self._parse_mode_key() != "single" and t_panel.transmission_source() == "ionchamber":
            raise ValueError("多列 XYXY/XYYY 模式暂不支持 T-背景扣除中的电离室自动匹配")
        if t_panel.transmission_source() == "manual":
            if t_panel.manual_mode() == "unified":
                return sample_paths, t_panel.transmission_spin.value(), None
            transmissions = t_panel.get_manual_transmissions()
            missing = [os.path.basename(path) for path in sample_paths if path not in transmissions]
            if missing:
                raise ValueError(
                    "分别设置透过率不完整，请为以下样品显式填写 T：" + ", ".join(missing)
                )
            return sample_paths, None, transmissions

        if not background_path:
            raise ValueError("电离室模式下请先选择背景曲线")
        if not self._ion_paths:
            raise ValueError("电离室模式下请先选择电离室文件")

        transmissions, match_results, background_result = build_transmission_map(
            sample_paths=sample_paths,
            background_path=background_path,
            ion_paths=self._ion_paths,
            background_channel=t_panel.bg_channel_combo.currentText(),
            background_method=t_panel.bg_method_combo.currentText(),
            sample_channel=t_panel.sample_channel_combo.currentText(),
            sample_method=t_panel.sample_method_combo.currentText(),
            user_regex=t_panel.user_regex() or None,
        )
        if background_result is None or not background_result.success:
            message = background_result.error_message if background_result else "背景电离室匹配失败"
            raise ValueError(message)

        lines = [
            f"背景: {background_result.sample_name} → {os.path.basename(background_result.ion_path or '')}"
            f" | score={background_result.score:.3f}"
        ]
        failed_names: List[str] = []
        for result in match_results:
            if result.success and result.transmission_percent is not None:
                lines.append(
                    f"✓ {result.sample_name}: T={result.transmission_percent:.2f}%"
                    f" | {os.path.basename(result.ion_path or '')}"
                )
            else:
                failed_names.append(result.sample_name)
                lines.append(f"✗ {result.sample_name}: {result.error_message or '透过率计算失败'}")

        t_panel.set_ion_summary("\n".join(lines[:8]))
        for line in lines:
            self._log.append_log(
                line, "ok" if line.startswith("✓") or line.startswith("背景:") else "warn"
            )

        if failed_names:
            self._log.append_log(
                f"电离室模式下有 {len(failed_names)} 个样品未获得有效 T，将跳过这些样品",
                "warn",
            )
        if not transmissions:
            raise ValueError("没有样品获得有效透过率")
        valid_sample_paths = [path for path in sample_paths if path in transmissions]
        return valid_sample_paths, None, transmissions

    def _build_tasks(self) -> List[TaskItem]:
        background_path = None
        transmission = None
        transmissions = None
        task_paths = list(self._paths)
        if self._mode == ProcessMode.T_BG_SUBTRACT:
            background_path = self._background_combo.currentData()
            task_paths, transmission, transmissions = self._resolve_transmission_settings(
                background_path
            )

        tasks: List[TaskItem] = []
        skip_header = self._skip_header_spin.value()
        delimiter = self._delimiter_value()
        parse_mode = self._parse_mode_key()
        for path in task_paths:
            column_specs = self._column_specs_for_path(path)
            for spec_index, spec in enumerate(column_specs, start=1):
                suffix = spec.label if parse_mode != "single" else ""
                task = TaskItem(
                    source_path=path,
                    background_path=background_path,
                    transmission=(
                        transmissions.get(path)
                        if transmissions is not None and path in transmissions
                        else transmission
                    ),
                    process_mode=self._mode,
                    x_column=spec.x_column,
                    y_column=spec.y_column,
                    skip_header=skip_header,
                    delimiter=delimiter,
                    extra={
                        "curve_label": spec.label,
                        "output_suffix": suffix,
                        "parse_mode": parse_mode,
                        "curve_index": spec_index,
                    },
                )
                tasks.append(task)
        return tasks

    def _save_merged_results(self, output_target: str) -> Optional[str]:
        completed_curves = [
            result.curve
            for result in self._last_results
            if result.status == TaskStatus.COMPLETED and result.curve is not None
        ]
        if not completed_curves:
            return None
        fmt = self._output_fmt_combo.currentText().lower()
        ok = save_curve_collection(
            curves=completed_curves,
            path=output_target,
            fmt=fmt,
            include_bg=self._export_bg.isChecked(),
            include_subtracted=self._export_sub.isChecked(),
        )
        return output_target if ok else None

    def _quality_text(self, curve: Optional[Curve1D]) -> str:
        if curve is None:
            return "-"
        warnings = curve.metadata.extra.get("quality_warnings") or []
        bg_ratio = curve.metadata.extra.get("quality_bg_ratio")
        if warnings:
            return "警告"
        if bg_ratio is None:
            return "正常"
        return f"bg={float(bg_ratio):.2f}"

    def _log_quality_warnings(self, filename: str, curve: Optional[Curve1D]) -> None:
        if curve is None:
            return
        warnings = curve.metadata.extra.get("quality_warnings") or []
        if warnings:
            self._log.append_log(
                f"{filename} 检测到处理告警: {', '.join(str(item) for item in warnings)}",
                "warn",
            )

    # ------------------------------------------------------------------
    # Run pipeline / 运行管线
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not self._paths:
            QMessageBox.warning(self, "提示", "请先加载输入文件")
            return

        if self._mode == ProcessMode.T_BG_SUBTRACT and not self._background_combo.currentData():
            QMessageBox.warning(self, "提示", "T-背景扣除模式下请先选择背景曲线")
            return
        if self._mode == ProcessMode.T_BG_SUBTRACT and not self._sample_paths_for_t_mode():
            QMessageBox.warning(self, "提示", "请选择至少一条样品曲线（背景曲线不会作为样品处理）")
            return

        output_target = self._output_target.text().strip()
        if self._save_results.isChecked() and not output_target:
            QMessageBox.warning(self, "提示", "启用自动保存时请指定输出目标")
            return
        if output_target and not self._is_merged_output_mode():
            os.makedirs(output_target, exist_ok=True)

        try:
            tasks = self._build_tasks()
        except Exception as exc:
            QMessageBox.warning(self, "透过率配置无效", str(exc))
            self._log.append_log(f"透过率配置失败: {exc}", "err")
            return
        self._last_results.clear()
        self._result_table.setRowCount(0)
        self._progress.setRange(0, max(1, len(tasks)))
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._is_running = True

        output_fmt = self._output_fmt_combo.currentText().lower()
        ctx = PipelineContext(
            processor_config=self._make_config(),
            output_dir="" if self._is_merged_output_mode() else output_target,
            save_results=self._save_results.isChecked(),
            output_format=output_fmt,
            output_mode="merged" if self._is_merged_output_mode() else "per_file",
            merged_output_path=output_target if self._is_merged_output_mode() else "",
            export_raw=self._export_raw.isChecked(),
            export_bg=self._export_bg.isChecked(),
            export_sub=self._export_sub.isChecked(),
        )
        engine = PipelineEngine(ctx)

        t0 = time.time()
        preview_limit = self._preview_limit.value()

        for idx, result in enumerate(engine.run(tasks), start=1):
            if not self._is_running:
                break

            self._progress.setValue(idx)
            self._status.setText(f"{idx}/{len(tasks)}")
            QApplication.processEvents()
            self._last_results.append(result)

            row = self._result_table.rowCount()
            self._result_table.insertRow(row)
            filename = os.path.basename(result.source_path)
            self._result_table.setItem(row, 0, QTableWidgetItem(filename))

            elapsed = time.time() - t0
            if result.status.value == "completed" and result.curve is not None:
                self._result_table.setItem(row, 1, QTableWidgetItem("成功"))
                self._result_table.setItem(
                    row, 2, QTableWidgetItem(self._quality_text(result.curve))
                )
                self._log.append_log(f"完成: {filename}", "ok")
                self._log_quality_warnings(filename, result.curve)
                if idx <= preview_limit:
                    self._show_curve(result.curve, filename)
            else:
                self._result_table.setItem(row, 1, QTableWidgetItem("失败"))
                self._result_table.setItem(row, 2, QTableWidgetItem("失败"))
                self._log.append_log(f"失败: {filename} - {result.error_msg}", "err")

            self._result_table.setItem(row, 3, QTableWidgetItem(f"{elapsed:.2f}s"))

        finished_normally = self._is_running
        self._is_running = False
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.setText("完成" if finished_normally else "已停止")
        if (
            finished_normally
            and self._save_results.isChecked()
            and self._is_merged_output_mode()
            and output_target
        ):
            merged_path = self._save_merged_results(output_target)
            if merged_path:
                self._log.append_log(f"已保存合并结果: {merged_path}", "ok")
            else:
                self._log.append_log("合并结果保存失败", "err")

    def _stop(self) -> None:
        self._is_running = False
        self._status.setText("正在停止...")
        self._log.append_log("用户终止处理", "warn")

    # ------------------------------------------------------------------
    # Plot / 绘图
    # ------------------------------------------------------------------

    def _show_curve(self, curve: Curve1D, legend: str = "") -> None:
        self._current_curve = curve
        self._update_plot()

    def _update_plot(self) -> None:
        curve = self._current_curve
        if curve is None:
            return

        self._canvas.clear()
        meta = curve.metadata

        if self._show_raw.isChecked():
            self._canvas.plot_curve(curve.x, curve.y, label="raw", color="#1565C0")
        if self._show_bg.isChecked() and curve.background is not None:
            self._canvas.plot_curve(curve.x, curve.background, label="background", color="#EF6C00")
        if self._show_sub.isChecked() and curve.subtracted is not None:
            self._canvas.plot_curve(curve.x, curve.subtracted, label="processed", color="#2E7D32")

        self._canvas.set_labels(
            x_label=meta.x_label or "x",
            y_label=meta.y_label or "I",
        )
        warnings = meta.extra.get("quality_warnings") or []
        if warnings:
            self._canvas.set_labels(
                x_label=meta.x_label or "x",
                y_label=meta.y_label or "I",
                title=" | ".join(str(item) for item in warnings[:2]),
            )
        self._canvas.finalize()
        self._canvas.reset_zoom()

    def _preview_result(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if row < 0 or row >= len(self._last_results):
            return
        result = self._last_results[row]
        if result.status.value == "completed" and result.curve is not None:
            self._show_curve(result.curve, os.path.basename(result.source_path))
            self._log.append_log(f"预览: {os.path.basename(result.source_path)}", "info")

    def _export_plot(self) -> None:
        if self._current_curve is None:
            QMessageBox.warning(self, "提示", "当前没有可导出的曲线")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)"
        )
        if path:
            if self._canvas.save_figure(path):
                self._log.append_log(f"图像已保存到 {path}", "ok")
            else:
                self._log.append_log(f"保存失败: {path}", "err")
