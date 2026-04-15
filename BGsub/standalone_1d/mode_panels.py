#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mode_panels.py — Processing-mode configuration panels
处理模式的参数配置面板
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from BGsub.core.curve_data import ProcessMode
from BGsub.core.curve_processor import CurveProcessorConfig
from BGsub.core.phys_fit import (
    energy_to_wavelength,
    simulate_phys_curve,
    validate_formula,
    wavelength_to_energy,
)
from BGsub.standalone_1d.plot_canvas import MiniCurveCanvas

# ---------------------------------------------------------------------------
# 形态学面板的中英文方法映射 / Chinese ↔ English method mapping
# ---------------------------------------------------------------------------
_METHOD_MAP = {
    "形态学开运算": "morph",
    "滚球法": "rolling_ball",
    "多项式拟合": "poly",
}


class Morph1DPanel(QWidget):
    """Mode 1: Morphological background estimation on 1D curve. 形态学背景估计。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QFormLayout(self)

        # 背景估计方法 / Background estimation method
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(list(_METHOD_MAP.keys()))
        self.bg_method_combo.setToolTip("选择背景估计算法。鼠标悬停在各选项上查看详细说明。")
        self.bg_method_combo.currentIndexChanged.connect(self._update_method_tooltip)
        layout.addRow("背景估计方法", self.bg_method_combo)

        # 结构元素半径 / Structuring element radius
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 5000)
        self.radius_spin.setValue(50)
        self.radius_spin.setToolTip("结构元素的半宽度（数据点数）。值越大背景越平滑。推荐 20~200。")
        layout.addRow("结构元素半径（点数）", self.radius_spin)

        # 迭代次数 / Iterations
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 20)
        self.iter_spin.setValue(1)
        self.iter_spin.setToolTip("重复开运算次数。多次迭代使背景更平滑。推荐 1~3。")
        layout.addRow("迭代次数", self.iter_spin)

        # 初始化 tooltip / Initialize tooltip
        self._update_method_tooltip(0)

    def _update_method_tooltip(self, index: int) -> None:
        """Update combo tooltip based on selected method. 根据选择更新提示。"""
        tooltips = {
            "形态学开运算": "先腐蚀后膨胀，平滑信号并提取背景轮廓。适合宽缓背景。",
            "滚球法": "在信号下方滚动虚拟球，球路径即为背景。适合快速估计。",
            "多项式拟合": "使用低阶多项式拟合数据下包络线。适合单调变化背景。",
        }
        text = self.bg_method_combo.currentText()
        tip = tooltips.get(text, "")
        self.bg_method_combo.setToolTip(
            tip + "\n选择背景估计算法。鼠标悬停在各选项上查看详细说明。"
        )

    def apply_to_config(self, config: CurveProcessorConfig) -> None:
        config.process_mode = ProcessMode.MORPH_1D
        config.morph_radius = self.radius_spin.value()
        config.morph_iterations = self.iter_spin.value()
        # 使用中文名映射到内部英文值 / Map Chinese display name to internal English value
        chinese_name = self.bg_method_combo.currentText()
        config.bg_method_1d = _METHOD_MAP.get(chinese_name, "morph")


class Fit1DPanel(QWidget):
    """Mode 2: Polynomial fitting background estimation on 1D curve. 多项式拟合背景。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QFormLayout(self)

        desc = QLabel('原理：选取低于分位阈值的"底部"数据点进行多项式拟合，拟合曲线作为背景估计。')
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 11px; padding: 4px;")
        layout.addRow(desc)

        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 20)
        self.degree_spin.setValue(4)
        self.degree_spin.setToolTip(
            "拟合多项式的最高阶数。低阶（3~5）适合平滑背景，高阶可拟合更复杂形状。"
        )
        layout.addRow("多项式阶数", self.degree_spin)

        self.quantile_spin = QDoubleSpinBox()
        self.quantile_spin.setRange(0.0, 1.0)
        self.quantile_spin.setSingleStep(0.05)
        self.quantile_spin.setValue(0.3)
        self.quantile_spin.setToolTip(
            "只使用强度低于该分位数的数据点进行拟合，避免峰干扰。推荐 0.2~0.5。"
        )
        layout.addRow("分位阈值", self.quantile_spin)

    def apply_to_config(self, config: CurveProcessorConfig) -> None:
        config.process_mode = ProcessMode.FIT_1D
        config.poly_degree = self.degree_spin.value()
        config.poly_quantile = self.quantile_spin.value()


class TBGSubtractPanel(QWidget):
    """Mode 3: Transmission-corrected background subtraction. 透过率修正背景扣除。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manual_transmissions: Dict[str, float] = {}
        self._manual_touched: Set[str] = set()
        self._sample_paths: List[str] = []

        root = QVBoxLayout(self)

        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("透过率来源 / Transmission source"))
        self.manual_radio = QRadioButton("手动输入 / Manual")
        self.ion_radio = QRadioButton("电离室文件 / Ionchamber")
        self.manual_radio.setChecked(True)
        self.manual_radio.toggled.connect(self._update_visibility)
        self.ion_radio.toggled.connect(self._update_visibility)
        source_layout.addWidget(self.manual_radio)
        source_layout.addWidget(self.ion_radio)
        source_layout.addStretch()
        root.addLayout(source_layout)

        self.manual_widget = QWidget()
        manual_layout = QVBoxLayout(self.manual_widget)
        manual_mode_layout = QHBoxLayout()
        self.unified_radio = QRadioButton("统一值 / Unified")
        self.per_file_radio = QRadioButton("分别设置 / Per-file")
        self.unified_radio.setChecked(True)
        self.unified_radio.toggled.connect(self._update_visibility)
        self.per_file_radio.toggled.connect(self._update_visibility)
        manual_mode_layout.addWidget(self.unified_radio)
        manual_mode_layout.addWidget(self.per_file_radio)
        manual_mode_layout.addStretch()
        manual_layout.addLayout(manual_mode_layout)

        self.unified_widget = QWidget()
        unified_layout = QFormLayout(self.unified_widget)
        self.transmission_spin = QDoubleSpinBox()
        self.transmission_spin.setRange(0.001, 10000.0)
        self.transmission_spin.setDecimals(6)
        self.transmission_spin.setSingleStep(0.1)
        self.transmission_spin.setValue(100.0)
        self.transmission_spin.setSuffix(" %")
        unified_layout.addRow("统一透过率 T / Unified transmission", self.transmission_spin)
        manual_layout.addWidget(self.unified_widget)

        self.per_file_scroll = QScrollArea()
        self.per_file_scroll.setWidgetResizable(True)
        self.per_file_scroll.setMinimumHeight(100)
        self.per_file_scroll.setMaximumHeight(220)
        self.per_file_content = QWidget()
        self.per_file_layout = QVBoxLayout(self.per_file_content)
        self.per_file_layout.addStretch()
        self.per_file_scroll.setWidget(self.per_file_content)
        manual_layout.addWidget(self.per_file_scroll)
        root.addWidget(self.manual_widget)

        self.ion_widget = QWidget()
        ion_layout = QVBoxLayout(self.ion_widget)
        ion_file_row = QHBoxLayout()
        self.pick_ion_files_btn = QPushButton("选择电离室文件")
        self.pick_ion_folder_btn = QPushButton("导入电离室文件夹")
        self.clear_ion_btn = QPushButton("清空电离室")
        self.ion_label = QLabel("未选择 / Not selected")
        ion_file_row.addWidget(self.pick_ion_files_btn)
        ion_file_row.addWidget(self.pick_ion_folder_btn)
        ion_file_row.addWidget(self.clear_ion_btn)
        ion_file_row.addWidget(self.ion_label)
        ion_file_row.addStretch()
        ion_layout.addLayout(ion_file_row)

        bg_row = QHBoxLayout()
        self.bg_channel_combo = QComboBox()
        self.bg_channel_combo.addItems(["Ionchamber0", "Ionchamber1", "Ionchamber2"])
        self.bg_channel_combo.setCurrentText("Ionchamber1")
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["mean", "median", "trimmed_mean"])
        self.bg_method_combo.setCurrentText("median")
        bg_row.addWidget(QLabel("背景通道 / Background channel"))
        bg_row.addWidget(self.bg_channel_combo)
        bg_row.addWidget(QLabel("方法 / Method"))
        bg_row.addWidget(self.bg_method_combo)
        bg_row.addStretch()
        ion_layout.addLayout(bg_row)

        sample_row = QHBoxLayout()
        self.sample_channel_combo = QComboBox()
        self.sample_channel_combo.addItems(["Ionchamber0", "Ionchamber1", "Ionchamber2"])
        self.sample_channel_combo.setCurrentText("Ionchamber1")
        self.sample_method_combo = QComboBox()
        self.sample_method_combo.addItems(["mean", "median", "trimmed_mean"])
        self.sample_method_combo.setCurrentText("median")
        sample_row.addWidget(QLabel("样品通道 / Sample channel"))
        sample_row.addWidget(self.sample_channel_combo)
        sample_row.addWidget(QLabel("方法 / Method"))
        sample_row.addWidget(self.sample_method_combo)
        sample_row.addStretch()
        ion_layout.addLayout(sample_row)

        regex_layout = QFormLayout()
        self.regex_combo = QComboBox()
        self.regex_combo.setEditable(True)
        self.regex_combo.addItem("")
        self.regex_combo.setCurrentText("")
        regex_layout.addRow("自定义正则 / Custom regex", self.regex_combo)
        ion_layout.addLayout(regex_layout)

        self.ion_summary_label = QLabel("运行时自动匹配样品/背景电离室并计算逐文件 T。")
        self.ion_summary_label.setWordWrap(True)
        ion_layout.addWidget(self.ion_summary_label)
        root.addWidget(self.ion_widget)

        self._update_visibility()

    def _update_visibility(self) -> None:
        use_manual = self.manual_radio.isChecked()
        use_unified = self.unified_radio.isChecked()
        self.manual_widget.setVisible(use_manual)
        self.ion_widget.setVisible(not use_manual)
        self.unified_widget.setVisible(use_manual and use_unified)
        self.per_file_scroll.setVisible(use_manual and not use_unified)

    def apply_to_config(self, config: CurveProcessorConfig) -> None:
        config.process_mode = ProcessMode.T_BG_SUBTRACT
        config.transmission = self.transmission_spin.value()

    def set_sample_paths(self, sample_paths: List[str]) -> None:
        self._sample_paths = list(sample_paths)
        self._refresh_per_file_rows()

    def _refresh_per_file_rows(self) -> None:
        while self.per_file_layout.count() > 1:
            item = self.per_file_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for sample_path in self._sample_paths:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(sample_path.split("\\")[-1].split("/")[-1])
            label.setMinimumWidth(220)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10000.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setSpecialValueText("未设置 / Unset")
            spin.setValue(self._manual_transmissions.get(sample_path, 0.0))
            spin.setSuffix(" %")
            spin.valueChanged.connect(
                lambda value, current_path=sample_path: self._on_manual_value_changed(
                    current_path,
                    value,
                )
            )
            row_layout.addWidget(label)
            row_layout.addWidget(spin)
            row_layout.addStretch()
            self.per_file_layout.insertWidget(self.per_file_layout.count() - 1, row_widget)

    def _on_manual_value_changed(self, sample_path: str, value: float) -> None:
        if value > 0:
            self._manual_transmissions[sample_path] = value
            self._manual_touched.add(sample_path)
        else:
            self._manual_transmissions.pop(sample_path, None)
            self._manual_touched.discard(sample_path)

    def transmission_source(self) -> str:
        return "manual" if self.manual_radio.isChecked() else "ionchamber"

    def manual_mode(self) -> str:
        return "unified" if self.unified_radio.isChecked() else "per-file"

    def get_manual_transmissions(self) -> Dict[str, float]:
        return {
            sample_path: self._manual_transmissions[sample_path]
            for sample_path in self._sample_paths
            if sample_path in self._manual_touched and sample_path in self._manual_transmissions
        }

    def user_regex(self) -> str:
        return self.regex_combo.currentText().strip()

    def set_ion_summary(self, text: str) -> None:
        self.ion_summary_label.setText(text)


class PhysFitPanel(QWidget):
    """Mode 4: Physics-based attenuation simulation. 物理衰减仿真背景。"""

    _KEV_ANGSTROM = 12.3984

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QFormLayout(self)

        desc = QLabel(
            "原理：基于空气吸收（角度相关）和可选样品吸收计算仿真衰减曲线，"
            "缩放后作为背景扣除。平面板探测器几何：更高角度 = 更长空气路径 = 更多吸收。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 11px; padding: 4px;")
        layout.addRow(desc)

        # 化学式 / Chemical formula
        formula_row = QHBoxLayout()
        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("如 C10H8O4, LaB6, C2H4")
        self.formula_edit.setToolTip(
            "输入化合物化学式。支持有机物（如 C10H8O4）和无机物（如 LaB6）。"
        )
        self.validate_btn = QPushButton("验证")
        self.validate_btn.setFixedWidth(50)
        self.validate_btn.clicked.connect(self._validate_formula)
        self.formula_result = QLabel("")
        self.formula_result.setStyleSheet("color: #888; font-size: 10px;")
        formula_row.addWidget(self.formula_edit, 1)
        formula_row.addWidget(self.validate_btn)
        formula_row.addWidget(self.formula_result)
        layout.addRow("化学式", formula_row)

        # 材料密度 / Material density
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.01, 50.0)
        self.density_spin.setSingleStep(0.1)
        self.density_spin.setValue(1.0)
        self.density_spin.setSuffix(" g/cm³")
        self.density_spin.setToolTip(
            "材料密度。常见值：PE≈0.95, PET≈1.38, 尼龙≈1.13, 水≈1.0 g/cm³。"
        )
        layout.addRow("材料密度", self.density_spin)

        # 样品厚度 / Sample thickness
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.01, 100.0)
        self.thickness_spin.setSingleStep(0.1)
        self.thickness_spin.setValue(1.0)
        self.thickness_spin.setSuffix(" mm")
        self.thickness_spin.setToolTip("片状样品的厚度（mm）。用于计算样品吸收。")
        layout.addRow("样品厚度", self.thickness_spin)

        # 样品-探测器距离 / Sample-detector distance
        self.sd_spin = QDoubleSpinBox()
        self.sd_spin.setRange(10.0, 10000.0)
        self.sd_spin.setSingleStep(10.0)
        self.sd_spin.setValue(200.0)
        self.sd_spin.setSuffix(" mm")
        self.sd_spin.setToolTip("样品到探测器的距离（mm）。SD 越大，空气吸收越强。")
        layout.addRow("样品-探测器距离 (SD)", self.sd_spin)

        # 能量 / 波长切换 / Energy / Wavelength toggle
        ew_row = QHBoxLayout()
        self._ew_group = QButtonGroup(self)
        self.energy_radio = QRadioButton("能量 (keV)")
        self.wavelength_radio = QRadioButton("波长 (Å)")
        self.energy_radio.setChecked(True)
        self._ew_group.addButton(self.energy_radio, 0)
        self._ew_group.addButton(self.wavelength_radio, 1)

        self.energy_spin = QDoubleSpinBox()
        self.energy_spin.setRange(1.0, 100.0)
        self.energy_spin.setSingleStep(0.1)
        self.energy_spin.setValue(12.7)
        self.energy_spin.setDecimals(4)
        self.energy_spin.setSuffix(" keV")

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(0.1, 10.0)
        self.wavelength_spin.setSingleStep(0.01)
        self.wavelength_spin.setValue(0.976)
        self.wavelength_spin.setDecimals(4)
        self.wavelength_spin.setSuffix(" Å")
        self.wavelength_spin.setVisible(False)

        self.energy_spin.setToolTip(
            "X 射线能量或波长。常用：Cu Kα=8.048 keV (1.5406 Å), "
            "Mo Kα=17.48 keV (0.7107 Å), 同步辐射通常已知。"
        )
        self.wavelength_spin.setToolTip(self.energy_spin.toolTip())

        self.energy_radio.toggled.connect(self._on_energy_radio_toggled)
        self.energy_spin.valueChanged.connect(self._sync_energy_to_wavelength)
        self.wavelength_spin.valueChanged.connect(self._sync_wavelength_to_energy)

        ew_row.addWidget(self.energy_radio)
        ew_row.addWidget(self.energy_spin)
        ew_row.addWidget(self.wavelength_radio)
        ew_row.addWidget(self.wavelength_spin)
        ew_row.addStretch()
        layout.addRow(ew_row)

        # 输出单位 / Output unit
        unit_row = QHBoxLayout()
        self._unit_group = QButtonGroup(self)
        self.unit_q_radio = QRadioButton("q (Å⁻¹)")
        self.unit_2t_radio = QRadioButton("2θ (°)")
        self.unit_q_radio.setChecked(True)
        self._unit_group.addButton(self.unit_q_radio, 0)
        self._unit_group.addButton(self.unit_2t_radio, 1)
        self.unit_q_radio.setToolTip("选择仿真曲线的横坐标单位。应与实验数据的单位一致。")
        self.unit_2t_radio.setToolTip(self.unit_q_radio.toolTip())
        unit_row.addWidget(self.unit_q_radio)
        unit_row.addWidget(self.unit_2t_radio)
        unit_row.addStretch()
        layout.addRow("输出单位", unit_row)

        # 样品吸收开关 / Sample absorption toggle
        self.sample_abs_check = QCheckBox("启用样品吸收（默认关闭）")
        self.sample_abs_check.setToolTip(
            "勾选后考虑样品自身对 X 射线的吸收衰减。默认关闭——仅计算空气吸收。"
        )
        layout.addRow(self.sample_abs_check)

        # 比例因子模式 / Scale factor mode
        scale_row = QHBoxLayout()
        self._scale_group = QButtonGroup(self)
        self.scale_auto_radio = QRadioButton("自动（前 10% 数据）")
        self.scale_manual_radio = QRadioButton("手动")
        self.scale_auto_radio.setChecked(True)
        self._scale_group.addButton(self.scale_auto_radio, 0)
        self._scale_group.addButton(self.scale_manual_radio, 1)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 1e9)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(4)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setVisible(False)
        self.scale_spin.setToolTip(
            "比例因子将仿真曲线缩放到与实验数据匹配。自动模式取前 10% 数据的中位数比值。"
        )
        self.scale_auto_radio.setToolTip(self.scale_spin.toolTip())
        self.scale_manual_radio.setToolTip(self.scale_spin.toolTip())
        self.scale_auto_radio.toggled.connect(self._on_scale_mode_toggled)

        scale_row.addWidget(self.scale_auto_radio)
        scale_row.addWidget(self.scale_manual_radio)
        scale_row.addWidget(self.scale_spin)
        scale_row.addStretch()
        layout.addRow("比例因子", scale_row)

        # 实时估算提示 / Live estimation hint
        self._estimation_label = QLabel("")
        self._estimation_label.setWordWrap(True)
        self._estimation_label.setStyleSheet("color: #1565C0; font-size: 11px; padding: 4px;")
        layout.addRow(self._estimation_label)

        # 嵌入式物理背景预览画布 / Embedded physics background preview canvas
        self._preview_canvas = MiniCurveCanvas(self)
        layout.addRow(self._preview_canvas)

        self._syncing = False

        # 初始刷新 / Initial refresh
        self._refresh_estimation()
        self._refresh_preview()

        # 信号连接 — 估算文本 / Signal connections — estimation text
        self.sd_spin.valueChanged.connect(lambda _: self._refresh_estimation())
        self.energy_spin.valueChanged.connect(lambda _: self._refresh_estimation())
        self.wavelength_spin.valueChanged.connect(lambda _: self._refresh_estimation())

        # 信号连接 — 预览画布 / Signal connections — preview canvas
        self.energy_spin.valueChanged.connect(lambda _: self._refresh_preview())
        self.wavelength_spin.valueChanged.connect(lambda _: self._refresh_preview())
        self.sd_spin.valueChanged.connect(lambda _: self._refresh_preview())
        self.unit_q_radio.toggled.connect(lambda _: self._refresh_preview())
        self.unit_2t_radio.toggled.connect(lambda _: self._refresh_preview())
        self.sample_abs_check.toggled.connect(lambda _: self._refresh_preview())
        self.formula_edit.textChanged.connect(lambda _: self._refresh_preview())
        self.density_spin.valueChanged.connect(lambda _: self._refresh_preview())
        self.thickness_spin.valueChanged.connect(lambda _: self._refresh_preview())

    # -- Energy / Wavelength sync / 能量-波长联动 ---------------------------

    def _on_energy_radio_toggled(self, checked: bool) -> None:
        self.energy_spin.setVisible(checked)
        self.wavelength_spin.setVisible(not checked)

    def _sync_energy_to_wavelength(self, value: float) -> None:
        if self._syncing or not self.energy_radio.isChecked():
            return
        self._syncing = True
        self.wavelength_spin.setValue(energy_to_wavelength(value))
        self._syncing = False

    def _sync_wavelength_to_energy(self, value: float) -> None:
        if self._syncing or not self.wavelength_radio.isChecked():
            return
        self._syncing = True
        self.energy_spin.setValue(wavelength_to_energy(value))
        self._syncing = False

    # -- Scale mode toggle / 比例因子切换 -----------------------------------

    def _on_scale_mode_toggled(self, auto_checked: bool) -> None:
        self.scale_spin.setVisible(not auto_checked)

    # -- Estimation / 实时估算 -----------------------------------------------

    def _refresh_estimation(self) -> None:
        """Update the estimation hint label with current physics parameters.
        根据当前物理参数更新估算提示标签。"""
        try:
            from BGsub.core.phys_fit import compute_air_mu

            energy_eV = self.get_energy_eV()
            sd_mm = self.sd_spin.value()
            sd_cm = sd_mm / 10.0
            mu_air = compute_air_mu(energy_eV)

            T_0 = np.exp(-mu_air * sd_cm)
            T_45 = np.exp(-mu_air * sd_cm / np.cos(np.deg2rad(45)))
            T_80 = np.exp(-mu_air * sd_cm / np.cos(np.deg2rad(80)))
            var_pct = (T_0 - T_80) / T_0 * 100 if T_0 > 0 else 0

            hint = (
                f"估算：μ_air = {mu_air:.5f} cm⁻¹，"
                f"SD = {sd_mm:.0f} mm → "
                f"T(0°) = {T_0:.4f}，T(45°) = {T_45:.4f}，T(80°) = {T_80:.4f}，"
                f"0°~80° 衰减幅度 = {var_pct:.1f}%"
            )
            if var_pct < 5:
                hint += "\n⚠ 衰减幅度很小，曲线接近水平直线。"
                hint += "低能量（<10 keV）或长 SD（>500 mm）时效果更明显。"
                self._estimation_label.setStyleSheet(
                    "color: #E65100; font-size: 11px; padding: 4px;"
                )
            else:
                self._estimation_label.setStyleSheet(
                    "color: #1565C0; font-size: 11px; padding: 4px;"
                )

            self._estimation_label.setText(hint)
        except Exception:
            self._estimation_label.setText("")

    # -- Preview / 物理背景预览 -----------------------------------------------

    def _refresh_preview(self) -> None:
        """Recompute and redraw the physics background preview curve.
        重新计算并重绘物理背景预览曲线。"""
        try:
            unit = "q" if self.unit_q_radio.isChecked() else "2theta"

            if unit == "q":
                x = np.linspace(0.01, 5.0, 300)
            else:
                x = np.linspace(0.1, 80.0, 300)

            energy_eV = self.get_energy_eV()
            wavelength_A = self.get_wavelength_A()
            sd_mm = self.sd_spin.value()
            sample_abs_on = self.sample_abs_check.isChecked()
            formula = self.formula_edit.text().strip() if sample_abs_on else ""
            density = self.density_spin.value()
            thickness_mm = self.thickness_spin.value()

            I_sim, _info = simulate_phys_curve(
                x=x,
                unit=unit,
                energy_eV=energy_eV,
                wavelength_A=wavelength_A,
                sd_mm=sd_mm,
                formula=formula,
                density=density,
                thickness_mm=thickness_mm,
                sample_abs_on=sample_abs_on,
            )

            self._preview_canvas.clear()
            self._preview_canvas.plot_curve(x, I_sim, label="物理背景预览", color="#1565C0")
            self._preview_canvas.plot_fill(x, I_sim, color="#1565C0", alpha=0.10)

            x_label = "q (Å⁻¹)" if unit == "q" else "2θ (°)"
            self._preview_canvas.set_labels(
                x_label=x_label,
                y_label="归一化透过率",
                title="物理背景预览",
            )
            self._preview_canvas.finalize()

        except Exception:
            self._preview_canvas.clear()
            self._estimation_label.setText(
                "⚠ 预览计算失败，请检查参数是否有效（如能量范围、化学式等）"
            )
            self._estimation_label.setStyleSheet("color: #C62828; font-size: 11px; padding: 4px;")

    # -- Formula validation / 化学式验证 -----------------------------------

    def _validate_formula(self) -> None:
        formula = self.formula_edit.text().strip()
        if not formula:
            self.formula_result.setText("⚠ 未输入")
            self.formula_result.setStyleSheet("color: #EF6C00; font-size: 10px;")
            return
        ok, msg = validate_formula(formula)
        if ok:
            self.formula_result.setText(f"✓ {msg}")
            self.formula_result.setStyleSheet("color: #2E7D32; font-size: 10px;")
        else:
            self.formula_result.setText(f"✗ {msg}")
            self.formula_result.setStyleSheet("color: #C62828; font-size: 10px;")

    # -- Public helpers / 公开辅助方法 --------------------------------------

    def get_energy_eV(self) -> float:
        """Return energy in eV. 返回能量（eV）。"""
        if self.energy_radio.isChecked():
            return self.energy_spin.value() * 1000.0
        return wavelength_to_energy(self.wavelength_spin.value()) * 1000.0

    def get_wavelength_A(self) -> float:
        """Return wavelength in Å. 返回波长（Å）。"""
        if self.wavelength_radio.isChecked():
            return self.wavelength_spin.value()
        return energy_to_wavelength(self.energy_spin.value())

    def apply_to_config(self, config: CurveProcessorConfig) -> None:
        config.process_mode = ProcessMode.PHYS_FIT
        config.phys_formula = self.formula_edit.text().strip()
        config.phys_density = self.density_spin.value()
        config.phys_thickness_mm = self.thickness_spin.value()
        config.phys_sd_mm = self.sd_spin.value()
        config.phys_energy_eV = self.get_energy_eV()
        config.phys_wavelength_A = self.get_wavelength_A()
        config.phys_sample_abs_on = self.sample_abs_check.isChecked()
        config.phys_unit = "q" if self.unit_q_radio.isChecked() else "2theta"
        config.phys_scale_mode = "auto" if self.scale_auto_radio.isChecked() else "manual"
        config.phys_scale_factor = self.scale_spin.value()


MODE_PANEL_MAP = {
    ProcessMode.MORPH_1D: Morph1DPanel,
    ProcessMode.FIT_1D: Fit1DPanel,
    ProcessMode.T_BG_SUBTRACT: TBGSubtractPanel,
    ProcessMode.PHYS_FIT: PhysFitPanel,
}
