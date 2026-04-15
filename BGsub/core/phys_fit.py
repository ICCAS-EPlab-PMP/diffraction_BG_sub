#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phys_fit.py — Physics-based attenuation curve simulation for 1D background estimation
基于物理衰减的 1D 背景仿真曲线计算

Simulates I_sim(q) or I_sim(2θ) considering:
  - Angle-dependent air absorption (flat-detector geometry)
  - Optional sample absorption (Beer-Lambert law)

Core formula (flat-detector):
    d_air(2θ) = SD / cos(2θ)
    T_air(2θ) = exp(-μ_air × SD / cos(2θ))
    T_sample  = exp(-μ_sample × t)       [optional, default off]
    I_sim     = scale × T_sample × T_air(2θ)

平面板探测器几何下的核心公式：
    空气路径随角度增大：d_air = SD / cos(2θ)
    空气透过率：T_air = exp(-μ_air × SD / cos(2θ))
    样品透过率：T_sample = exp(-μ_sample × t)  [可选，默认关闭]
    仿真曲线：I_sim = scale × T_sample × T_air(2θ)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / 常量
# ---------------------------------------------------------------------------

# E (keV) × λ (Å) = 12.3984
KEV_ANGSTROM = 12.3984

# Standard dry air composition (molar fraction)
# 标准干燥空气组成（摩尔分数）
AIR_FORMULA = "N0.78O0.21Ar0.01"
AIR_DENSITY = 0.001225  # g/cm³ at STP

# Maximum 2θ (degrees) to avoid division-by-zero in cos(2θ)
# 最大 2θ 角度限制，避免 cos(2θ) → 0 的除零问题
MAX_TWO_THETA_DEG = 89.5


# ---------------------------------------------------------------------------
# Energy ↔ Wavelength conversion / 能量-波长转换
# ---------------------------------------------------------------------------


def energy_to_wavelength(energy_keV: float) -> float:
    """
    Convert photon energy to wavelength.
    将光子能量转换为波长。

    Parameters / 参数
    -----------------
    energy_keV : float
        Photon energy in keV / 光子能量（keV）

    Returns / 返回
    --------------
    float
        Wavelength in Å / 波长（Å）
    """
    return KEV_ANGSTROM / energy_keV


def wavelength_to_energy(wavelength_A: float) -> float:
    """
    Convert wavelength to photon energy.
    将波长转换为光子能量。

    Parameters / 参数
    -----------------
    wavelength_A : float
        Wavelength in Å / 波长（Å）

    Returns / 返回
    --------------
    float
        Photon energy in keV / 光子能量（keV）
    """
    return KEV_ANGSTROM / wavelength_A


# ---------------------------------------------------------------------------
# Absorption coefficient computation / 吸收系数计算
# ---------------------------------------------------------------------------


def compute_sample_mu(
    formula: str,
    energy_eV: float,
    density: float,
) -> float:
    """
    Compute linear absorption coefficient μ for a chemical compound.
    计算化合物的线性吸收系数 μ。

    Uses xraydb.material_mu which returns μ (cm⁻¹) when density is given.
    使用 xraydb.material_mu，在给定密度时返回 μ（cm⁻¹）。

    Parameters / 参数
    -----------------
    formula : str
        Chemical formula, e.g. 'LaB6', 'C10H8O4', 'C2H4'
        化学式，如 'LaB6', 'C10H8O4', 'C2H4'
    energy_eV : float
        Photon energy in eV / 光子能量（eV）
    density : float
        Material density in g/cm³ / 材料密度（g/cm³）

    Returns / 返回
    --------------
    float
        Linear absorption coefficient μ (cm⁻¹) / 线性吸收系数（cm⁻¹）
    """
    from xraydb import material_mu

    return float(material_mu(formula, energy_eV, density=density))


def compute_air_mu(energy_eV: float) -> float:
    """
    Compute linear absorption coefficient μ for standard air.
    计算标准空气的线性吸收系数 μ。

    Uses standard dry-air composition N₂(78%) + O₂(21%) + Ar(1%),
    density ≈ 0.001225 g/cm³ at STP.

    Parameters / 参数
    -----------------
    energy_eV : float
        Photon energy in eV / 光子能量（eV）

    Returns / 返回
    --------------
    float
        Air linear absorption coefficient μ_air (cm⁻¹)
        空气线性吸收系数（cm⁻¹）
    """
    from xraydb import material_mu

    return float(material_mu(AIR_FORMULA, energy_eV, density=AIR_DENSITY))


def compute_sample_transmission(
    formula: str,
    energy_eV: float,
    density: float,
    thickness_mm: float,
) -> Tuple[float, float]:
    """
    Compute sample transmission T_sample = exp(-μ × t).
    计算样品透过率 T_sample = exp(-μ × t)。

    Parameters / 参数
    -----------------
    formula : str
        Chemical formula / 化学式
    energy_eV : float
        Photon energy in eV / 光子能量（eV）
    density : float
        Material density (g/cm³) / 材料密度（g/cm³）
    thickness_mm : float
        Sample thickness (mm) / 样品厚度（mm）

    Returns / 返回
    --------------
    (T_sample, mu_sample) : Tuple[float, float]
        T_sample : transmission (0~1) / 透过率（0~1）
        mu_sample : linear absorption coefficient (cm⁻¹) / 线性吸收系数
    """
    mu_sample = compute_sample_mu(formula, energy_eV, density)
    t_cm = thickness_mm / 10.0  # mm → cm
    T_sample = float(np.exp(-mu_sample * t_cm))
    return T_sample, mu_sample


# ---------------------------------------------------------------------------
# q ↔ 2θ conversion / q 与 2θ 转换
# ---------------------------------------------------------------------------


def q_to_twotheta(q: np.ndarray, wavelength_A: float) -> np.ndarray:
    """
    Convert scattering vector q to scattering angle 2θ.
    将散射矢量 q 转换为散射角 2θ。

    q = 4π sin(θ) / λ  →  2θ = 2 arcsin(qλ / 4π)

    Parameters / 参数
    -----------------
    q : np.ndarray
        Scattering vector in Å⁻¹ / 散射矢量（Å⁻¹）
    wavelength_A : float
        X-ray wavelength in Å / X 射线波长（Å）

    Returns / 返回
    --------------
    np.ndarray
        2θ in degrees / 2θ（度）
    """
    sin_theta = q * wavelength_A / (4.0 * np.pi)
    sin_theta = np.clip(sin_theta, 0.0, 1.0)
    theta_rad = np.arcsin(sin_theta)
    return np.rad2deg(theta_rad) * 2.0


def twotheta_to_q(twotheta_deg: np.ndarray, wavelength_A: float) -> np.ndarray:
    """
    Convert scattering angle 2θ to scattering vector q.
    将散射角 2θ 转换为散射矢量 q。

    q = 4π sin(θ) / λ

    Parameters / 参数
    -----------------
    twotheta_deg : np.ndarray
        Scattering angle 2θ in degrees / 散射角 2θ（度）
    wavelength_A : float
        X-ray wavelength in Å / X 射线波长（Å）

    Returns / 返回
    --------------
    np.ndarray
        q in Å⁻¹ / 散射矢量（Å⁻¹）
    """
    theta_rad = np.deg2rad(twotheta_deg / 2.0)
    return 4.0 * np.pi * np.sin(theta_rad) / wavelength_A


# ---------------------------------------------------------------------------
# Main simulation / 主仿真函数
# ---------------------------------------------------------------------------


def simulate_phys_curve(
    x: np.ndarray,
    unit: str,
    energy_eV: float,
    wavelength_A: float,
    sd_mm: float,
    formula: str = "",
    density: float = 1.0,
    thickness_mm: float = 1.0,
    sample_abs_on: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Simulate physics-based attenuation curve I_sim(x).
    基于物理衰减仿真曲线 I_sim(x)。

    For a flat-detector geometry, higher 2θ means longer air path:
        d_air(2θ) = SD / cos(2θ)
        T_air(2θ) = exp(-μ_air × SD / cos(2θ))

    Optional sample absorption:
        T_sample = exp(-μ_sample × t)

    Parameters / 参数
    -----------------
    x : np.ndarray
        q (Å⁻¹) or 2θ (degrees) array / q 或 2θ 数组
    unit : str
        'q' or '2theta' / 单位标识
    energy_eV : float
        Photon energy (eV) / 光子能量（eV）
    wavelength_A : float
        X-ray wavelength (Å) / X 射线波长（Å）
    sd_mm : float
        Sample-detector distance (mm) / 样品-探测器距离（mm）
    formula : str
        Chemical formula (for sample absorption) / 化学式
    density : float
        Sample density (g/cm³) / 样品密度（g/cm³）
    thickness_mm : float
        Sample thickness (mm) / 样品厚度（mm）
    sample_abs_on : bool
        Enable sample absorption / 启用样品吸收

    Returns / 返回
    --------------
    (I_sim, info) : Tuple[np.ndarray, dict]
        I_sim : simulated attenuation curve (normalized to ~1 at 2θ=0)
        info : dict with physics parameters for logging
    """
    x = np.asarray(x, dtype=np.float64)

    # 1. Convert x to 2θ (degrees)
    # 将 x 转换为 2θ（度）
    if unit == "q":
        two_theta = q_to_twotheta(x, wavelength_A)
    else:
        two_theta = x.copy()

    # 2. Clamp 2θ to avoid cos → 0
    # 限制 2θ 避免除零
    two_theta_clamped = np.clip(two_theta, 0.0, MAX_TWO_THETA_DEG)

    # 3. Compute air path: d_air = SD / cos(2θ)
    # 计算空气路径
    two_theta_rad = np.deg2rad(two_theta_clamped)
    cos_2theta = np.cos(two_theta_rad)
    cos_2theta = np.maximum(cos_2theta, 0.01)  # safety floor
    sd_cm = sd_mm / 10.0  # mm → cm
    d_air_cm = sd_cm / cos_2theta

    # 4. Air absorption (angle-dependent)
    # 空气吸收（角度相关）
    mu_air = compute_air_mu(energy_eV)
    T_air = np.exp(-mu_air * d_air_cm)

    # 5. Optional sample absorption
    # 可选样品吸收
    if sample_abs_on and formula and density > 0:
        T_sample, mu_sample = compute_sample_transmission(formula, energy_eV, density, thickness_mm)
    else:
        T_sample = 1.0
        mu_sample = 0.0

    # 6. Simulated curve (normalized so I_sim[0] ≈ 1)
    # 仿真曲线（归一化使 I_sim[0] ≈ 1）
    I_sim = T_sample * T_air

    info = {
        "phys_energy_eV": energy_eV,
        "phys_wavelength_A": wavelength_A,
        "phys_sd_mm": sd_mm,
        "phys_formula": formula if sample_abs_on else "(disabled)",
        "phys_mu_air_cm": round(mu_air, 6),
        "phys_mu_sample_cm": round(mu_sample, 4),
        "phys_T_sample": round(T_sample, 6) if isinstance(T_sample, float) else "N/A",
        "phys_T_air_min": round(float(T_air.min()), 6),
        "phys_T_air_max": round(float(T_air.max()), 6),
        "phys_unit": unit,
    }

    logger.info(
        "物理仿真: μ_air=%.6f cm⁻¹, μ_sample=%.4f cm⁻¹, T_sample=%.4f, T_air 范围=[%.4f, %.4f]",
        mu_air,
        mu_sample,
        T_sample if isinstance(T_sample, float) else 1.0,
        float(T_air.min()),
        float(T_air.max()),
    )

    return I_sim, info


# ---------------------------------------------------------------------------
# Scale factor / 比例因子
# ---------------------------------------------------------------------------


def auto_scale_factor(
    I_measured: np.ndarray,
    I_simulated: np.ndarray,
    fraction: float = 0.1,
) -> float:
    """
    Compute automatic scale factor from the first fraction of data.
    利用前 fraction 比例的数据自动计算比例因子。

    scale = median(I_measured / I_simulated) over first `fraction` of data.

    Parameters / 参数
    -----------------
    I_measured : np.ndarray
        Experimental intensity / 实验强度
    I_simulated : np.ndarray
        Simulated attenuation curve / 仿真衰减曲线
    fraction : float
        Fraction of data to use (default 0.1 = first 10%)
        使用的数据比例（默认 0.1 = 前 10%）

    Returns / 返回
    --------------
    float
        Scale factor / 比例因子
    """
    n = max(1, int(len(I_measured) * fraction))
    I_meas_slice = np.asarray(I_measured[:n], dtype=np.float64)
    I_sim_slice = np.asarray(I_simulated[:n], dtype=np.float64)

    # Avoid division by zero or negative
    valid = (I_sim_slice > 1e-30) & np.isfinite(I_meas_slice)
    if valid.sum() == 0:
        logger.warning("自动比例因子：前 %.0f%% 数据无有效点，使用默认值 1.0", fraction * 100)
        return 1.0

    ratio = I_meas_slice[valid] / I_sim_slice[valid]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]

    if len(ratio) == 0:
        return 1.0

    scale = float(np.median(ratio))
    logger.info("自动比例因子: %.4f (基于前 %d 个有效数据点)", scale, len(ratio))
    return scale


# ---------------------------------------------------------------------------
# Convenience: validate formula / 便捷：验证化学式
# ---------------------------------------------------------------------------


def validate_formula(formula: str) -> Tuple[bool, str]:
    """
    Validate a chemical formula string.
    验证化学式字符串。

    Parameters / 参数
    -----------------
    formula : str
        Chemical formula / 化学式

    Returns / 返回
    --------------
    (valid, message) : Tuple[bool, str]
        valid : True if formula is parseable / 化学式是否可解析
        message : error message if invalid / 错误信息
    """
    if not formula or not formula.strip():
        return False, "化学式为空 / Formula is empty"

    try:
        from xraydb import chemparse

        parsed = chemparse(formula.strip())
        if not parsed:
            return False, f"无法解析化学式: {formula}"
        return True, f"OK: {parsed}"
    except Exception as e:
        return False, f"化学式错误: {e}"
