#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bg_fitter.py — Background fitter interface
背景拟合器接口
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from BGsub.core.bg_math import (
    rolling_ball_background_subtraction,
    polynomial_background,
    morphological_background,
    apply_mask,
    auto_detect_mask_thresholds,
)


class BgMethod(Enum):
    """Background estimation methods / 背景估计方法"""
    ROLLING_BALL_CLASSIC = "rolling_ball_classic"
    ROLLING_BALL_SMOOTH = "rolling_ball_smooth"
    POLYNOMIAL = "polynomial"
    MORPHOLOGICAL = "morphological"


@dataclass
class BgFitterConfig:
    """Configuration for background fitting / 背景拟合配置"""
    method: BgMethod = BgMethod.ROLLING_BALL_CLASSIC
    radius: float = 20.0
    polynomial_degree: int = 2
    mask_low_percentile: float = 0.1
    mask_high_percentile: float = 99.9
    fill_value: float = 0.0
    smooth_result: bool = True
    smooth_sigma: float = 1.0


class BgFitter:
    """
    Main interface for background estimation and subtraction.
    背景估计与扣除的主接口。

    Supports multiple methods:
      - Rolling Ball (classic / smooth)
      - Polynomial fitting
      - Morphological operations
    """

    def __init__(self, config: Optional[BgFitterConfig] = None):
        """
        Initialize fitter with configuration.
        用配置初始化拟合器。

        Parameters
        ----------
        config : BgFitterConfig, optional
            Configuration object / 配置对象
        """
        self.config = config or BgFitterConfig()

    def set_config(self, **kwargs) -> None:
        """Update configuration / 更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit background to data.
        对数据拟合背景。

        Parameters
        ----------
        data : np.ndarray
            2D input image data / 2D输入图像数据

        Returns
        -------
        background : np.ndarray
            Estimated background / 估计的背景
        subtracted : np.ndarray
            Background-subtracted result / 扣除背景后的结果
        mask : np.ndarray
            Applied mask /应用的掩膜
        """
        config = self.config

        mask_min, mask_max = auto_detect_mask_thresholds(
            data,
            low_percentile=config.mask_low_percentile,
            high_percentile=config.mask_high_percentile,
        )
        masked_data, mask = apply_mask(
            data, mask_min, mask_max, fill_value=config.fill_value
        )

        if config.method == BgMethod.ROLLING_BALL_CLASSIC:
            background, subtracted = rolling_ball_background_subtraction(
                masked_data,
                radius=config.radius,
                rolling_ball_type="classic",
                mask_array=mask,
                fill_value=config.fill_value,
            )
        elif config.method == BgMethod.ROLLING_BALL_SMOOTH:
            background, subtracted = rolling_ball_background_subtraction(
                masked_data,
                radius=config.radius,
                rolling_ball_type="smooth",
                mask_array=mask,
                fill_value=config.fill_value,
            )
        elif config.method == BgMethod.POLYNOMIAL:
            background = polynomial_background(
                masked_data, degree=config.polynomial_degree, axis=0
            )
            background = np.minimum(background, masked_data)
            subtracted = masked_data - background
            subtracted = np.maximum(subtracted, 0)
            if mask is not None:
                subtracted[mask] = config.fill_value
        elif config.method == BgMethod.MORPHOLOGICAL:
            background = morphological_background(
                masked_data, radius=int(config.radius), operation="opening"
            )
            background = np.minimum(background, masked_data)
            subtracted = masked_data - background
            subtracted = np.maximum(subtracted, 0)
            if mask is not None:
                subtracted[mask] = config.fill_value
        else:
            raise ValueError(f"Unknown method: {config.method}")

        if config.smooth_result and config.smooth_sigma > 0:
            from scipy import ndimage
            subtracted = ndimage.gaussian_filter(subtracted, sigma=config.smooth_sigma)

        return background, subtracted, mask

    def fit_with_reference(
        self,
        sample: np.ndarray,
        reference: np.ndarray,
        transmission: float = 1.0,
    ) -> np.ndarray:
        """
        Subtract reference background with optional transmission correction.
        使用参考背景进行扣除，可选透射率校正。

        Formula: result = sample / T - reference

        Parameters
        ----------
        sample : np.ndarray
            Sample image / 样品图像
        reference : np.ndarray
            Reference background image / 参考背景图像
        transmission : float
            Transmission factor (0-1) / 透射率因子

        Returns
        -------
        result : np.ndarray
            Background-subtracted result / 扣除背景后的结果
        """
        if transmission <= 0:
            transmission = 1.0

        sr, sc = sample.shape
        rr, rc = reference.shape
        cr, cc = min(sr, rr), min(sc, rc)

        result = np.zeros_like(sample, dtype=np.float64)
        result[:cr, :cc] = (
            sample[:cr, :cc] / transmission - reference[:cr, :cc]
        )

        return result


class RollingBallMethod:
    """Rolling ball method convenience class / 滚球法便捷类"""

    @staticmethod
    def classic(
        data: np.ndarray,
        radius: float = 20.0,
        mask_min: Optional[float] = None,
        mask_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classic rolling ball method.
        经典滚球法。

        Parameters
        ----------
        data : np.ndarray
            Input data / 输入数据
        radius : float
            Ball radius / 滚球半径
        mask_min, mask_max : float, optional
            Mask thresholds / 掩膜阈值

        Returns
        -------
        background, subtracted : Tuple[np.ndarray, np.ndarray]
        """
        if mask_min is None or mask_max is None:
            mask_min, mask_max = auto_detect_mask_thresholds(data)

        masked_data, mask = apply_mask(data, mask_min, mask_max)
        return rolling_ball_background_subtraction(
            masked_data, radius, "classic", mask
        )

    @staticmethod
    def smooth(
        data: np.ndarray,
        radius: float = 20.0,
        mask_min: Optional[float] = None,
        mask_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smooth rolling ball method.
        平滑滚球法。

        Parameters
        ----------
        data : np.ndarray
            Input data / 输入数据
        radius : float
            Ball radius / 滚球半径
        mask_min, mask_max : float, optional
            Mask thresholds / 掩膜阈值

        Returns
        -------
        background, subtracted : Tuple[np.ndarray, np.ndarray]
        """
        if mask_min is None or mask_max is None:
            mask_min, mask_max = auto_detect_mask_thresholds(data)

        masked_data, mask = apply_mask(data, mask_min, mask_max)
        return rolling_ball_background_subtraction(
            masked_data, radius, "smooth", mask
        )