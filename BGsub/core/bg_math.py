#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bg_math.py — Background estimation algorithms
背景估计算法

数学方法拟合背景，包括 Rolling Ball 等算法。
Mathematical background estimation methods.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import minimum_filter
from typing import Optional, Tuple


def apply_mask(
    data: np.ndarray,
    mask_min: Optional[float] = None,
    mask_max: Optional[float] = None,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply threshold mask to data.
    对数据应用阈值掩膜。

    Parameters
    ----------
    data : np.ndarray
        Input data / 输入数据
    mask_min : float, optional
        Minimum threshold, values below this will be masked
        最小阈值，低于此值的数据将被掩膜
    mask_max : float, optional
        Maximum threshold, values above this will be masked
        最大阈值，高于此值的数据将被掩膜
    fill_value : float
        Fill value for masked regions / 掩膜区域的填充值

    Returns
    -------
    masked_data : np.ndarray
        Data with masked regions filled / 应用掩膜后的数据
    mask_array : np.ndarray
        Boolean mask array (True = masked region)
        布尔掩膜数组 (True = 被掩膜的区域)
    """
    data = data.astype(np.float32)
    mask_array = np.zeros_like(data, dtype=bool)

    if mask_min is not None:
        mask_array |= (data < mask_min)

    if mask_max is not None:
        mask_array |= (data > mask_max)

    mask_array |= ~np.isfinite(data)

    masked_data = data.copy()
    masked_data[mask_array] = fill_value

    return masked_data, mask_array


def rolling_ball_background_subtraction(
    data: np.ndarray,
    radius: float,
    rolling_ball_type: str = "classic",
    mask_array: Optional[np.ndarray] = None,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling ball background subtraction.
    滚球法背景扣除。

    Parameters
    ----------
    data : np.ndarray
        Input image data / 输入图像数据
    radius : float
        Radius of rolling ball in pixels / 滚球半径（像素单位）
    rolling_ball_type : str
        'classic' or 'smooth' / 'classic'（经典）或 'smooth'（平滑）
    mask_array : np.ndarray, optional
        Boolean mask (True = masked region) / 布尔掩膜
    fill_value : float
        Fill value for masked regions / 掩膜区域填充值

    Returns
    -------
    background : np.ndarray
        Estimated background / 估计的背景
    subtracted : np.ndarray
        Background-subtracted signal / 扣除背景后的信号
    """
    data = data.astype(np.float32)

    if mask_array is not None and mask_array.any():
        data_for_rolling = data.copy()
        data_for_rolling[mask_array] = np.nan

        valid_mask = ~np.isnan(data_for_rolling)
        if valid_mask.any():
            valid_coords = np.argwhere(valid_mask)
            valid_values = data_for_rolling[valid_mask]
            all_coords = np.argwhere(np.ones_like(data_for_rolling, dtype=bool))

            from scipy.spatial import KDTree
            tree = KDTree(valid_coords)
            distances, indices = tree.query(all_coords, k=1)
            interpolated_values = valid_values[indices]
            data_for_rolling = interpolated_values.reshape(data.shape)
        else:
            data_for_rolling = np.zeros_like(data)
    else:
        data_for_rolling = data.copy()

    if rolling_ball_type == "classic":
        if radius > 0:
            size = 2 * radius + 1
            y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            kernel = x * x + y * y <= radius * radius
            kernel = kernel.astype(np.float32)
        else:
            kernel = np.ones((1, 1), dtype=np.float32)

        try:
            background = minimum_filter(
                data_for_rolling, footprint=kernel, mode="nearest"
            )
            if radius > 2:
                background = ndimage.gaussian_filter(
                    background, sigma=max(1, radius / 4.0)
                )
        except Exception:
            background = ndimage.median_filter(
                data_for_rolling, size=max(3, radius)
            )

    elif rolling_ball_type == "smooth":
        if radius > 0:
            y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            kernel = np.sqrt(x * x + y * y) <= radius
            kernel = kernel.astype(np.float32)
        else:
            kernel = np.ones((1, 1), dtype=np.float32)

        try:
            eroded = ndimage.grey_erosion(data_for_rolling, footprint=kernel, mode="nearest")
            background = ndimage.grey_dilation(eroded, footprint=kernel, mode="nearest")
        except Exception:
            background = minimum_filter(
                data_for_rolling, footprint=kernel, mode="nearest"
            )
    else:
        raise ValueError("rolling_ball_type must be 'classic' or 'smooth'")

    background = np.minimum(background, data)

    if mask_array is not None and mask_array.any():
        background[mask_array] = fill_value

    subtracted = data - background
    subtracted = np.maximum(subtracted, 0)

    if mask_array is not None:
        subtracted[mask_array] = fill_value

    return background, subtracted


def auto_detect_mask_thresholds(
    data: np.ndarray,
    low_percentile: float = 0.1,
    high_percentile: float = 99.9,
) -> Tuple[float, float]:
    """
    Auto-detect mask thresholds based on percentiles.
    基于百分位数自动检测掩膜阈值。

    Parameters
    ----------
    data : np.ndarray
        Input data / 输入数据
    low_percentile : float
        Lower percentile for threshold / 下百分位数阈值
    high_percentile : float
        Upper percentile for threshold / 上百分位数阈值

    Returns
    -------
    mask_min : float
        Minimum threshold / 最小阈值
    mask_max : float
        Maximum threshold / 最大阈值
    """
    flat_data = data.ravel()
    flat_data = flat_data[np.isfinite(flat_data)]

    if len(flat_data) == 0:
        return 0.0, 100.0

    mask_min = float(np.percentile(flat_data, low_percentile))
    mask_max = float(np.percentile(flat_data, high_percentile))

    return mask_min, mask_max


def polynomial_background(
    data: np.ndarray,
    degree: int = 2,
    axis: int = 0,
) -> np.ndarray:
    """
    Fit polynomial to each row/column and estimate background.
    对每行/列拟合多项式并估计背景。

    Parameters
    ----------
    data : np.ndarray
        Input 2D data / 输入2D数据
    degree : int
        Polynomial degree / 多项式阶数
    axis : int
        Axis along which to fit (0: rows, 1: columns) / 拟合轴 (0: 行, 1: 列)

    Returns
    -------
    background : np.ndarray
        Polynomial-fitted background / 多项式拟合背景
    """
    if axis == 0:
        background = np.zeros_like(data)
        for i in range(data.shape[0]):
            y = data[i, :]
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, degree)
            background[i, :] = np.polyval(coeffs, x)
    else:
        background = np.zeros_like(data)
        for i in range(data.shape[1]):
            y = data[:, i]
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, degree)
            background[:, i] = np.polyval(coeffs, x)

    return background


def morphological_background(
    data: np.ndarray,
    radius: int = 10,
    operation: str = "opening",
) -> np.ndarray:
    """
    Estimate background using morphological operations.
    使用形态学操作估计背景。

    Parameters
    ----------
    data : np.ndarray
        Input data / 输入数据
    radius : int
        Structure element radius / 结构元素半径
    operation : str
        'opening' or 'closing' / 'opening'（开运算）或 'closing'（闭运算）

    Returns
    -------
    background : np.ndarray
        Morphological background estimate / 形态学背景估计
    """
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = x * x + y * y <= radius * radius
    kernel = kernel.astype(np.float32)

    if operation == "opening":
        eroded = ndimage.grey_erosion(data, footprint=kernel, mode="nearest")
        background = ndimage.grey_dilation(eroded, footprint=kernel, mode="nearest")
    else:
        dilated = ndimage.grey_dilation(data, footprint=kernel, mode="nearest")
        background = ndimage.grey_erosion(dilated, footprint=kernel, mode="nearest")

    return background