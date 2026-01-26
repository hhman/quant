"""
面板数据转换模块

提供扁平数据与面板数据之间的转换功能：
- 扁平数组 → 面板数据
- 面板数据 → 扁平数组
- 双面板数据构建
- 数据清洗
- DataFrame → 扁平数组 + MultiIndex + 边界索引
- MultiIndex DataFrame → GP 训练数据（展平 + 边界计算）
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def flatten_to_panel(arr: np.ndarray, index: pd.MultiIndex) -> pd.DataFrame:
    """
    扁平数组 → 面板数据

    Args:
        arr: 扁平数组 (n_samples,)
        index: MultiIndex (instrument, datetime)

    Returns:
        面板数据 (n_dates, n_stocks)
    """
    df = pd.DataFrame({"value": arr}, index=index)
    return df["value"].unstack(level=0)


def panel_to_flatten(panel: pd.DataFrame) -> np.ndarray:
    """
    面板数据 → 扁平数组

    Args:
        panel: 面板数据 (n_dates, n_stocks)

    Returns:
        扁平数组 (n_samples,)
    """
    return panel.stack().values


def clean_panel(
    panel: pd.DataFrame, axis: int = 1, min_samples: int = 1
) -> pd.DataFrame:
    """
    清洗面板数据：删除全 NaN 列/行

    Args:
        panel: 面板数据
        axis: 删除维度 (0=行, 1=列)
        min_samples: 最小样本数

    Returns:
        清洗后的面板数据

    Raises:
        ValueError: 样本量不足
    """
    if axis == 0:
        valid_mask = ~panel.isna().all(axis=1)
        result = panel.loc[valid_mask]
    else:
        valid_mask = ~panel.isna().all(axis=0)
        result = panel.loc[:, valid_mask]

    n_samples = result.shape[1] if axis == 0 else result.shape[0]
    if n_samples < min_samples:
        raise ValueError(f"样本量不足: {n_samples} < {min_samples}")

    return result


def build_dual_panel(
    y_true: np.ndarray, y_pred: np.ndarray, index: pd.MultiIndex
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    同时构建 y_true 和 y_pred 的面板数据

    Args:
        y_true: 真实值（扁平）
        y_pred: 预测值（扁平）
        index: MultiIndex

    Returns:
        (y_true_panel, y_pred_panel)
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index)

    y_true_panel = df["y_true"].unstack(level=0)
    y_pred_panel = df["y_pred"].unstack(level=0)

    return y_true_panel, y_pred_panel


def dataframe_to_flatten(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.MultiIndex, List[int]]:
    """
    将 MultiIndex DataFrame 展平为 1D 数组

    Args:
        df: MultiIndex DataFrame (instrument, datetime)

    Returns:
        arr: 展平的 1D 数组
        index: MultiIndex
        boundaries: 边界索引列表（每只股票的起始位置）

    Example:
        >>> df = pd.DataFrame(..., index=pd.MultiIndex.from_tuples(...))
        >>> arr, index, boundaries = dataframe_to_flatten(df)
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame 必须有 MultiIndex")

    index = df.index
    arr = df.values.flatten(order="F")  # 按列展平（先股票后时间）
    boundaries = calc_boundaries(index)

    return arr, index, boundaries


def calc_boundaries(index: pd.MultiIndex) -> List[int]:
    """
    计算每只股票的起始位置

    Args:
        index: MultiIndex (instrument, datetime)

    Returns:
        边界索引列表（每只股票的起始位置）

    Example:
        >>> index = pd.MultiIndex.from_tuples([
        ...     ('stock1', '2020-01-01'),
        ...     ('stock1', '2020-01-02'),
        ...     ('stock2', '2020-01-01'),
        ... ])
        >>> calc_boundaries(index)
        [0, 2]  # stock1 从 0 开始，stock2 从 2 开始
    """
    if not isinstance(index, pd.MultiIndex):
        return []

    boundaries = []
    prev_instrument = None

    for i, (instrument, _) in enumerate(index):
        if instrument != prev_instrument:
            boundaries.append(i)
            prev_instrument = instrument

    return boundaries


def flatten_features(features_df: pd.DataFrame) -> np.ndarray:
    """
    展平特征 DataFrame

    Args:
        features_df: (n_samples, n_features) MultiIndex DataFrame

    Returns:
        X: (n_samples, n_features) 展平的特征数组
    """
    return features_df.values.flatten(order="F").reshape(-1, features_df.shape[1])


def flatten_target(target_df: pd.DataFrame) -> np.ndarray:
    """
    展平标签 DataFrame

    Args:
        target_df: (n_samples, 1) MultiIndex DataFrame

    Returns:
        y: (n_samples,) 展平的标签数组
    """
    return target_df.values.flatten()
