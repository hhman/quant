"""Panel数据转换模块，处理MultiIndex DataFrame与Panel数据之间的转换。"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def flatten_to_panel(arr: np.ndarray, index: pd.MultiIndex) -> pd.DataFrame:
    """将扁平化数组转换为Panel格式的DataFrame。

    Args:
        arr: 扁平化数组，形状为(n_samples,)
        index: MultiIndex，层级为(instrument, datetime)

    Returns:
        Panel格式的DataFrame，形状为(n_dates, n_stocks)
    """
    df = pd.DataFrame({"value": arr}, index=index)
    return df["value"].unstack(level=0)


def panel_to_flatten(panel: pd.DataFrame) -> np.ndarray:
    """将Panel格式的DataFrame转换为扁平化数组。

    Args:
        panel: Panel格式的DataFrame，形状为(n_dates, n_stocks)

    Returns:
        扁平化数组，形状为(n_samples,)
    """
    return panel.stack().values


def clean_panel(
    panel: pd.DataFrame, axis: int = 1, min_samples: int = 1
) -> pd.DataFrame:
    """清理Panel数据，删除全部为NaN的行或列。

    Args:
        panel: Panel格式的DataFrame
        axis: 清理轴（0=删除行，1=删除列）
        min_samples: 最小样本数量要求

    Returns:
        清理后的DataFrame

    Raises:
        ValueError: 当清理后的样本数小于最小要求时
    """
    if axis == 0:
        valid_mask = ~panel.isna().all(axis=1)
        result = panel.loc[valid_mask]
    else:
        valid_mask = ~panel.isna().all(axis=0)
        result = panel.loc[:, valid_mask]

    n_samples = result.shape[1] if axis == 0 else result.shape[0]
    if n_samples < min_samples:
        raise ValueError(f"清理后样本数不足: {n_samples} < {min_samples}")

    return result


def build_dual_panel(
    y_true: np.ndarray, y_pred: np.ndarray, index: pd.MultiIndex
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """构建包含真实值和预测值的Panel数据。

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        index: MultiIndex

    Returns:
        (y_true_panel, y_pred_panel) 元组
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index)

    y_true_panel = df["y_true"].unstack(level=0)
    y_pred_panel = df["y_pred"].unstack(level=0)

    return y_true_panel, y_pred_panel


def dataframe_to_flatten(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.MultiIndex, List[int]]:
    """将MultiIndex DataFrame转换为扁平化数组和边界信息。

    Args:
        df: MultiIndex DataFrame，层级为(instrument, datetime)

    Returns:
        arr: 扁平化的1D数组
        index: MultiIndex
        boundaries: 每个instrument的起始索引列表

    Example:
        >>> df = pd.DataFrame(..., index=pd.MultiIndex.from_tuples(...))
        >>> arr, index, boundaries = dataframe_to_flatten(df)
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame必须具有MultiIndex")

    index = df.index
    arr = df.values.flatten(order="F")
    boundaries = calc_boundaries(index)

    return arr, index, boundaries


def calc_boundaries(index: pd.MultiIndex) -> List[int]:
    """计算每个instrument在MultiIndex中的起始位置。

    Args:
        index: MultiIndex，层级为(instrument, datetime)

    Returns:
        边界索引列表

    Example:
        >>> index = pd.MultiIndex.from_tuples([
        ...     ('stock1', '2020-01-01'),
        ...     ('stock1', '2020-01-02'),
        ...     ('stock2', '2020-01-01'),
        ... ])
        >>> calc_boundaries(index)
        [0, 2]
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
    """将特征DataFrame转换为扁平化数组。

    Args:
        features_df: MultiIndex DataFrame，形状为(n_samples, n_features)

    Returns:
        特征数组，形状为(n_samples, n_features)
    """
    return features_df.values.flatten(order="F").reshape(-1, features_df.shape[1])


def flatten_target(target_df: pd.DataFrame) -> np.ndarray:
    """将目标DataFrame转换为扁平化数组。

    Args:
        target_df: MultiIndex DataFrame，形状为(n_samples, 1)

    Returns:
        目标数组，形状为(n_samples,)
    """
    return target_df.values.flatten()
