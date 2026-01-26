"""
Gplearn 算子库

提供时间序列算子和截面算子。
"""

import numpy as np
import pandas as pd

from .common.registry import (
    register_operator,
    get_operator,
    list_operators,
    get_all_operators,
    list_registered_operators,
    get_operators_by_category,
)
from .common.decorators import with_boundary_check


# ==================== 基础算子 ====================


@register_operator(name="abs", category="basic", arity=1)
def op_abs(arr: np.ndarray) -> np.ndarray:
    """绝对值"""
    return np.abs(arr)


@register_operator(name="sqrt", category="basic", arity=1)
def op_sqrt(arr: np.ndarray) -> np.ndarray:
    """平方根（带保护）"""
    return np.sqrt(np.abs(arr) + 1e-10)


@register_operator(name="log", category="basic", arity=1)
def op_log(arr: np.ndarray) -> np.ndarray:
    """自然对数（带保护）"""
    return np.log(np.abs(arr) + 1.0)


# ==================== 时间序列算子 ====================


@register_operator(name="sma", category="time_series", arity=2)
@with_boundary_check
def rolling_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均"""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).mean().values
    result = np.nan_to_num(result, nan=0.0)
    return result


@register_operator(name="ema", category="time_series", arity=2)
@with_boundary_check
def rolling_ema(arr: np.ndarray, window: int) -> np.ndarray:
    """指数移动平均"""
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=float)

    series = pd.Series(arr)
    result = series.ewm(span=window, adjust=False).mean().values
    return result


@register_operator(name="std", category="time_series", arity=2)
@with_boundary_check
def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差"""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).std().values
    result = np.nan_to_num(result, nan=0.0)
    return result


@register_operator(name="momentum", category="time_series", arity=2)
@with_boundary_check
def rolling_momentum(arr: np.ndarray, window: int) -> np.ndarray:
    """动量"""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    result = np.zeros_like(arr, dtype=float)
    result[window:] = arr[window:] / (arr[:-window] + 1e-10) - 1
    return result


@register_operator(name="delta", category="time_series", arity=2)
@with_boundary_check
def rolling_delta(arr: np.ndarray, window: int) -> np.ndarray:
    """一阶差分"""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    result = np.zeros_like(arr, dtype=float)
    result[window:] = arr[window:] - arr[:-window]
    return result


@register_operator(name="max", category="time_series", arity=2)
@with_boundary_check
def rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动最大值"""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).max().values
    result = np.nan_to_num(result, nan=0.0)
    return result


@register_operator(name="min", category="time_series", arity=2)
@with_boundary_check
def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动最小值"""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).min().values
    result = np.nan_to_num(result, nan=0.0)
    return result


@register_operator(name="ts_rank", category="time_series", arity=2)
@with_boundary_check
def ts_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """时间序列排名"""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    result = np.zeros_like(arr, dtype=float)

    for i in range(window - 1, len(arr)):
        window_data = arr[i - window + 1 : i + 1]
        rank = pd.Series(window_data).rank().iloc[-1]
        result[i] = rank / window

    return result


# @register_operator(name="corr", category="time_series", arity=3)
# @with_boundary_check
# def rolling_corr(arr1: np.ndarray, arr2: np.ndarray, window: int) -> np.ndarray:
#     """滚动相关系数"""
#     if len(arr1) < window or len(arr2) < window:
#         return np.full_like(arr1, np.nan, dtype=float)
#
#     series1 = pd.Series(arr1)
#     series2 = pd.Series(arr2)
#     result = series1.rolling(window=window, min_periods=1).corr(series2).values
#     return result
#
# 注：gplearn 仅支持 arity 1 或 2，corr 需要 3 个参数（arr1, arr2, window）
# TODO: 需要重新设计以适应 gplearn 的限制（例如将 arr1 和 arr2 合并为单个参数）


# ==================== 截面算子 ====================


# @register_operator(name="rank", category="cross_sectional", arity=1)
# @with_panel_builder
# def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
#     """横截面排名"""
#     return panel.rank(axis=1, pct=True).fillna(0.5)
#
#
# @register_operator(name="zscore", category="cross_sectional", arity=1)
# @with_panel_builder
# def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
#     """横截面标准化"""
#     mean = panel.mean(axis=1)
#     std = panel.std(axis=1)
#     result = (panel.sub(mean, axis=0)).div(std, axis=0)
#     return result.fillna(0)
#
# 注：截面算子暂时禁用，需要解决形状不一致问题
# TODO: 修复 panel.stack() 后的形状不一致问题


__all__ = [
    "register_operator",
    "get_operator",
    "list_operators",
    "get_all_operators",
    "list_registered_operators",
    "get_operators_by_category",
]
