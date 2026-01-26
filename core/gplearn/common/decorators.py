"""
通用装饰器模块

提供算子和适应度函数使用的装饰器：
- 边界检测装饰器
- 面板数据转换装饰器
"""

from functools import wraps
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from .state import get_index, get_boundary_indices
from .panel import build_dual_panel, clean_panel


def with_boundary_check(func: Callable) -> Callable:
    """
    为时序算子添加边界检测，防止跨股票污染

    Args:
        func: 原始算子函数

    Returns:
        包装后的函数

    使用示例：
        @register_operator(name="sma", category="time_series")
        @with_boundary_check
        def rolling_sma(arr, window):
            return pd.Series(arr).rolling(window).mean()
    """

    @wraps(func)
    def wrapper(arr: np.ndarray, window: int = None) -> np.ndarray:
        result = func(arr, window)

        boundary_indices = get_boundary_indices()
        if not boundary_indices:
            return result

        arr_length = len(result)
        boundary_mask = np.zeros(arr_length, dtype=bool)

        if window is None:
            window = 1

        # 跳过第一个边界（起始位置），只标记股票间的边界
        # boundaries[0] = 0 是第一只股票的起始位置，不应被标记
        for b in boundary_indices[1:]:
            start_idx = b
            end_idx = min(b + window, arr_length)
            boundary_mask[start_idx:end_idx] = True

        result[boundary_mask] = np.nan
        return result

    return wrapper


def _clean_dual_panel(
    y_true_panel: pd.DataFrame,
    y_pred_panel: pd.DataFrame,
    min_samples: int,
    clean_axis: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    清洗双面板数据

    Args:
        y_true_panel: 真实值面板
        y_pred_panel: 预测值面板
        min_samples: 最小样本数
        clean_axis: 清洗维度

    Returns:
        (清洗后的 y_true_panel, 清洗后的 y_pred_panel)
    """
    y_true_panel = clean_panel(y_true_panel, axis=clean_axis, min_samples=min_samples)
    y_pred_panel = clean_panel(y_pred_panel, axis=clean_axis, min_samples=min_samples)
    return y_true_panel, y_pred_panel


def with_panel_convert(
    min_samples: int = 100,
    clean_axis: int = 1,
):
    """
    为适应度函数添加面板数据转换

    Args:
        min_samples: 最小样本数
        clean_axis: 清洗维度 (0=行, 1=列)

    使用示例：
        @register_fitness(name="rank_ic")
        @with_panel_convert(min_samples=100)
        def rank_ic_fitness(y_true_panel, y_pred_panel):
            ic_series = y_pred_panel.corrwith(y_true_panel, axis=1)
            return ic_series.mean()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            index: pd.MultiIndex = None,
            boundary_indices: list = None,
            **kwargs,
        ):
            if index is None:
                index = get_index()

            y_true_panel, y_pred_panel = build_dual_panel(y_true, y_pred, index)
            y_true_panel, y_pred_panel = _clean_dual_panel(
                y_true_panel, y_pred_panel, min_samples, clean_axis
            )

            return func(y_true_panel, y_pred_panel, **kwargs)

        return wrapper

    return decorator


def with_panel_builder(func: Callable) -> Callable:
    """
    为截面算子添加面板数据转换

    Args:
        func: 原始算子函数

    Returns:
        包装后的函数

    使用示例：
        @register_operator(name="rank", category="cross_sectional")
        @with_panel_builder
        def cross_sectional_rank(panel):
            return panel.rank(axis=1, pct=True).fillna(0.5)
    """

    @wraps(func)
    def wrapper(arr: np.ndarray) -> np.ndarray:
        try:
            index = get_index()
        except RuntimeError:
            # Index 未设置（gplearn 测试阶段），直接返回 arr
            # 这样可以满足 gplearn 的 arity 测试要求
            return arr

        # 如果输入数组长度与 index 长度不匹配（gplearn 测试阶段），
        # 直接返回 arr，避免索引长度不匹配的错误
        if len(arr) != len(index):
            return arr

        df = pd.DataFrame({"value": arr}, index=index)
        panel = df["value"].unstack(level=0)

        result_panel = func(panel)

        return result_panel.stack().values

    return wrapper
