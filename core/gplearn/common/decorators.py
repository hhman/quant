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


def with_boundary_check(
    func: Callable = None, *, window_size: int | None = 1
) -> Callable:
    """
    为时序算子添加边界检测，防止跨股票污染

    Args:
        func: 原始算子函数
        window_size: 窗口大小
            - arity=1 的时序算子：必须指定窗口大小（如 20）
            - arity=2 的相关性算子：必须指定窗口大小（如 10）
            - arity=2 的算术运算：设为 None，不进行边界检测
            - 默认值：1（向后兼容，但不推荐使用）

    Returns:
        包装后的函数

    使用示例：
        # arity=1 的时序算子（必须指定 window_size）
        @register_operator(name="sma_20", category="time_series", arity=1)
        @with_boundary_check(window_size=20)
        def sma_20(arr):
            return pd.Series(arr).rolling(20).mean().values

        # arity=2 的相关性算子（必须指定 window_size）
        @register_operator(name="corr_10", category="time_series", arity=2)
        @with_boundary_check(window_size=10)
        def corr_10(arr1, arr2):
            return pd.Series(arr1).rolling(10).corr(pd.Series(arr2))

        # arity=2 的算术运算（设为 None，不进行边界检测）
        @register_operator(name="add", category="basic", arity=2)
        @with_boundary_check(window_size=None)
        def op_add(arr1, arr2):
            return arr1 + arr2
    """

    def decorator(f: Callable) -> Callable:
        import inspect

        # 检查函数签名，确定参数数量
        sig = inspect.signature(f)
        n_params = len(
            [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        )

        # 根据 arity 定义不同的 wrapper
        if n_params == 1:

            @wraps(f)
            def wrapper(arr: np.ndarray) -> np.ndarray:
                result = f(arr)

                # window_size=None 表示不进行边界检测
                if window_size is None:
                    return result

                boundary_indices = get_boundary_indices()
                if not boundary_indices:
                    return result

                arr_length = len(result)
                boundary_mask = np.zeros(arr_length, dtype=bool)

                for b in boundary_indices[1:]:
                    start_idx = b
                    end_idx = min(b + window_size, arr_length)
                    boundary_mask[start_idx:end_idx] = True

                result[boundary_mask] = np.nan
                return result

        elif n_params == 2:

            @wraps(f)
            def wrapper(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
                result = f(arr1, arr2)

                # window_size=None 表示不进行边界检测
                if window_size is None:
                    return result

                boundary_indices = get_boundary_indices()
                if not boundary_indices:
                    return result

                arr_length = len(result)
                boundary_mask = np.zeros(arr_length, dtype=bool)

                for b in boundary_indices[1:]:
                    start_idx = b
                    end_idx = min(b + window_size, arr_length)
                    boundary_mask[start_idx:end_idx] = True

                result[boundary_mask] = np.nan
                return result

        else:
            raise ValueError(f"不支持的参数数量: {n_params}，仅支持 arity=1 或 arity=2")

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


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
