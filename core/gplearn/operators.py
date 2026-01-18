"""
Gplearn 算子库

本模块提供支持数据展平方案的自定义算子，包括时间序列算子和截面算子。

核心机制：
1. 时间序列算子：基于 Pandas Rolling 在一维数组上实现
2. 截面算子：在适应度函数中还原为面板后计算
3. 训练阶段：仅使用可展平的时间序列算子
4. 评估阶段：使用完整的面板算子

设计原则：
- 算子函数必须支持 numpy 数组输入
- 时间序列算子使用固定窗口大小
- 截面算子标记为 "panel_only"，仅在评估时使用

使用示例：
    >>> # 训练阶段（时间序列算子）
    >>> from core.gplearn.operators import rolling_sma
    >>> result = rolling_sma(prices, window=10)
    >>>
    >>> # 评估阶段（面板算子）
    >>> factor_expr = "Rank($close)"
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List

# 全局窗口大小（通过 set_window_size 设置）
_GLOBAL_WINDOW_SIZE = 10


def set_window_size(window_size: int):
    """
    设置全局窗口大小

    Args:
        window_size: 窗口大小
    """
    global _GLOBAL_WINDOW_SIZE
    _GLOBAL_WINDOW_SIZE = window_size


def get_window_size() -> int:
    """
    获取当前全局窗口大小

    Returns:
        窗口大小
    """
    return _GLOBAL_WINDOW_SIZE


# ========== 时间序列算子（训练阶段使用） ==========


def rolling_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """
    简单移动平均 (Simple Moving Average)

    计算公式：
        SMA_t = mean(price_{t-window+1:t})

    Args:
        arr: 价格序列（一维数组）
        window: 窗口大小

    Returns:
        移动平均序列
    """
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).mean().values
    # 替换 NaN 为 0
    result = np.nan_to_num(result, nan=0.0)
    return result


def rolling_ema(arr: np.ndarray, window: int) -> np.ndarray:
    """
    指数移动平均 (Exponential Moving Average)

    计算公式：
        EMA_t = alpha * price_t + (1-alpha) * EMA_{t-1}
        alpha = 2 / (window + 1)

    Args:
        arr: 价格序列
        window: 窗口大小

    Returns:
        指数移动平均序列
    """
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=float)

    series = pd.Series(arr)
    result = series.ewm(span=window, adjust=False).mean().values
    return result


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """
    滚动标准差 (Rolling Standard Deviation)

    计算公式：
        STD_t = std(price_{t-window+1:t})

    Args:
        arr: 价格序列
        window: 窗口大小

    Returns:
        标准差序列
    """
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).std().values
    # 替换 NaN 为 0（处理常数序列的情况）
    result = np.nan_to_num(result, nan=0.0)
    return result


def rolling_momentum(arr: np.ndarray, window: int) -> np.ndarray:
    """
    动量 (Momentum)

    计算公式：
        MOM_t = price_t / price_{t-window} - 1

    Args:
        arr: 价格序列
        window: 窗口大小

    Returns:
        动量序列
    """
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    result = np.zeros_like(arr, dtype=float)
    # 添加小常数避免除零
    result[window:] = arr[window:] / (arr[:-window] + 1e-10) - 1
    return result


def rolling_delta(arr: np.ndarray, window: int = 1) -> np.ndarray:
    """
    一阶差分 (Delta)

    计算公式：
        DELTA_t = price_t - price_{t-window}

    Args:
        arr: 价格序列
        window: 差分步长（默认 1）

    Returns:
        差分序列
    """
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    result = np.zeros_like(arr, dtype=float)
    result[window:] = arr[window:] - arr[:-window]
    return result


def rolling_corr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    滚动相关系数 (Rolling Correlation)

    计算公式：
        CORR_t = corr(arr1_{t-window+1:t}, arr2_{t-window+1:t})

    Args:
        arr1: 序列1
        arr2: 序列2
        window: 窗口大小

    Returns:
        相关系数序列
    """
    if len(arr1) < window or len(arr2) < window:
        return np.full_like(arr1, np.nan, dtype=float)

    series1 = pd.Series(arr1)
    series2 = pd.Series(arr2)
    result = series1.rolling(window=window, min_periods=1).corr(series2).values
    return result


def rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """
    滚动最大值 (Rolling Maximum)

    Args:
        arr: 价格序列
        window: 窗口大小

    Returns:
        最大值序列
    """
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).max().values
    result = np.nan_to_num(result, nan=0.0)
    return result


def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """
    滚动最小值 (Rolling Minimum)

    Args:
        arr: 价格序列
        window: 窗口大小

    Returns:
        最小值序列
    """
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    series = pd.Series(arr)
    result = series.rolling(window=window, min_periods=1).min().values
    result = np.nan_to_num(result, nan=0.0)
    return result


def ts_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """
    时间序列排名 (Time Series Rank)

    计算当前值在窗口内的排名位置。

    计算公式：
        RANK_t = rank(price_t, window=window)

    Args:
        arr: 价格序列
        window: 窗口大小

    Returns:
        排名序列（归一化到 0-1）
    """
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)

    result = np.zeros_like(arr, dtype=float)

    for i in range(window - 1, len(arr)):
        window_data = arr[i - window + 1 : i + 1]
        rank = pd.Series(window_data).rank().iloc[-1]
        result[i] = rank / window  # 归一化到 0-1

    return result


# ========== 截面算子（仅在评估阶段使用） ==========

# 注意：截面算子无法在训练阶段使用，因为它们需要面板数据。
# 这些算子仅用于表达式转换，在评估时通过 Qlib 计算。

CROSS_SECTIONAL_OPERATORS = {
    "Rank": "横截面排名",
    "ZScore": "横截面标准化",
    "Quantile": "分位数",
}


# ========== 算子包装器（适配 Gplearn 接口） ==========


def make_rolling_operator(
    func: Callable,
    window: int = None,
) -> Callable:
    """
    将时间序列算子包装为 Gplearn 可用的函数

    Args:
        func: 原始算子函数
        window: 窗口大小（None 则使用全局窗口）

    Returns:
        包装后的函数
    """
    if window is None:
        window = get_window_size()

    def wrapper(arr: np.ndarray) -> np.ndarray:
        return func(arr, window)

    wrapper.__name__ = func.__name__
    return wrapper


def make_binary_rolling_operator(
    func: Callable,
    window: int = None,
) -> Callable:
    """
    将二元时间序列算子包装为 Gplearn 可用的函数

    Args:
        func: 原始二元算子函数
        window: 窗口大小

    Returns:
        包装后的函数
    """
    if window is None:
        window = get_window_size()

    def wrapper(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        return func(arr1, arr2, window)

    wrapper.__name__ = func.__name__
    return wrapper


# ========== 算子集合（用于 Gplearn 初始化） ==========


def get_time_series_operators(window_size: int = 10) -> List[Callable]:
    """
    获取时间序列算子列表

    Args:
        window_size: 窗口大小

    Returns:
        算子函数列表（gplearn._Function 对象）
    """
    try:
        from gplearn.functions import make_function
    except ImportError:
        raise ImportError("请先安装 gplearn: pip install gplearn")

    set_window_size(window_size)

    # 使用 make_function 包装每个算子
    # 为了避免 gplearn 的严格验证，使用 wrap=False 并确保函数安全
    operators = [
        # 基础算子（最安全的）
        make_function(function=np.abs, name="abs", arity=1, wrap=False),
        make_function(
            function=lambda x: np.sqrt(np.abs(x) + 1e-10),
            name="sqrt",
            arity=1,
            wrap=False,
        ),
        make_function(
            function=lambda x: np.log(np.abs(x) + 1.0), name="log", arity=1, wrap=False
        ),
        # 时间序列算子（固定窗口，添加保护）
        make_function(
            function=lambda x: rolling_sma(x, window_size),
            name="sma",
            arity=1,
            wrap=False,
        ),
        make_function(
            function=lambda x: rolling_std(x, window_size),
            name="std",
            arity=1,
            wrap=False,
        ),
        make_function(
            function=lambda x: rolling_max(x, window_size),
            name="max",
            arity=1,
            wrap=False,
        ),
        make_function(
            function=lambda x: rolling_min(x, window_size),
            name="min",
            arity=1,
            wrap=False,
        ),
        make_function(
            function=lambda x: ts_rank(x, window_size),
            name="ts_rank",
            arity=1,
            wrap=False,
        ),
        make_function(
            function=lambda x: rolling_momentum(x, window_size),
            name="momentum",
            arity=1,
            wrap=False,
        ),
        make_function(
            function=lambda x: rolling_delta(x, 1), name="delta", arity=1, wrap=False
        ),
    ]

    return operators


def get_operator_mapping() -> Dict[str, str]:
    """
    获取算子名称映射表

    注意：此函数仅保留用于参考，实际系统直接使用 Gplearn 表达式。
    如需转换为 Qlib 格式，可由 LLM agent 处理。

    Returns:
        算子映射字典 {gplearn_name: qlib_name}
    """
    return {
        "rolling_sma": "Ref($close, 1).mean(10)",
        "rolling_ema": "Ref($close, 1).ewm(10)",
        "rolling_std": "Ref($close, 1).std(10)",
        "rolling_momentum": "Ref($close, 10) / Ref($close, 0) - 1",
        "rolling_delta": "Ref($close, 1) - Ref($close, 0)",
        "rolling_max": "Ref($close, 1).max(10)",
        "rolling_min": "Ref($close, 1).min(10)",
        "ts_rank": "Ref($close, 1).ts_rank(10)",
        "abs": "Abs",
        "sqrt": "Sqrt",
        "log": "Log",
        "add": "+",
        "sub": "-",
        "mul": "Mul",
        "div": "Div",
    }
