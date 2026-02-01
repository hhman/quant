"""
Gplearn


"""

import numpy as np
import pandas as pd
import warnings

from .common.registry import (
    register_operator,
    get_operator,
    list_operators,
    get_all_operators,
    list_registered_operators,
    get_operators_by_category,
)
from .common.decorators import with_boundary_check, with_panel_builder


# ==================== TA-Lib  ====================

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print(": TA-Lib  TA-Lib ")


#
WINDOWS_SHORT = [5, 10]
WINDOWS_MEDIUM = [20, 60]
WINDOWS_LONG = [120, 250]
ALL_WINDOWS = WINDOWS_SHORT + WINDOWS_MEDIUM + WINDOWS_LONG
WINDOWS_RSI = [6, 12, 24]
WINDOWS_ROC = [5, 10, 20]


def _create_talib_operator(
    name: str,
    talib_func: callable,
    window: int,
    category: str,
    multi_output_index: int | None = None,
    **kwargs: object,
) -> None:
    """TA-Lib arity=1"""
    if not TALIB_AVAILABLE:
        return

    boundary_window = window if window > 0 else 1

    @register_operator(name=name, category=category, arity=1)
    @with_boundary_check(window_size=boundary_window)
    def operator(arr: np.ndarray) -> np.ndarray:
        """动态生成的单参数算子函数。

        Args:
            arr: 输入数组

        Returns:
            处理后的数组
        """
        try:
            params = {"timeperiod": window, **kwargs} if window > 0 else kwargs
            result = talib_func(arr, **params)
            if multi_output_index is not None and isinstance(result, tuple):
                result = result[multi_output_index]
            return np.nan_to_num(result, nan=0.0)
        except Exception as e:
            warnings.warn(
                f"TA-Lib  {name} : {type(e).__name__}: {str(e)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros_like(arr, dtype=float)


# ====================  ====================


@register_operator(name="abs", category="basic", arity=1)
def op_abs(arr: np.ndarray) -> np.ndarray:
    """计算绝对值。

    Args:
        arr: 输入数组

    Returns:
        绝对值数组
    """
    return np.abs(arr)


@register_operator(name="sqrt", category="basic", arity=1)
def op_sqrt(arr: np.ndarray) -> np.ndarray:
    """计算平方根（带数值稳定性处理）。

    Args:
        arr: 输入数组

    Returns:
        平方根数组
    """
    return np.sqrt(np.abs(arr) + 1e-10)


@register_operator(name="log", category="basic", arity=1)
def op_log(arr: np.ndarray) -> np.ndarray:
    """计算自然对数（带数值稳定性处理）。

    Args:
        arr: 输入数组

    Returns:
        对数值数组
    """
    return np.log(np.abs(arr) + 1.0)


@register_operator(name="sign", category="basic", arity=1)
def op_sign(arr: np.ndarray) -> np.ndarray:
    """计算符号函数。

    Args:
        arr: 输入数组

    Returns:
        符号数组（-1, 0, 1）
    """
    return np.sign(arr)


# ==================== arity=2====================


@register_operator(name="add", category="basic", arity=2)
@with_boundary_check(window_size=None)
def op_add(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """数组加法。

    Args:
        arr1: 第一个数组
        arr2: 第二个数组

    Returns:
        加法结果数组
    """
    return arr1 + arr2


@register_operator(name="sub", category="basic", arity=2)
@with_boundary_check(window_size=None)
def op_sub(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """数组减法。

    Args:
        arr1: 第一个数组
        arr2: 第二个数组

    Returns:
        减法结果数组
    """
    return arr1 - arr2


@register_operator(name="mul", category="basic", arity=2)
@with_boundary_check(window_size=None)
def op_mul(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """数组乘法。

    Args:
        arr1: 第一个数组
        arr2: 第二个数组

    Returns:
        乘法结果数组
    """
    return arr1 * arr2


@register_operator(name="div", category="basic", arity=2)
@with_boundary_check(window_size=None)
def op_div(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """数组除法（带除零保护）。

    Args:
        arr1: 被除数数组
        arr2: 除数数组

    Returns:
        除法结果数组
    """
    return np.divide(arr1, arr2 + 1e-10)


# ==================== arity=1====================


def _create_sma_operator(name: str, window: int) -> None:
    """创建简单移动平均算子。

    Args:
        name: 算子名称
        window: 窗口大小
    """
    if TALIB_AVAILABLE:
        _create_talib_operator(name, talib.SMA, window, "time_series")
    else:

        @register_operator(name=name, category="time_series", arity=1)
        @with_boundary_check(window_size=window)
        def operator(arr: np.ndarray) -> np.ndarray:
            """计算简单移动平均。"""
            series = pd.Series(arr)
            result = series.rolling(window=window, min_periods=1).mean().values
            return np.nan_to_num(result, nan=0.0)


def _create_ema_operator(name: str, window: int) -> None:
    """创建指数移动平均算子。

    Args:
        name: 算子名称
        window: 窗口大小
    """
    if TALIB_AVAILABLE:
        _create_talib_operator(name, talib.EMA, window, "time_series")
    else:

        @register_operator(name=name, category="time_series", arity=1)
        @with_boundary_check(window_size=window)
        def operator(arr: np.ndarray) -> np.ndarray:
            """计算指数移动平均。"""
            series = pd.Series(arr)
            result = series.ewm(span=window, adjust=False).mean().values
            return np.nan_to_num(result, nan=0.0)


def _create_std_operator(name: str, window: int) -> None:
    """创建滚动标准差算子。

    Args:
        name: 算子名称
        window: 窗口大小
    """
    if TALIB_AVAILABLE:
        _create_talib_operator(name, talib.STDDEV, window, "time_series")
    else:

        @register_operator(name=name, category="time_series", arity=1)
        @with_boundary_check(window_size=window)
        def operator(arr: np.ndarray) -> np.ndarray:
            """计算滚动标准差。"""
            series = pd.Series(arr)
            result = series.rolling(window=window, min_periods=1).std().values
            return np.nan_to_num(result, nan=0.0)


def _create_delta_operator(name: str, window: int) -> None:
    """创建差分算子。

    Args:
        name: 算子名称
        window: 窗口大小
    """

    @register_operator(name=name, category="time_series", arity=1)
    @with_boundary_check(window_size=window)
    def operator(arr: np.ndarray) -> np.ndarray:
        """计算差分值。"""
        result = np.zeros_like(arr, dtype=float)
        result[window:] = arr[window:] - arr[:-window]
        return result


def _create_max_operator(name: str, window: int) -> None:
    """创建滚动最大值算子。

    Args:
        name: 算子名称
        window: 窗口大小
    """

    @register_operator(name=name, category="time_series", arity=1)
    @with_boundary_check(window_size=window)
    def operator(arr: np.ndarray) -> np.ndarray:
        """计算滚动最大值。"""
        series = pd.Series(arr)
        result = series.rolling(window=window, min_periods=1).max().values
        return np.nan_to_num(result, nan=0.0)


def _create_min_operator(name: str, window: int) -> None:
    """创建滚动最小值算子。

    Args:
        name: 算子名称
        window: 窗口大小
    """

    @register_operator(name=name, category="time_series", arity=1)
    @with_boundary_check(window_size=window)
    def operator(arr: np.ndarray) -> np.ndarray:
        """计算滚动最小值。"""
        series = pd.Series(arr)
        result = series.rolling(window=window, min_periods=1).min().values
        return np.nan_to_num(result, nan=0.0)


def _create_ts_rank_operator(name: str, window: int) -> None:
    """创建时间序列排名算子。

    Args:
        name: 算子名称
        window: 窗口大小
    """

    @register_operator(name=name, category="time_series", arity=1)
    @with_boundary_check(window_size=window)
    def operator(arr: np.ndarray) -> np.ndarray:
        """计算滚动窗口内当前值的排名百分比。"""
        result = np.zeros_like(arr, dtype=float)
        for i in range(window - 1, len(arr)):
            window_data = arr[i - window + 1 : i + 1]
            rank = pd.Series(window_data).rank().iloc[-1]
            result[i] = rank / window
        return result


#
for w in ALL_WINDOWS:
    _create_sma_operator(f"sma_{w}", w)
    _create_ema_operator(f"ema_{w}", w)

for w in WINDOWS_SHORT + WINDOWS_MEDIUM:
    _create_std_operator(f"std_{w}", w)
    _create_delta_operator(f"delta_{w}", w)
    _create_max_operator(f"max_{w}", w)
    _create_min_operator(f"min_{w}", w)
    _create_ts_rank_operator(f"ts_rank_{w}", w)


# ==================== arity=2====================


def _create_corr_operator(name: str, window: int) -> None:
    """创建相关性算子函数。

    Args:
        name: 算子名称
        window: 窗口大小
    """

    @register_operator(name=name, category="time_series", arity=2)
    @with_boundary_check(window_size=window)
    def operator(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """计算两个数组的滚动相关性。

        Args:
            arr1: 第一个数组
            arr2: 第二个数组

        Returns:
            相关性数组
        """
        # 常数序列相关性为0
        if len(arr1) > 0 and len(arr2) > 0:
            arr1_const = np.all(arr1 == arr1.flat[0])
            arr2_const = np.all(arr2 == arr2.flat[0])
            if arr1_const or arr2_const:
                return np.zeros_like(arr1)

        series1 = pd.Series(arr1)
        series2 = pd.Series(arr2)
        result = series1.rolling(window=window, min_periods=1).corr(series2).values
        return np.nan_to_num(result, nan=0.0)


#  corr
for w in [10, 20]:
    _create_corr_operator(f"corr_{w}", w)


# ====================  TA-Lib arity=1====================

if TALIB_AVAILABLE:
    #
    for w in WINDOWS_RSI:
        _create_talib_operator(f"rsi_{w}", talib.RSI, w, "momentum")

    for w in WINDOWS_ROC:
        _create_talib_operator(f"roc_{w}", talib.ROC, w, "momentum")
        _create_talib_operator(f"rocp_{w}", talib.ROCP, w, "momentum")
        _create_talib_operator(f"mom_{w}", talib.MOM, w, "momentum")

    #
    for w in WINDOWS_SHORT + WINDOWS_MEDIUM:
        _create_talib_operator(f"wma_{w}", talib.WMA, w, "trend")

    for w in [20, 60]:
        _create_talib_operator(f"dema_{w}", talib.DEMA, w, "trend")
        _create_talib_operator(f"tema_{w}", talib.TEMA, w, "trend")

    _create_talib_operator(
        "macd", talib.MACD, 0, "trend", 0, fastperiod=12, slowperiod=26, signalperiod=9
    )
    _create_talib_operator(
        "macd_signal",
        talib.MACD,
        0,
        "trend",
        1,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )
    _create_talib_operator(
        "macd_hist",
        talib.MACD,
        0,
        "trend",
        2,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )

    #
    for w in [20]:
        _create_talib_operator(
            f"bbands_upper_{w}", talib.BBANDS, w, "volatility", 0, nbdevup=2, nbdevdn=2
        )
        _create_talib_operator(
            f"bbands_middle_{w}", talib.BBANDS, w, "volatility", 1, nbdevup=2, nbdevdn=2
        )
        _create_talib_operator(
            f"bbands_lower_{w}", talib.BBANDS, w, "volatility", 2, nbdevup=2, nbdevdn=2
        )


# ====================  ====================


@register_operator(name="rank", category="cross_sectional", arity=1)
@with_panel_builder
def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """计算横截面排名。

    Args:
        panel: Panel格式数据

    Returns:
        排名百分比的DataFrame
    """
    return panel.rank(axis=1, pct=True).fillna(0.5)


@register_operator(name="zscore", category="cross_sectional", arity=1)
@with_panel_builder
def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """计算横截面Z-score标准化。

    Args:
        panel: Panel格式数据

    Returns:
        Z-score标准化后的DataFrame
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    result = (panel.sub(mean, axis=0)).div(std, axis=0)
    return result.fillna(0)


__all__ = [
    "register_operator",
    "get_operator",
    "list_operators",
    "get_all_operators",
    "list_registered_operators",
    "get_operators_by_category",
]
