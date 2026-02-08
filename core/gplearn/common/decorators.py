"""GP 装饰器模块。

提供边界检查、面板数据转换等装饰器函数。
"""

from functools import wraps
from typing import Callable
import numpy as np
import pandas as pd
from .state import get_index, get_boundary_indices
from .panel import build_dual_panel


def with_boundary_check(
    func: Callable = None, *, window_size: int | None = 1
) -> Callable:
    """边界检查装饰器。

    WHY: GP 算子计算需要使用历史数据，边界位置样本不足会导致结果不可靠。

    Args:
        func: 被装饰的函数
        window_size: 窗口大小，None 表示不检查

    Returns:
        装饰后的函数

    Example:
        # arity=1 使用 window_size
        @register_operator(name="sma_20", category="time_series", arity=1)
        @with_boundary_check(window_size=20)
        def sma_20(arr):
            return pd.Series(arr).rolling(20).mean().values

        # arity=2 使用 window_size
        @register_operator(name="corr_10", category="time_series", arity=2)
        @with_boundary_check(window_size=10)
        def corr_10(arr1, arr2):
            return pd.Series(arr1).rolling(10).corr(pd.Series(arr2))

        # arity=2 不使用边界检查
        @register_operator(name="add", category="basic", arity=2)
        @with_boundary_check(window_size=None)
        def op_add(arr1, arr2):
            return arr1 + arr2
    """

    def decorator(f: Callable) -> Callable:
        """装饰器内部函数，用于包装被装饰的函数。

        Args:
            f: 被装饰的函数

        Returns:
            包装后的函数
        """
        import inspect

        sig = inspect.signature(f)
        n_params = len(
            [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        )

        if n_params == 1:

            @wraps(f)
            def wrapper(arr: np.ndarray) -> np.ndarray:
                """单参数函数的包装器，处理边界检查。

                Args:
                    arr: 输入数组

                Returns:
                    处理后的数组，边界位置被设为NaN
                """
                result = f(arr)

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
                """双参数函数的包装器，处理边界检查。

                Args:
                    arr1: 第一个输入数组
                    arr2: 第二个输入数组

                Returns:
                    处理后的数组，边界位置被设为NaN
                """
                result = f(arr1, arr2)

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
            raise ValueError(f": {n_params} arity=1  arity=2")

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def with_panel_convert() -> Callable:
    """面板数据转换装饰器，将一维数组转换为面板数据格式。

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        """内部装饰器函数。

        Args:
            func: 被装饰的适应度函数

        Returns:
            包装后的函数
        """

        def wrapper(
            y: np.ndarray,
            y_pred: np.ndarray,
            w: np.ndarray,
        ) -> float:
            """Gplearn metric 兼容的包装器。

            Args:
                y: 真实值数组
                y_pred: 预测值数组
                w: 样本权重（不使用）

            Returns:
                适应度值
            """
            # 检查是否是 gplearn 的验证调用
            if len(y) == 2 and np.array_equal(y, [1, 1]):
                return 0.0

            # 多线程训练时子进程无法访问全局变量
            try:
                index = get_index()
            except RuntimeError:
                return 0.0

            y_true_panel, y_pred_panel = build_dual_panel(y, y_pred, index)

            return func(y_true_panel, y_pred_panel)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__

        return wrapper

    return decorator


def with_panel_builder(func: Callable) -> Callable:
    """面板构建装饰器，将一维数组转换为面板数据后应用函数。

    Args:
        func: 接收面板数据（DataFrame）的函数

    Returns:
        包装后的函数，将一维数组转换为面板数据后调用原函数

    Example:
        @register_operator(name="rank", category="cross_sectional")
        @with_panel_builder
        def cross_sectional_rank(panel):
            return panel.rank(axis=1, pct=True).fillna(0.5)
    """

    @wraps(func)
    def wrapper(arr: np.ndarray) -> np.ndarray:
        """包装器函数，将一维数组转换为面板数据后调用原函数。

        Args:
            arr: 输入的一维数组

        Returns:
            处理后的一维数组
        """
        try:
            index = get_index()
        except RuntimeError:
            # 无法获取索引时，直接返回原数组（非gplearn环境）
            return arr

        # 数组长度与索引长度不匹配时，直接返回原数组
        if len(arr) != len(index):
            return arr

        df = pd.DataFrame({"value": arr}, index=index)
        panel = df["value"].unstack(level=0)

        result_panel = func(panel)

        return result_panel.stack().values

    return wrapper
