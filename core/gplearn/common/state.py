"""全局状态管理模块。"""

from typing import List, Optional, Generator
import pandas as pd
from contextlib import contextmanager

_index_global: Optional[pd.MultiIndex] = None
_boundaries_global: Optional[List[int]] = None


def set_index(index: pd.MultiIndex) -> None:
    """设置全局MultiIndex。

    Args:
        index: MultiIndex对象
    """
    global _index_global
    _index_global = index


def get_index() -> pd.MultiIndex:
    """获取全局MultiIndex。

    Returns:
        全局MultiIndex对象

    Raises:
        RuntimeError: 当索引未设置时
    """
    if _index_global is None:
        raise RuntimeError("Index未设置，请先调用set_index()")
    return _index_global


def set_boundary_indices(indices: List[int]) -> None:
    """设置全局边界索引。

    Args:
        indices: 边界索引列表
    """
    global _boundaries_global
    _boundaries_global = indices


def get_boundary_indices() -> List[int]:
    """获取全局边界索引。

    Returns:
        边界索引列表，未设置时返回空列表
    """
    if _boundaries_global is None:
        return []
    return _boundaries_global


def clear_globals() -> None:
    """清空所有全局状态。"""
    global _index_global, _boundaries_global
    _index_global = None
    _boundaries_global = None


@contextmanager
def global_state(
    index: pd.MultiIndex, boundaries: List[int]
) -> Generator[None, None, None]:
    """设置全局状态的上下文管理器。

    Args:
        index: 多重索引
        boundaries: 边界索引列表

    Yields:
        None
    """
    set_index(index)
    set_boundary_indices(boundaries)
    try:
        yield
    finally:
        clear_globals()


__all__ = [
    "set_index",
    "get_index",
    "set_boundary_indices",
    "get_boundary_indices",
    "clear_globals",
    "global_state",
]
