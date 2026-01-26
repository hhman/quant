"""全局状态管理模块"""

from typing import List, Optional
import pandas as pd
from contextlib import contextmanager

# 全局变量
_index_global: Optional[pd.MultiIndex] = None
_boundaries_global: Optional[List[int]] = None


def set_index(index: pd.MultiIndex) -> None:
    """设置 MultiIndex 到全局变量"""
    global _index_global
    _index_global = index


def get_index() -> pd.MultiIndex:
    """从全局变量获取 MultiIndex"""
    if _index_global is None:
        raise RuntimeError("Index 未设置，请先调用 set_index()")
    return _index_global


def set_boundary_indices(indices: List[int]) -> None:
    """设置边界索引到全局变量"""
    global _boundaries_global
    _boundaries_global = indices


def get_boundary_indices() -> List[int]:
    """从全局变量获取边界索引"""
    if _boundaries_global is None:
        return []
    return _boundaries_global


def clear_globals() -> None:
    """清除所有全局数据"""
    global _index_global, _boundaries_global
    _index_global = None
    _boundaries_global = None


@contextmanager
def global_state(index: pd.MultiIndex, boundaries: List[int]):
    """全局状态上下文管理器"""
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
