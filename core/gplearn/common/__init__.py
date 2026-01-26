"""
Gplearn 共享基础框架

提供算子和适应度函数模块共享的基础设施：
- 全局状态管理
- 注册机制
- 面板数据转换
- 通用装饰器
"""

from .state import (
    set_index,
    get_index,
    set_boundary_indices,
    get_boundary_indices,
)
from .registry import (
    create_registry,
    register,
    get,
    list_all,
    get_meta,
)
from .panel import (
    flatten_to_panel,
    panel_to_flatten,
    clean_panel,
    build_dual_panel,
    dataframe_to_flatten,
    calc_boundaries,
)
from .decorators import (
    with_boundary_check,
    with_panel_convert,
)

__all__ = [
    # 全局状态管理
    "set_index",
    "get_index",
    "set_boundary_indices",
    "get_boundary_indices",
    # 注册机制
    "create_registry",
    "register",
    "get",
    "list_all",
    "get_meta",
    # 面板数据转换
    "flatten_to_panel",
    "panel_to_flatten",
    "clean_panel",
    "build_dual_panel",
    "dataframe_to_flatten",
    "calc_boundaries",
    # 装饰器
    "with_boundary_check",
    "with_panel_convert",
    "with_panel_builder",
]
