"""
注册表管理模块

提供通用的注册机制和算子/适应度函数的管理：
- 统一的注册表工厂
- 通用的注册装饰器
- 元数据管理
- 算子注册表管理
- 适应度函数注册表管理
- Gplearn 适配层
"""

from typing import Dict, Callable, Any, List
import inspect
import numpy as np


def create_registry(name: str) -> Dict[str, Dict[str, Any]]:
    """
    创建注册表

    Args:
        name: 注册表名称（用于标识）

    Returns:
        空的注册表字典
    """
    return {}


def register(registry: Dict[str, Dict[str, Any]], name: str, **meta) -> Callable:
    """
    通用注册装饰器

    Args:
        registry: 注册表字典
        name: 注册名称
        **meta: 元数据（如 category, arity 等）

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        # 自动推断 arity（如果未提供）
        if "arity" not in meta:
            sig = inspect.signature(func)
            arity = len(
                [
                    p
                    for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
            )
            meta["arity"] = arity

        registry[name] = {"function": func, "name": name, **meta}
        return func

    return decorator


def get(
    registry: Dict[str, Dict[str, Any]], name: str, registry_name: str = "Registry"
) -> Callable:
    """
    从注册表获取函数

    Args:
        registry: 注册表字典
        name: 函数名称
        registry_name: 注册表名称（用于错误提示）

    Returns:
        注册的函数
    """
    if name not in registry:
        raise KeyError(
            f"{registry_name}: 未注册的名称 '{name}'. 可用选项: {list(registry.keys())}"
        )
    return registry[name]["function"]


def list_all(registry: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    列出注册表所有名称

    Args:
        registry: 注册表字典

    Returns:
        名称列表
    """
    return list(registry.keys())


def get_meta(registry: Dict[str, Dict[str, Any]], name: str) -> Dict[str, Any]:
    """
    获取元数据

    Args:
        registry: 注册表字典
        name: 名称

    Returns:
        元数据字典
    """
    return registry[name].copy()


# ==================== 注册表实例管理 ====================

_OPERATOR_REGISTRY = None
_FITNESS_REGISTRY = None


def _get_operator_registry() -> Dict[str, Dict[str, Any]]:
    """获取算子注册表（模块级单例）"""
    global _OPERATOR_REGISTRY
    if _OPERATOR_REGISTRY is None:
        _OPERATOR_REGISTRY = create_registry("Operator")
    return _OPERATOR_REGISTRY


def _get_fitness_registry() -> Dict[str, Dict[str, Any]]:
    """获取适应度函数注册表（模块级单例）"""
    global _FITNESS_REGISTRY
    if _FITNESS_REGISTRY is None:
        _FITNESS_REGISTRY = create_registry("Fitness")
    return _FITNESS_REGISTRY


# ==================== 算子注册管理 ====================


def register_operator(name: str, category: str = "time_series", **meta) -> Callable:
    """
    算子注册装饰器

    Args:
        name: 算子名称
        category: 算子类别
        **meta: 其他元数据（如 arity）

    Returns:
        装饰器函数
    """
    registry = _get_operator_registry()
    return register(registry, name, category=category, **meta)


def get_operator(name: str) -> Callable:
    """
    获取算子函数

    Args:
        name: 算子名称

    Returns:
        算子函数
    """
    registry = _get_operator_registry()
    return get(registry, name, "Operator")


def list_operators() -> List[str]:
    """列出所有算子名称"""
    registry = _get_operator_registry()
    return list_all(registry)


def _get_operator_meta(name: str) -> Dict[str, Any]:
    """获取算子元数据（内部函数）"""
    registry = _get_operator_registry()
    return get_meta(registry, name)


# ==================== 适应度函数注册管理 ====================


def register_fitness(name: str, **meta) -> Callable:
    """
    适应度函数注册装饰器

    Args:
        name: 函数名称
        **meta: 其他元数据

    Returns:
        装饰器函数
    """
    registry = _get_fitness_registry()
    return register(registry, name, **meta)


def _get_fitness_raw(name: str) -> Callable:
    """获取适应度函数原始版本（内部函数）"""
    registry = _get_fitness_registry()
    return get(registry, name, "Fitness")


def list_fitness() -> List[str]:
    """列出所有适应度函数名称"""
    registry = _get_fitness_registry()
    return list_all(registry)


def _get_fitness_meta(name: str) -> Dict[str, Any]:
    """获取适应度函数元数据（内部函数）"""
    registry = _get_fitness_registry()
    return get_meta(registry, name)


# ==================== Gplearn 适配层 ====================


def _validate_window_param(w: Any, default: int = 20) -> int:
    """
    验证并标准化 window 参数

    Args:
        w: window 参数（可能是标量或数组）
        default: 默认值

    Returns:
        标准化后的 window 值
    """
    if isinstance(w, np.ndarray):
        if len(w) > 0 and not np.isnan(w[0]) and not np.isinf(w[0]):
            return max(1, int(w[0]))
        else:
            return default
    else:
        if np.isnan(w) or np.isinf(w):
            return default
        else:
            return max(1, int(np.clip(w, 1, 250)))


def _adapt_to_gplearn_arity1(func: Callable, name: str) -> Callable:
    """将 arity=1 的函数适配为 gplearn 函数"""
    try:
        from gplearn.functions import make_function
    except ImportError:
        raise ImportError("请先安装 gplearn: pip install gplearn")

    return make_function(function=func, name=name, arity=1, wrap=False)


def _adapt_to_gplearn_arity2(func: Callable, name: str) -> Callable:
    """将 arity=2 的函数适配为 gplearn 函数"""
    try:
        from gplearn.functions import make_function
    except ImportError:
        raise ImportError("请先安装 gplearn: pip install gplearn")

    def _wrapper(x, w):
        w_val = _validate_window_param(w)
        return func(x, w_val)

    _wrapper.__name__ = name
    return make_function(function=_wrapper, name=name, arity=2, wrap=False)


def _adapt_to_gplearn_arity3(func: Callable, name: str) -> Callable:
    """将 arity=3 的函数适配为 gplearn 函数"""
    try:
        from gplearn.functions import make_function
    except ImportError:
        raise ImportError("请先安装 gplearn: pip install gplearn")

    def _wrapper(x, y, w):
        w_val = _validate_window_param(w)
        return func(x, y, w_val)

    _wrapper.__name__ = name
    return make_function(function=_wrapper, name=name, arity=3, wrap=False)


def adapt_operator_to_gplearn(func: Callable, arity: int, name: str) -> Callable:
    """
    将算子函数适配为 gplearn 函数对象

    Args:
        func: 原始算子函数
        arity: 参数数量
        name: 函数名称

    Returns:
        gplearn 函数对象
    """
    if arity == 1:
        return _adapt_to_gplearn_arity1(func, name)
    elif arity == 2:
        return _adapt_to_gplearn_arity2(func, name)
    elif arity == 3:
        return _adapt_to_gplearn_arity3(func, name)
    else:
        raise ValueError(f"不支持的 arity: {arity}")


# ==================== 算子查询与管理 ====================


def get_all_operators() -> List[Callable]:
    """
    获取所有 Gplearn 兼容的算子

    Returns:
        gplearn 函数对象列表
    """
    try:
        import importlib.util

        if importlib.util.find_spec("gplearn") is None:
            raise ImportError("请先安装 gplearn: pip install gplearn")
    except ImportError:
        raise ImportError("请先安装 gplearn: pip install gplearn")

    operator_names = list_operators()

    if not operator_names:
        import warnings

        warnings.warn("算子注册表为空！")
        return []

    operators = []
    for op_name in operator_names:
        func = get_operator(op_name)
        meta = _get_operator_meta(op_name)
        arity = meta["arity"]

        gplearn_func = adapt_operator_to_gplearn(func, arity, op_name)
        operators.append(gplearn_func)

    return operators


def list_registered_operators() -> List[Dict[str, Any]]:
    """
    列出所有已注册算子的元数据

    Returns:
        算子元数据列表
    """
    operator_names = list_operators()
    return [
        {
            "name": name,
            "arity": _get_operator_meta(name)["arity"],
            "category": _get_operator_meta(name)["category"],
        }
        for name in operator_names
    ]


def get_operators_by_category(category: str) -> List[Callable]:
    """
    按类别获取算子

    Args:
        category: 算子类别

    Returns:
        gplearn 函数对象列表
    """
    operator_names = list_operators()
    operators = []

    for op_name in operator_names:
        meta = _get_operator_meta(op_name)
        if meta["category"] == category:
            func = get_operator(op_name)
            gplearn_func = adapt_operator_to_gplearn(func, meta["arity"], op_name)
            operators.append(gplearn_func)

    return operators


# ==================== 适应度函数管理 ====================


def get_fitness(name: str, **kwargs) -> Callable:
    """
    获取适应度函数（部分应用参数）

    Args:
        name: 函数名称
        **kwargs: 默认参数

    Returns:
        部分应用的函数
    """
    import functools

    func = _get_fitness_raw(name)
    return functools.partial(func, **kwargs)


def list_registered_fitness() -> List[str]:
    """列出所有适应度函数名称"""
    return list_fitness()
