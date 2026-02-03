"""注册表管理模块，用于管理算子和适应度函数的注册。"""

from typing import Dict, Callable, Any, List
import inspect
import numpy as np


def create_registry(name: str) -> Dict[str, Dict[str, Any]]:
    """创建空的注册表字典。

    Args:
        name: 注册表名称（用于标识和调试）

    Returns:
        空的注册表字典
    """
    return {}


def register(registry: Dict[str, Dict[str, Any]], name: str, **meta) -> Callable:
    """创建装饰器函数，将函数注册到注册表。

    Args:
        registry: 注册表字典
        name: 注册名称
        **meta: 元数据（包含category、arity等字段）

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        """装饰器内部函数，注册函数到注册表。

        Args:
            func: 被装饰的函数

        Returns:
            原函数（未修改）
        """
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
    """从注册表中获取函数。

    Args:
        registry: 注册表字典
        name: 注册名称
        registry_name: 注册表名称（用于错误提示）

    Returns:
        注册的函数

    Raises:
        KeyError: 当名称不存在时
    """
    if name not in registry:
        raise KeyError(
            f"{registry_name}: 未找到 '{name}'。可用选项: {list(registry.keys())}"
        )
    return registry[name]["function"]


def list_all(registry: Dict[str, Dict[str, Any]]) -> List[str]:
    """列出注册表中的所有名称。

    Args:
        registry: 注册表字典

    Returns:
        所有注册名称的列表
    """
    return list(registry.keys())


def get_meta(registry: Dict[str, Dict[str, Any]], name: str) -> Dict[str, Any]:
    """获取注册项的元数据。

    Args:
        registry: 注册表字典
        name: 注册名称

    Returns:
        元数据字典的副本
    """
    return registry[name].copy()


# ==================== 全局注册表 ====================

_OPERATOR_REGISTRY = None
_FITNESS_REGISTRY = None


def _get_operator_registry() -> Dict[str, Dict[str, Any]]:
    """获取全局算子注册表。"""
    global _OPERATOR_REGISTRY
    if _OPERATOR_REGISTRY is None:
        _OPERATOR_REGISTRY = create_registry("Operator")
    return _OPERATOR_REGISTRY


def _get_fitness_registry() -> Dict[str, Dict[str, Any]]:
    """获取全局适应度函数注册表。"""
    global _FITNESS_REGISTRY
    if _FITNESS_REGISTRY is None:
        _FITNESS_REGISTRY = create_registry("Fitness")
    return _FITNESS_REGISTRY


# ==================== 算子注册接口 ====================


def register_operator(name: str, category: str = "time_series", **meta) -> Callable:
    """注册算子函数。

    Args:
        name: 算子名称
        category: 算子类别（time_series或cross_sectional）
        **meta: 其他元数据（arity等）

    Returns:
        装饰器函数
    """
    registry = _get_operator_registry()
    return register(registry, name, category=category, **meta)


def get_operator(name: str) -> Callable:
    """获取注册的算子函数。

    Args:
        name: 算子名称

    Returns:
        算子函数
    """
    registry = _get_operator_registry()
    return get(registry, name, "Operator")


def list_operators() -> List[str]:
    """列出所有已注册的算子名称。

    Returns:
        算子名称列表
    """
    registry = _get_operator_registry()
    return list_all(registry)


def _get_operator_meta(name: str) -> Dict[str, Any]:
    """获取算子的元数据。

    Args:
        name: 算子名称

    Returns:
        元数据字典
    """
    registry = _get_operator_registry()
    return get_meta(registry, name)


# ==================== 适应度函数注册接口 ====================


def register_fitness(name: str, stopping_criteria: float = None, **meta) -> Callable:
    """注册适应度函数。

    Args:
        name: 函数名称
        stopping_criteria: 推荐的早停阈值（与 fitness 返回值同量纲）
        **meta: 其他元数据

    Returns:
        装饰器函数
    """
    registry = _get_fitness_registry()
    if stopping_criteria is not None:
        meta["stopping_criteria"] = stopping_criteria
    return register(registry, name, **meta)


def _get_fitness_raw(name: str) -> Callable:
    """获取原始的适应度函数。

    Args:
        name: 函数名称

    Returns:
        适应度函数
    """
    registry = _get_fitness_registry()
    return get(registry, name, "Fitness")


def list_fitness() -> List[str]:
    """列出所有已注册的适应度函数名称。

    Returns:
        函数名称列表
    """
    registry = _get_fitness_registry()
    return list_all(registry)


def _get_fitness_meta(name: str) -> Dict[str, Any]:
    """获取适应度函数的元数据。

    Args:
        name: 函数名称

    Returns:
        元数据字典
    """
    registry = _get_fitness_registry()
    return get_meta(registry, name)


# ==================== Gplearn 适配器 ====================


def _validate_window_param(w: Any, default: int = 20) -> int:
    """验证并规范化窗口参数。

    Args:
        w: 窗口参数（可以是数组或标量）
        default: 默认窗口大小

    Returns:
        有效的窗口大小
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
    """将arity=1的函数适配为gplearn格式。

    Args:
        func: 原始函数
        name: 函数名称

    Returns:
        gplearn格式的函数
    """
    try:
        from gplearn.functions import make_function
    except ImportError:
        raise ImportError("请先安装gplearn: pip install gplearn")

    return make_function(function=func, name=name, arity=1, wrap=False)


def _adapt_to_gplearn_arity2(func: Callable, name: str) -> Callable:
    """将arity=2的函数适配为gplearn格式。

    Args:
        func: 原始函数
        name: 函数名称

    Returns:
        gplearn格式的函数
    """
    try:
        from gplearn.functions import make_function
    except ImportError:
        raise ImportError("请先安装gplearn: pip install gplearn")

    return make_function(function=func, name=name, arity=2, wrap=False)


def adapt_operator_to_gplearn(func: Callable, arity: int, name: str) -> Callable:
    """将算子函数适配为gplearn格式。

    Args:
        func: 原始函数
        arity: 参数数量
        name: 函数名称

    Returns:
        gplearn格式的函数

    Raises:
        ValueError: 当arity不支持时
    """
    if arity == 1:
        return _adapt_to_gplearn_arity1(func, name)
    elif arity == 2:
        return _adapt_to_gplearn_arity2(func, name)
    else:
        raise ValueError(f"不支持的arity: {arity}")


# ==================== 算子导出 ====================


def get_all_operators() -> List[Callable]:
    """获取所有已注册算子的gplearn格式列表。

    Returns:
        gplearn格式的算子函数列表
    """
    try:
        import importlib.util

        if importlib.util.find_spec("gplearn") is None:
            raise ImportError("请先安装gplearn: pip install gplearn")
    except ImportError:
        raise ImportError("请先安装gplearn: pip install gplearn")

    operator_names = list_operators()

    if not operator_names:
        import warnings

        warnings.warn("没有已注册的算子")
        return []

    operators = []
    for op_name in operator_names:
        func = get_operator(op_name)
        meta = _get_operator_meta(op_name)
        arity = meta["arity"]
        category = meta.get("category", "")

        if hasattr(func, "__wrapped__") and category != "cross_sectional":
            original_func = func.__wrapped__
        else:
            original_func = func

        gplearn_func = adapt_operator_to_gplearn(original_func, arity, op_name)
        operators.append(gplearn_func)

    return operators


def list_registered_operators() -> List[Dict[str, Any]]:
    """列出所有已注册算子的信息。

    Returns:
        算子信息列表（包含name、arity、category）
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
    """获取指定类别的算子列表。

    Args:
        category: 算子类别

    Returns:
        gplearn格式的算子函数列表
    """
    operator_names = list_operators()
    operators = []

    for op_name in operator_names:
        meta = _get_operator_meta(op_name)
        if meta["category"] == category:
            func = get_operator(op_name)
            if hasattr(func, "__wrapped__") and category != "cross_sectional":
                original_func = func.__wrapped__
            else:
                original_func = func
            gplearn_func = adapt_operator_to_gplearn(
                original_func, meta["arity"], op_name
            )
            operators.append(gplearn_func)

    return operators


# ==================== 适应度函数导出 ====================


def get_fitness(name: str, **kwargs) -> Callable:
    """获取适应度函数并绑定参数。

    Args:
        name: 函数名称
        **kwargs: 绑定的参数

    Returns:
        绑定参数后的函数
    """
    import functools

    func = _get_fitness_raw(name)
    return functools.partial(func, **kwargs)


def list_registered_fitness() -> List[str]:
    """列出所有已注册的适应度函数名称。

    Returns:
        函数名称列表
    """
    return list_fitness()
