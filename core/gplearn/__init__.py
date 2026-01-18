"""
Gplearn 遗传算法因子挖掘核心模块

本模块提供基于 Gplearn 的自动化因子挖掘功能，采用数据展平方案
解决三维面板数据适配问题，实现时间序列因子的自动发现。

核心组件：
- data_adapter: 数据适配器（展平/还原）
- fitness: 适应度函数库
- operators: 算子库（时间序列）
- miner: 主挖掘器

使用示例：
    >>> from core.gplearn import GplearnFactorMiner
    >>> miner = GplearnFactorMiner(
    ...     market='csi300',
    ...     start_date='2023-01-01',
    ...     end_date='2024-12-31',
    ...     base_features=['$close', '$volume', '$total_mv']
    ... )
    >>> factors = miner.mine_factors()
    >>>
    >>> # 使用挖掘出的因子
    >>> for factor in factors:
    ...     program = factor['program']
    ...     factor_values = program.execute(X_test)
    ...
    >>> # 或者直接查看表达式（Gplearn 格式）
    >>> print(factor['expression'])
    >>> # 输出: ts_rank(momentum(min(std(-0.021))))
"""

__all__ = [
    # 核心类
    "GplearnDataAdapter",
    "GplearnFactorMiner",
    # 适应度函数
    "RankICFitness",
    "WeightedICFitness",
    "CompositeICFitness",
    # 异常类
    "GplearnDataError",
    "BoundaryPollutionError",
    "ExpressionConversionError",
]

# 导入核心类
from .data_adapter import GplearnDataAdapter
from .miner import GplearnFactorMiner

# 导入适应度函数
from .fitness import RankICFitness, WeightedICFitness, CompositeICFitness

# 导入异常类
from .exceptions import (
    GplearnDataError,
    BoundaryPollutionError,
    ExpressionConversionError,
)

__version__ = "1.0.0"
