"""
Gplearn 遗传算法因子挖掘核心模块

极简 API：
>>> from core.gplearn import FactorMiner
>>> miner = FactorMiner(features=[...], target="...")
>>> expressions = miner.run(features_df, target_df)

架构说明：
- miner: GP 挖掘器（接收 MultiIndex DataFrame）
- operators: 算子库（时间序列算子）
- fitness: 适应度函数库
- common: 共享基础框架（TLS、注册机制、面板数据转换）

数据加载由 step5/遗传算法因子挖掘.py 提供（与 Qlib 解耦）。
"""

__version__ = "2.2.0"

from .miner import FactorMiner

from . import operators  # noqa: F401
from . import fitness  # noqa: F401

__all__ = ["FactorMiner"]
