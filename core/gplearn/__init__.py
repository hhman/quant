"""GP 因子挖掘模块。

使用遗传规划自动发现因子表达式。

API 示例:
    >>> from core.gplearn import FactorMiner
    >>> miner = FactorMiner(features=[...], target="...")
    >>> expressions = miner.run(features_df, target_df)

模块组成:
    - miner: GP 因子挖掘器，接受 MultiIndex DataFrame 输入
    - operators: 算子注册表
    - fitness: 适应度函数
    - common: 公共工具（装饰器、注册表、状态管理）

使用场景:
    见 step5/genetic_algorithm_factor_mining.py，需先运行 Qlib 初始化
"""

__version__ = "2.2.0"

from .miner import FactorMiner

from . import operators  # noqa: F401
from . import fitness  # noqa: F401

__all__ = ["FactorMiner"]
