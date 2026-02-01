"""
Gplearn

 API
>>> from core.gplearn import FactorMiner
>>> miner = FactorMiner(features=[...], target="...")
>>> expressions = miner.run(features_df, target_df)


- miner: GP  MultiIndex DataFrame
- operators:
- fitness:
- common: TLS

 step5/.py  Qlib
"""

__version__ = "2.2.0"

from .miner import FactorMiner

from . import operators  # noqa: F401
from . import fitness  # noqa: F401

__all__ = ["FactorMiner"]
