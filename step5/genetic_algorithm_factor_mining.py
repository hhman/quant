#!/usr/bin/env python3
"""
Step5:
 Gplearn  SymbolicTransformer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import qlib
from qlib.constant import REG_CN
from qlib.data import D

from core.gplearn import FactorMiner
from core.gplearn.config import DEFAULT_FEATURES, DEFAULT_TARGET, GPConfig
from utils.cache_manager import CacheManager


def mine_factors_with_gp(
    market: str,
    start_date: str,
    end_date: str,
    provider_uri: str,
    random_state: int = None,
) -> None:
    """使用遗传规划挖掘因子表达式。

    Parameters:
    -----------
    market : str
        股票池名称
    start_date : str
        开始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    provider_uri : str
        Qlib数据目录
    random_state : int
        随机种子
    """
    if random_state is None:
        import random

        random_state = random.randint(0, 2**32 - 1)
        print(f"  随机种子: {random_state}")

    print(f"初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    cache_mgr = CacheManager(market, start_date, end_date)

    print("\nStep5: GP因子挖掘")
    print(f"  市场: {market}")
    print(f"  日期: {start_date} ~ {end_date}")
    print(f"  特征数: {len(DEFAULT_FEATURES)}")
    print(f"  目标: {DEFAULT_TARGET}")
    print(f"  随机种子: {random_state}")

    instruments = D.instruments(market=market)

    print("  加载特征数据...")
    features_df = D.features(
        instruments=instruments,
        fields=DEFAULT_FEATURES,
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    features_df.columns = DEFAULT_FEATURES
    features_df = features_df.groupby(level="instrument", group_keys=False).apply(
        lambda x: x.ffill().bfill()
    )
    print(f"    特征数据: {features_df.shape}")

    print("  加载收益率数据...")
    ret_df = cache_mgr.read_dataframe("returns")
    ret_df = ret_df[["ret_1d"]]
    ret_df.columns = [DEFAULT_TARGET]
    print(f"    收益率数据: {ret_df.shape}")

    print("  训练GP模型...")
    miner = FactorMiner(
        features=DEFAULT_FEATURES,
        target=DEFAULT_TARGET,
        gp_config=GPConfig(),
        random_state=random_state,
    )

    expressions = miner.run(features_df, ret_df)

    print("  保存结果...")
    output_dir = Path(".cache")
    output_dir.mkdir(exist_ok=True)

    start_compact = start_date.replace("-", "")
    end_compact = end_date.replace("-", "")
    filename = (
        f"{market}_{start_compact}_{end_compact}__gp_seed{random_state}.expression.txt"
    )
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        for expr in expressions:
            f.write(f"{expr}\n")

    print(f"    表达式文件: {output_path}")

    print(f"\n  共生成 {len(expressions)} 个因子:")
    for i, expr in enumerate(expressions, 1):
        print(f"\n  Factor {i}:")
        print(f"    {expr}")

    print("\nStep5完成!")
