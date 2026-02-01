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
    """


    Parameters:
    -----------
    market : str

    start_date : str
         (YYYY-MM-DD)
    end_date : str
         (YYYY-MM-DD)
    provider_uri : str
        Qlib
    random_state : int

    """
    #
    if random_state is None:
        import random

        random_state = random.randint(0, 2**32 - 1)
        print(f"  : {random_state}")

    #  Qlib
    print(f"  Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # cache
    cache_mgr = CacheManager(market, start_date, end_date)

    print("\n Step5: ")
    print(f"  : {market}")
    print(f"  : {start_date} ~ {end_date}")
    print(f"  : {len(DEFAULT_FEATURES)} ")
    print(f"  : {DEFAULT_TARGET}")
    print(f"  : {random_state}")

    #
    instruments = D.instruments(market=market)

    #
    print(" ...")
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
    print(f"   : {features_df.shape}")

    # cache
    print(" ...")
    ret_df = cache_mgr.read_dataframe("returns")
    ret_df = ret_df[["ret_1d"]]
    ret_df.columns = [DEFAULT_TARGET]
    print(f"   : {ret_df.shape}")

    # GP
    print("   GP ...")
    miner = FactorMiner(
        features=DEFAULT_FEATURES,
        target=DEFAULT_TARGET,
        gp_config=GPConfig(),
        random_state=random_state,
    )

    expressions = miner.run(features_df, ret_df)

    #
    print(" ...")
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

    print(f"   : {output_path}")

    #
    print(f"\n  {len(expressions)} :")
    for i, expr in enumerate(expressions, 1):
        print(f"\n  Factor {i}:")
        print(f"    {expr}")

    print("\n Step5!")
