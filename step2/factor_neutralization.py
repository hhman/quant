#!/usr/bin/env python3
"""
Step2:

cache
qlib
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils.cache_manager import CacheManager
from core.factor_analysis import neutralize_industry_marketcap


def neutralize_factors(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """


    Parameters:
    -----------
    market : str

    start_date : str
         (YYYY-MM-DD)
    end_date : str
         (YYYY-MM-DD)
    factor_formulas : list[str]

    provider_uri : str
        Qlib

    Returns:
    --------
    None
    """
    #  cache manager
    cache_mgr = CacheManager(market, start_date, end_date)

    print("\n Step2: ")

    #  step1  cache
    print(" ...")
    try:
        factor_std = cache_mgr.read_dataframe("factor_std")
        styles_df = cache_mgr.read_dataframe("styles")
    except FileNotFoundError as e:
        print(f" : {e}")
        sys.exit(1)

    print(f"   : {factor_std.shape}")
    print(f"   : {styles_df.shape}")

    #  - factor_formulas
    factor_cols = [col for col in factor_std.columns if col in factor_formulas]

    #
    if not factor_cols:
        print(f" :  {factor_formulas} cache")
        print(
            f"  cache: {[col for col in factor_std.columns if col not in ['$total_mv', '$industry', '$float_mv']]}"
        )
        sys.exit(1)

    #
    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [col for col in required_style_cols if col not in styles_df.columns]
    if missing_cols:
        print(f" : : {missing_cols}")
        sys.exit(1)

    #
    data_for_neutralize = pd.concat(
        [factor_std[factor_cols], styles_df[required_style_cols]], axis=1
    )

    print(f"   : {len(factor_cols)} {factor_cols}")
    print(f"   : {required_style_cols}")

    #
    print("  ...")
    result_list = []
    for dt in data_for_neutralize.index.get_level_values("datetime").unique():
        daily_group = (
            data_for_neutralize.xs(dt, level="datetime")
            .assign(datetime=dt)
            .set_index("datetime", append=True)
            .reorder_levels(["instrument", "datetime"])
        )
        daily = neutralize_industry_marketcap(
            daily_group,
            factor_list=factor_cols,
            total_mv_col="$total_mv",
            industry_col="$industry",
            float_mv_col="$float_mv",
        )
        result_list.append(daily)

    if result_list:
        result = pd.concat(result_list, axis=0).sort_index(
            level=["instrument", "datetime"]
        )
        cache_mgr.write_dataframe(result, "neutralized")
        print(f"   : neutralized ({result.shape})")

        #
        print("\n :")
        merged = result.join(styles_df[["$total_mv"]])
        for factor_col in result.columns:
            corr = merged[factor_col].corr(np.log(merged["$total_mv"]))
            print(f"  {factor_col} log(): {corr:.4f}")

        print("\n :")
        neutralized_stats = pd.DataFrame(
            {
                "mean": result.mean(),
                "std": result.std(),
                "min": result.min(),
                "max": result.max(),
                "na_ratio": result.isna().mean(),
            }
        )
        print(neutralized_stats.head(10))

        print("\n Step2!")
    else:
        print(" : ")
        sys.exit(1)
