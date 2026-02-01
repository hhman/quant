#!/usr/bin/env python3
"""
Step1:
qlib
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from utils.cache_manager import CacheManager
from core.factor_analysis import ext_out_3std, z_score


def calculate_factors(
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
         ()
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
    #  1d, 1w, 1m
    periods = {"1d": 1, "1w": 5, "1m": 20}
    # ,
    max_lag = max(periods.values())  # 20

    # all,
    returns_market = "all"

    # ,
    #  =  + 10
    buffer_days = max_lag + 10
    end_date_extended = (
        pd.Timestamp(end_date) + pd.Timedelta(days=buffer_days)
    ).strftime("%Y-%m-%d")

    #  cache manager (end_date)
    cache_mgr = CacheManager(market, start_date, end_date)

    # qlib
    print(f" Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\n Step1: ")
    print(f"  : {market}")
    print(f"  : [{start_date}, {end_date}]")
    print(f"  : [{start_date}, {end_date_extended}] ({buffer_days})")
    print(f"  : {len(factor_formulas)}")
    for i, formula in enumerate(factor_formulas[:5], 1):
        print(f"    {i}. {formula}")
    if len(factor_formulas) > 5:
        print(f"    ... ({len(factor_formulas)})")
    print(f"  : {list(periods.keys())}")

    instruments = D.instruments(market=market)

    # CLI
    factor_df = D.features(
        instruments=instruments,
        fields=factor_formulas,
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    #
    factor_df.columns = factor_formulas

    #
    dropped_cols = [col for col in factor_df.columns if factor_df[col].isna().all()]
    if dropped_cols:
        print(f"    : {dropped_cols}")
        factor_df = factor_df.drop(columns=dropped_cols)

    valid_factor_cols = factor_df.columns.to_list()

    # ========== 4cache ==========
    print("  cache...")

    # 1.
    factor_raw = factor_df.copy()
    cache_mgr.write_dataframe(factor_raw, "factor_raw")
    print(f"   : factor_raw ({factor_raw.shape})")

    #
    print("  ...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: ext_out_3std(x, valid_factor_cols)
    )

    #
    print("  ...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: z_score(x, valid_factor_cols)
    )

    # 2.
    cache_mgr.write_dataframe(factor_df, "factor_std")
    print(f"   : factor_std ({factor_df.shape})")

    #
    print("  ...")
    print(f"    : {returns_market}")
    print(f"    : {start_date} ~ {end_date_extended}")
    print(f"    : {buffer_days} ({max_lag})")

    ret_map = {
        f"ret_{label}": f"Ref($close, -{lag})/$close - 1"
        for label, lag in periods.items()
    }

    # all,
    ret_instruments = D.instruments(market=returns_market)
    print(f"     {returns_market}  ()")

    ret_df = D.features(
        instruments=ret_instruments,
        fields=ret_map.values(),
        start_time=start_date,
        end_time=end_date_extended,  #
        freq="day",
    )
    ret_df.columns = ret_map.keys()

    print(
        f"    : {ret_df.index.get_level_values('datetime').min()} ~ {ret_df.index.get_level_values('datetime').max()}"
    )

    #
    ret_df = ret_df[ret_df.index.get_level_values("datetime") <= pd.Timestamp(end_date)]
    print(
        f"    : {ret_df.index.get_level_values('datetime').min()} ~ {ret_df.index.get_level_values('datetime').max()}"
    )

    # 3.
    cache_mgr.write_dataframe(ret_df, "returns")
    print(f"   : returns ({ret_df.shape})")
    print(f"    - : {len(ret_df.index.get_level_values('instrument').unique())}")

    # ==========  ==========
    print("  ...")
    print("     all")

    #  all
    all_instruments = D.instruments(market="all")

    #
    total_mv = D.features(
        instruments=all_instruments,
        fields=["$total_mv"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    total_mv["$total_mv"] = (
        total_mv.groupby(level="instrument")["$total_mv"].ffill().bfill()
    )

    #
    industry = D.features(
        instruments=all_instruments,
        fields=["$industry"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    industry["$industry"] = (
        industry.groupby(level="instrument")["$industry"].ffill().bfill()
    )

    #
    float_mv = D.features(
        instruments=all_instruments,
        fields=["$float_mv"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    float_mv["$float_mv"] = (
        float_mv.groupby(level="instrument")["$float_mv"].ffill().bfill()
    )

    # 4.
    styles_df = pd.concat([total_mv, industry, float_mv], axis=1)
    cache_mgr.write_dataframe(styles_df, "styles")
    print(f"   : styles ({styles_df.shape})")
    print(f"    - : {len(styles_df.columns)} ($total_mv, $industry, $float_mv)")

    #
    print("\n :")
    factor_missing_rate = factor_df.isna().mean().mean()
    ret_missing_rate = ret_df.isna().mean().mean()
    total_mv_missing_rate = total_mv["$total_mv"].isna().mean()
    industry_missing_rate = industry["$industry"].isna().mean()
    float_mv_missing_rate = float_mv["$float_mv"].isna().mean()

    print(f"  : {factor_missing_rate:.2%}")
    print(f"  : {ret_missing_rate:.2%}")
    print(f"  : {total_mv_missing_rate:.2%}")
    print(f"  : {industry_missing_rate:.2%}")
    print(f"  : {float_mv_missing_rate:.2%}")

    #
    MISSING_RATE_THRESHOLD = 0.8  # 80%
    if total_mv_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(f" {total_mv_missing_rate:.2%}  {MISSING_RATE_THRESHOLD:.2%}")
    if industry_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(f" {industry_missing_rate:.2%}  {MISSING_RATE_THRESHOLD:.2%}")
    if float_mv_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(f" {float_mv_missing_rate:.2%}  {MISSING_RATE_THRESHOLD:.2%}")

    #
    print("\n :")
    factor_stats = pd.DataFrame(
        {
            "mean": factor_df.mean(),
            "std": factor_df.std(),
            "min": factor_df.min(),
            "max": factor_df.max(),
            "na_ratio": factor_df.isna().mean(),
        }
    )
    print(factor_stats.head(10))  # 10
    if len(factor_stats) > 10:
        print(f"  ... ({len(factor_stats)})")

    print("\n Step1!")
    print(f"   Cache: {cache_mgr.CACHE_DIR}")
    print("   : factor_raw, factor_std, returns, styles")
