#!/usr/bin/env python3
"""
Step4:
IC
cache
qlibcalc_ic, calc_long_short_return, pred_autocorr
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import qlib
from qlib.constant import REG_CN

from utils.cache_manager import CacheManager
from core.factor_analysis import (
    summarize_ic,
    summarize_group_return,
    summarize_autocorr,
    summarize_turnover,
    save_performance_graphs,
)


def evaluate_performance(
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
    # cache
    cache_mgr = CacheManager(market, start_date, end_date)

    # qlib
    print(f" Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\n Step4: ")

    #
    print(" ...")
    factor_df = cache_mgr.read_dataframe("neutralized")
    print(f"   : {factor_df.shape}")

    ret_df = cache_mgr.read_dataframe("returns")
    print(f"   : {ret_df.shape}")

    if factor_df.empty:
        print(" : ")
        print("   step2")
        sys.exit(1)

    if ret_df.empty:
        print(" : ")
        print("   step1")
        sys.exit(1)

    #
    if factor_df.index.nlevels != 2 or ret_df.index.nlevels != 2:
        print(" : ")
        print(f"   factor_df: {factor_df.index.names}")
        print(f"   ret_df: {ret_df.index.names}")
        print("   : (instrument, datetime)")
        sys.exit(1)

    merged_df = factor_df.join(ret_df, how="left")
    factor_list = list(factor_df.columns)
    ret_list = list(ret_df.columns)

    #
    start_compact = start_date.replace("-", "")
    end_compact = end_date.replace("-", "")

    # IC / RankIC
    print("  IC/RankIC...")
    ic_df, ric_df, ic_summary, ric_summary = summarize_ic(
        merged_df, factor_list=factor_list, ret_list=ret_list
    )
    cache_mgr.write_dataframe(ic_df, "ic")
    cache_mgr.write_dataframe(ric_df, "rank_ic")
    ic_summary.to_excel(
        f".cache/{market}_{start_compact}_{end_compact}__ic_summary.xlsx", index=True
    )
    ric_summary.to_excel(
        f".cache/{market}_{start_compact}_{end_compact}__rank_ic_summary.xlsx",
        index=True,
    )
    print(f"   : ic ({ic_df.shape}), rank_ic ({ric_df.shape})")

    #
    print("  ...")
    group_daily_df, group_summary = summarize_group_return(
        merged_df,
        factor_list=factor_list,
        ret_list=ret_list,
        quantile=0.2,
    )
    cache_mgr.write_dataframe(group_daily_df, "group_return")
    group_summary.to_excel(
        f".cache/{market}_{start_compact}_{end_compact}__group_return_summary.xlsx",
        index=True,
    )
    print(f"   : group_return ({group_daily_df.shape})")

    #
    print("  ...")
    ac_df, ac_summary = summarize_autocorr(
        merged_df,
        factor_list=factor_list,
        lag=1,
    )
    cache_mgr.write_dataframe(ac_df, "autocorr")
    ac_summary.to_excel(
        f".cache/{market}_{start_compact}_{end_compact}__autocorr_summary.xlsx",
        index=True,
    )
    print(f"   : autocorr ({ac_df.shape})")

    #
    print("  ...")
    turnover_daily_df, turnover_summary = summarize_turnover(
        merged_df,
        factor_list=factor_list,
        N=5,
        lag=1,
    )
    cache_mgr.write_dataframe(turnover_daily_df, "turnover")
    turnover_summary.to_excel(
        f".cache/{market}_{start_compact}_{end_compact}__turnover_summary.xlsx",
        index=True,
    )
    print(f"   : turnover ({turnover_daily_df.shape})")

    #
    print("\n  ...")
    try:
        graphs_dir = Path(".cache") / "graphs"
        save_performance_graphs(
            merged_df=merged_df,
            factor_list=factor_list,
            ret_list=ret_list,
            output_dir=graphs_dir,
            graph_names=["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
        )
        print(f"   : {graphs_dir}")
    except Exception as e:
        print(f"    : {e}")
        print("     ...")

    print("\n Step4!")
    print("   : .cache/")
