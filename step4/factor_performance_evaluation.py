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
    """评估因子绩效指标（IC、分组收益、自相关、换手率等）。

    Parameters:
    -----------
    market : str
        股票池名称
    start_date : str
        开始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    factor_formulas : list[str]
        因子公式列表
    provider_uri : str
        Qlib数据目录

    Returns:
    --------
    None
    """
    cache_mgr = CacheManager(market, start_date, end_date)

    print(f"初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\nStep4: 因子绩效评估")

    print("  读取缓存...")
    factor_df = cache_mgr.read_dataframe("neutralized")
    print(f"    neutralized: {factor_df.shape}")

    ret_df = cache_mgr.read_dataframe("returns")
    print(f"    returns: {ret_df.shape}")

    if factor_df.empty:
        print("  错误: 中性化因子为空")
        print("    请先运行 step2")
        sys.exit(1)

    if ret_df.empty:
        print("  错误: 收益率为空")
        print("    请先运行 step1")
        sys.exit(1)

    if factor_df.index.nlevels != 2 or ret_df.index.nlevels != 2:
        print("  错误: 索引格式错误")
        print(f"    factor_df: {factor_df.index.names}")
        print(f"    ret_df: {ret_df.index.names}")
        print("    期望格式: (instrument, datetime)")
        sys.exit(1)

    merged_df = factor_df.join(ret_df, how="left")
    factor_list = list(factor_df.columns)
    ret_list = list(ret_df.columns)

    print("  计算IC/RankIC...")
    ic_df, ric_df, ic_summary, ric_summary = summarize_ic(
        merged_df, factor_list=factor_list, ret_list=ret_list
    )
    cache_mgr.write_dataframe(ic_df, "ic")
    cache_mgr.write_dataframe(ric_df, "rank_ic")
    cache_mgr.write_summary(ic_summary, "ic")
    cache_mgr.write_summary(ric_summary, "rank_ic")
    print(f"    ic: {ic_df.shape}, rank_ic: {ric_df.shape}")

    print("  计算分组收益...")
    group_daily_df, group_summary = summarize_group_return(
        merged_df,
        factor_list=factor_list,
        ret_list=ret_list,
        quantile=0.2,
    )
    cache_mgr.write_dataframe(group_daily_df, "group_return")
    cache_mgr.write_summary(group_summary, "group_return")
    print(f"    group_return: {group_daily_df.shape}")

    print("  计算自相关...")
    ac_df, ac_summary = summarize_autocorr(
        merged_df,
        factor_list=factor_list,
        lag=1,
    )
    cache_mgr.write_dataframe(ac_df, "autocorr")
    cache_mgr.write_summary(ac_summary, "autocorr")
    print(f"    autocorr: {ac_df.shape}")

    print("  计算换手率...")
    turnover_daily_df, turnover_summary = summarize_turnover(
        merged_df,
        factor_list=factor_list,
        N=5,
        lag=1,
    )
    cache_mgr.write_dataframe(turnover_daily_df, "turnover")
    cache_mgr.write_summary(turnover_summary, "turnover")
    print(f"    turnover: {turnover_daily_df.shape}")

    print("\n  生成图表...")
    try:
        graphs_dir = Path(".cache") / "graphs"
        save_performance_graphs(
            merged_df=merged_df,
            factor_list=factor_list,
            ret_list=ret_list,
            output_dir=graphs_dir,
            graph_names=["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
        )
        print(f"    图表目录: {graphs_dir}")
    except Exception as e:
        print(f"    警告: 图表生成失败: {e}")
        print("    继续执行...")

    print("\nStep4完成!")
    print("  输出目录: .cache/")
