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
from utils import info, warning, error
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

    Args:
        market: 股票池名称
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        factor_formulas: 因子公式列表
        provider_uri: Qlib数据目录

    Returns:
        None
    """
    cache_mgr = CacheManager(market, start_date, end_date)

    info(f"初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    info("\nStep4: 因子绩效评估")

    info("  读取缓存...")
    factor_df = cache_mgr.read_dataframe("neutralized")
    info(f"    neutralized: {factor_df.shape}")

    ret_df = cache_mgr.read_dataframe("returns")
    info(f"    returns: {ret_df.shape}")

    if factor_df.empty:
        error("  错误: 中性化因子为空")
        warning("    请先运行 step2")
        sys.exit(1)

    if ret_df.empty:
        error("  错误: 收益率为空")
        warning("    请先运行 step1")
        sys.exit(1)

    if factor_df.index.nlevels != 2 or ret_df.index.nlevels != 2:
        error("  错误: 索引格式错误")
        warning(f"    factor_df: {factor_df.index.names}")
        warning(f"    ret_df: {ret_df.index.names}")
        warning("    期望格式: (instrument, datetime)")
        sys.exit(1)

    merged_df = factor_df.join(ret_df, how="left")
    factor_list = list(factor_df.columns)
    ret_list = list(ret_df.columns)

    info("  计算IC/RankIC...")
    ic_df, ric_df, ic_summary, ric_summary = summarize_ic(
        merged_df, factor_list=factor_list, ret_list=ret_list
    )
    cache_mgr.write_dataframe(ic_df, "ic")
    cache_mgr.write_dataframe(ric_df, "rank_ic")
    cache_mgr.write_summary(ic_summary, "ic")
    cache_mgr.write_summary(ric_summary, "rank_ic")
    info(f"    ic: {ic_df.shape}, rank_ic: {ric_df.shape}")

    info("  计算分组收益...")
    group_daily_df, group_summary = summarize_group_return(
        merged_df,
        factor_list=factor_list,
        ret_list=ret_list,
        quantile=0.2,
    )
    cache_mgr.write_dataframe(group_daily_df, "group_return")
    cache_mgr.write_summary(group_summary, "group_return")
    info(f"    group_return: {group_daily_df.shape}")

    info("  计算自相关...")
    ac_df, ac_summary = summarize_autocorr(
        merged_df,
        factor_list=factor_list,
        lag=1,
    )
    cache_mgr.write_dataframe(ac_df, "autocorr")
    cache_mgr.write_summary(ac_summary, "autocorr")
    info(f"    autocorr: {ac_df.shape}")

    info("  计算换手率...")
    turnover_daily_df, turnover_summary = summarize_turnover(
        merged_df,
        factor_list=factor_list,
        N=5,
        lag=1,
    )
    cache_mgr.write_dataframe(turnover_daily_df, "turnover")
    cache_mgr.write_summary(turnover_summary, "turnover")
    info(f"    turnover: {turnover_daily_df.shape}")

    info("\n  生成图表...")
    try:
        start_compact = start_date.replace("-", "")
        end_compact = end_date.replace("-", "")
        graphs_dir = (
            Path(".cache") / "graphs" / f"{market}_{start_compact}_{end_compact}"
        )
        save_performance_graphs(
            merged_df=merged_df,
            factor_list=factor_list,
            ret_list=ret_list,
            output_dir=graphs_dir,
            graph_names=["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
        )
        info(f"    图表目录: {graphs_dir}")
    except Exception as e:
        warning(f"    警告: 图表生成失败: {e}")
        info("    继续执行...")

    info("\nStep4完成!")
    info("  输出目录: .cache/")
