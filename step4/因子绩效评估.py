#!/usr/bin/env python3
"""
Step4: 因子绩效评估
功能：全面评估因子有效性（IC、分组收益、自相关、换手率等）
支持智能cache子集匹配
依赖qlib的绩效评估函数（calc_ic, calc_long_short_return, pred_autocorr）
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
    评估因子绩效的核心逻辑函数

    Parameters:
    -----------
    market : str
        市场标识
    start_date : str
        起始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    factor_formulas : list[str]
        因子表达式列表
    provider_uri : str
        Qlib数据路径

    Returns:
    --------
    None
    """
    # 创建cache管理器
    cache_mgr = CacheManager(market, start_date, end_date)

    # 初始化qlib
    print(f"📊 初始化Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\n📈 Step4: 因子绩效评估")

    # 加载数据
    print("📥 加载数据...")
    factor_df = cache_mgr.read_dataframe("neutralized")
    print(f"  ✓ 因子数据（中性化后）: {factor_df.shape}")

    ret_df = cache_mgr.read_dataframe("returns")
    print(f"  ✓ 收益率数据: {ret_df.shape}")

    if factor_df.empty:
        print("❌ 错误: 因子数据为空")
        print("   请检查step2是否成功生成")
        sys.exit(1)

    if ret_df.empty:
        print("❌ 错误: 收益率数据为空")
        print("   请检查step1是否成功生成")
        sys.exit(1)

    # 检查索引一致性
    if factor_df.index.nlevels != 2 or ret_df.index.nlevels != 2:
        print("❌ 错误: 数据索引格式不正确")
        print(f"   factor_df索引: {factor_df.index.names}")
        print(f"   ret_df索引: {ret_df.index.names}")
        print("   期望索引: (instrument, datetime)")
        sys.exit(1)

    merged_df = factor_df.join(ret_df, how="left")
    factor_list = list(factor_df.columns)
    ret_list = list(ret_df.columns)

    # 使用紧凑日期格式保存汇总文件
    start_compact = start_date.replace("-", "")
    end_compact = end_date.replace("-", "")

    # IC / RankIC分析
    print("⚙️  计算IC/RankIC...")
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
    print(f"  ✓ 保存: ic ({ic_df.shape}), rank_ic ({ric_df.shape})")

    # 分组收益分析
    print("⚙️  计算分组收益...")
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
    print(f"  ✓ 保存: group_return ({group_daily_df.shape})")

    # 自相关分析
    print("⚙️  计算自相关...")
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
    print(f"  ✓ 保存: autocorr ({ac_df.shape})")

    # 换手率分析
    print("⚙️  计算换手率...")
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
    print(f"  ✓ 保存: turnover ({turnover_daily_df.shape})")

    # 生成性能可视化图表
    print("\n⚙️  生成性能可视化图表...")
    try:
        graphs_dir = Path(".cache") / "graphs"
        save_performance_graphs(
            merged_df=merged_df,
            factor_list=factor_list,
            ret_list=ret_list,
            output_dir=graphs_dir,
            graph_names=["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
        )
        print(f"  ✓ 可视化图表已保存到: {graphs_dir}")
    except Exception as e:
        print(f"  ⚠️  生成可视化图表失败: {e}")
        print("     跳过图表生成，继续...")

    print("\n✅ Step4完成!")
    print("   所有评估结果已保存到: .cache/")
