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

import pandas as pd
import qlib
from qlib.constant import REG_CN

from cli_config import parse_common_args, normalize_args
from cache_manager import CacheManager
from factor_analysis import (
    summarize_ic,
    summarize_group_return,
    summarize_autocorr,
    summarize_turnover,
    save_performance_graphs,
)


def main() -> None:
    # 解析CLI参数
    args = parse_common_args()
    params = normalize_args(args, "step4")

    # 创建cache管理器
    cache_mgr = CacheManager(params)

    if params['dry_run']:
        print("🔍 模拟运行模式")
        print(f"  市场: {params['market']}")
        print(f"  Cache目录: {cache_mgr.cache_dir}")
        return

    # 初始化qlib
    print(f"📊 初始化Qlib: {params['provider_uri']}")
    qlib.init(provider_uri=params['provider_uri'], region=REG_CN)

    print(f"\n📈 Step4: 因子绩效评估")

    # 校验step2的cache（step4依赖step2的输出：中性化后的因子）
    try:
        cache_meta, match_info = cache_mgr.load_and_validate_metadata("step2")
        if params['verbose']:
            cache_mgr.print_match_summary(match_info)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

    # 加载中性化后的因子数据（从step2）
    print("📥 加载数据...")
    factor_df = cache_mgr.load_data_subset(
        cache_mgr.get_data_path("factor_行业市值中性化.parquet"),
        match_info
    )
    print(f"  ✓ 因子数据（中性化后）: {factor_df.shape}")

    # 加载收益率数据（从step1）
    step1_meta, step1_match = cache_mgr.load_and_validate_metadata("step1")
    ret_df = cache_mgr.load_data_subset(
        cache_mgr.get_data_path("data_returns.parquet"),
        step1_match
    )
    print(f"  ✓ 收益率数据: {ret_df.shape}")

    if factor_df.empty:
        print("❌ 错误: 因子数据为空")
        print(f"   请检查step2是否成功生成: factor_行业市值中性化.parquet")
        sys.exit(1)

    if ret_df.empty:
        print("❌ 错误: 收益率数据为空")
        print(f"   请检查step1是否成功生成: data_returns.parquet")
        sys.exit(1)

    # 检查索引一致性
    if factor_df.index.nlevels != 2 or ret_df.index.nlevels != 2:
        print(f"❌ 错误: 数据索引格式不正确")
        print(f"   factor_df索引: {factor_df.index.names}")
        print(f"   ret_df索引: {ret_df.index.names}")
        print(f"   期望索引: (instrument, datetime)")
        sys.exit(1)

    merged_df = factor_df.join(ret_df, how="left")
    factor_list = list(factor_df.columns)
    ret_list = list(ret_df.columns)

    # IC / RankIC分析
    print("⚙️  计算IC/RankIC...")
    ic_df, ric_df, ic_summary, ric_summary = summarize_ic(
        merged_df,
        factor_list=factor_list,
        ret_list=ret_list
    )
    ic_df.to_parquet(
        cache_mgr.get_data_path("factor_ic.parquet"),
        compression="snappy",
        index=True
    )
    ric_df.to_parquet(
        cache_mgr.get_data_path("factor_rank_ic.parquet"),
        compression="snappy",
        index=True
    )
    ic_summary.to_excel(
        cache_mgr.get_data_path("factor_ic_summary.xlsx"),
        index=True
    )
    ric_summary.to_excel(
        cache_mgr.get_data_path("factor_rank_ic_summary.xlsx"),
        index=True
    )
    print("  ✓ IC/RankIC分析完成")
    print("IC summary:\n", ic_summary)
    print("Rank IC summary:\n", ric_summary)

    # 分组收益分析
    print("⚙️  计算分组收益...")
    group_daily_df, group_summary = summarize_group_return(
        merged_df,
        factor_list=factor_list,
        ret_list=ret_list,
        quantile=0.2,
    )
    group_daily_df.to_parquet(
        cache_mgr.get_data_path("factor_group_return.parquet"),
        compression="snappy",
        index=True
    )
    group_summary.to_excel(
        cache_mgr.get_data_path("factor_group_return_summary.xlsx"),
        index=True
    )
    print("  ✓ 分组收益分析完成")
    print("Group return summary:\n", group_summary)

    # 自相关分析
    print("⚙️  计算自相关...")
    ac_df, ac_summary = summarize_autocorr(
        merged_df,
        factor_list=factor_list,
        lag=1,
    )
    ac_df.to_parquet(
        cache_mgr.get_data_path("factor_autocorr.parquet"),
        compression="snappy",
        index=True
    )
    ac_summary.to_excel(
        cache_mgr.get_data_path("factor_autocorr_summary.xlsx"),
        index=True
    )
    print("  ✓ 自相关分析完成")
    print("Autocorr summary:\n", ac_summary)

    # 换手率分析
    print("⚙️  计算换手率...")
    turnover_daily_df, turnover_summary = summarize_turnover(
        merged_df,
        factor_list=factor_list,
        N=5,
        lag=1,
    )
    turnover_daily_df.to_parquet(
        cache_mgr.get_data_path("factor_turnover.parquet"),
        compression="snappy",
        index=True
    )
    turnover_summary.to_excel(
        cache_mgr.get_data_path("factor_turnover_summary.xlsx"),
        index=True
    )
    print("  ✓ 换手率分析完成")
    print("Turnover summary:\n", turnover_summary)

    # 生成性能可视化图表
    print("\n⚙️  生成性能可视化图表...")
    try:
        graphs_dir = cache_mgr.cache_dir / "graphs"
        save_performance_graphs(
            merged_df=merged_df,
            factor_list=factor_list,
            ret_list=ret_list,
            output_dir=graphs_dir,
            graph_names=["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
        )
        print(f"  ✓ 可视化图表已保存到: {graphs_dir}")
    except Exception as e:
        print(f"  ⚠️  生成可视化图表失败: {e}")
        print(f"     跳过图表生成，继续保存元数据...")

    # 保存元数据
    cache_mgr.save_metadata("step4")
    print(f"  ✓ 保存: step4_metadata.json")

    print(f"\n✅ Step4完成!")
    print(f"   所有评估结果已保存到: {cache_mgr.cache_dir}")


if __name__ == "__main__":
    main()
