#!/usr/bin/env python3
"""
Step2: 行业市值中性化
功能：对因子数据进行行业和市值中性化处理
支持智能cache子集匹配
完全脱离qlib依赖
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
    因子中性化的核心逻辑函数

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
    # 创建 cache manager
    cache_mgr = CacheManager(market, start_date, end_date)

    print("\n🎯 Step2: 行业市值中性化")

    # 加载 step1 的 cache
    print("📥 加载数据...")
    try:
        factor_std = cache_mgr.read_dataframe("factor_std")
        styles_df = cache_mgr.read_dataframe("styles")
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

    print(f"  ✓ 标准化因子: {factor_std.shape}")
    print(f"  ✓ 风格数据: {styles_df.shape}")

    # 提取因子列 - 使用显式传入的factor_formulas参数
    factor_cols = [col for col in factor_std.columns if col in factor_formulas]

    # 如果没有匹配的因子，说明参数错误
    if not factor_cols:
        print(f"❌ 错误: 请求的因子 {factor_formulas} 在cache中不存在")
        print(
            f"  cache中的因子列: {[col for col in factor_std.columns if col not in ['$total_mv', '$industry', '$float_mv']]}"
        )
        sys.exit(1)

    # 检查必需的风格列
    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [col for col in required_style_cols if col not in styles_df.columns]
    if missing_cols:
        print(f"❌ 错误: 缺少列: {missing_cols}")
        sys.exit(1)

    # 合并因子和风格数据
    data_for_neutralize = pd.concat(
        [factor_std[factor_cols], styles_df[required_style_cols]], axis=1
    )

    print(f"  ✓ 因子列: {len(factor_cols)}个 {factor_cols}")
    print(f"  ✓ 风格列: {required_style_cols}")

    # 执行中性化
    print("⚙️  执行行业市值中性化...")
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
        print(f"  ✓ 保存: neutralized ({result.shape})")

        # 中性化效果摘要
        print("\n📊 中性化效果摘要:")
        merged = result.join(styles_df[["$total_mv"]])
        for factor_col in result.columns:
            corr = merged[factor_col].corr(np.log(merged["$total_mv"]))
            print(f"  {factor_col} 与log(市值)相关性: {corr:.4f}")

        print("\n📈 中性化因子分布统计:")
        neutralized_stats = pd.DataFrame(
            {
                "均值": result.mean(),
                "标准差": result.std(),
                "最小值": result.min(),
                "最大值": result.max(),
                "缺失率": result.isna().mean(),
            }
        )
        print(neutralized_stats.head(10))

        print("\n✅ Step2完成!")
    else:
        print("❌ 错误: 中性化失败，结果为空")
        sys.exit(1)
