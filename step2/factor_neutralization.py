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

from utils.cache_manager import CacheManager
from core.factor_analysis import neutralize_industry_marketcap


def neutralize_factors(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """对因子进行行业市值中性化处理。

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

    print("\nStep2: 因子中性化")

    print("  读取缓存...")
    try:
        factor_std = cache_mgr.read_dataframe("factor_std")
        styles_df = cache_mgr.read_dataframe("styles")
    except FileNotFoundError as e:
        print(f"  错误: {e}")
        sys.exit(1)

    print(f"    factor_std: {factor_std.shape}")
    print(f"    styles_df: {styles_df.shape}")

    factor_cols = [col for col in factor_std.columns if col in factor_formulas]

    if not factor_cols:
        print(f"  错误: 未找到因子 {factor_formulas} 在缓存中")
        print(
            f"  可用因子: {[col for col in factor_std.columns if col not in ['$total_mv', '$industry', '$float_mv']]}"
        )
        sys.exit(1)

    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [col for col in required_style_cols if col not in styles_df.columns]
    if missing_cols:
        print(f"  错误: 缺失列: {missing_cols}")
        sys.exit(1)

    data_for_neutralize = pd.concat(
        [factor_std[factor_cols], styles_df[required_style_cols]], axis=1
    )

    print(f"    待中性化因子数: {len(factor_cols)}")
    print(f"    因子列: {factor_cols}")
    print(f"    风格列: {required_style_cols}")

    print("  执行中性化...")
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
        print(f"    neutralized: {result.shape}")

        print("\nStep2完成!")
    else:
        print("  错误: 中性化结果为空")
        sys.exit(1)
