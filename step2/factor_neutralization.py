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
from utils import info, warning, error
from core.factor_analysis import neutralize_industry_marketcap


def neutralize_factors(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """对因子进行行业市值中性化处理。

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

    info("\nStep2: 因子中性化")

    info("  读取缓存...")
    try:
        factor_std = cache_mgr.read_dataframe("factor_std")
        styles_df = cache_mgr.read_dataframe("styles")
    except FileNotFoundError as e:
        error(f"  错误: {e}")
        sys.exit(1)

    info(f"    factor_std: {factor_std.shape}")
    info(f"    styles_df: {styles_df.shape}")

    factor_cols = [col for col in factor_std.columns if col in factor_formulas]

    if not factor_cols:
        error(f"  错误: 未找到因子 {factor_formulas} 在缓存中")
        warning(
            f"  可用因子: {[col for col in factor_std.columns if col not in ['$total_mv', '$industry', '$float_mv']]}"
        )
        sys.exit(1)

    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [col for col in required_style_cols if col not in styles_df.columns]
    if missing_cols:
        error(f"  错误: 缺失列: {missing_cols}")
        sys.exit(1)

    data = factor_std.join(styles_df, how="left")

    info(f"    合并后数据: {data.shape}")
    info(
        f"    因子股票数: {len(factor_std.index.get_level_values('instrument').unique())}"
    )
    info(
        f"    风格股票数: {len(styles_df.index.get_level_values('instrument').unique())}"
    )
    info(f"    合并后股票数: {len(data.index.get_level_values('instrument').unique())}")

    needed_cols = factor_cols + required_style_cols
    data = data[needed_cols]

    info(f"    待中性化因子数: {len(factor_cols)}")
    info(f"    因子列: {factor_cols}")
    info(f"    风格列: {required_style_cols}")

    info("  执行中性化...")
    result_list = []
    for dt in data.index.get_level_values("datetime").unique():
        daily_group = (
            data.xs(dt, level="datetime")
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
        info(f"    neutralized: {result.shape}")

        info("\nStep2完成!")
    else:
        error("  错误: 中性化结果为空")
        sys.exit(1)
