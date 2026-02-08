#!/usr/bin/env python3
"""
Step1:
qlib
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import qlib
from qlib.constant import REG_CN
from qlib.data import D

from utils.cache_manager import CacheManager
from utils import info, warning
from core.factor_analysis import ext_out_3std, z_score


def calculate_factors(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """计算因子并保存到缓存。

    Args:
        market: 股票池名称 (如 csi300)
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        factor_formulas: 因子表达式列表
        provider_uri: Qlib数据目录

    Returns:
        None
    """
    cache_mgr = CacheManager(market, start_date, end_date)

    info(f"初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    info("\nStep1参数:")
    info(f"  市场: {market}")
    info(f"  日期: [{start_date}, {end_date}]")
    info(f"  因子数: {len(factor_formulas)}")

    instruments = D.instruments(market=market)

    factor_df = D.features(
        instruments=instruments,
        fields=factor_formulas,
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    factor_df.columns = factor_formulas

    dropped_cols = [col for col in factor_df.columns if factor_df[col].isna().all()]
    if dropped_cols:
        warning(f"  删除全NaN列: {dropped_cols}")
        factor_df = factor_df.drop(columns=dropped_cols)

    valid_factor_cols = factor_df.columns.to_list()

    info("  保存缓存...")

    factor_raw = factor_df.copy()
    cache_mgr.write_dataframe(factor_raw, "factor_raw")
    info(f"    factor_raw: {factor_raw.shape}")

    info("  去极值...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: ext_out_3std(x, valid_factor_cols)
    )

    info("  标准化...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: z_score(x, valid_factor_cols)
    )

    cache_mgr.write_dataframe(factor_df, "factor_std")
    info(f"    factor_std: {factor_df.shape}")

    info("\nStep1完成!")
    info(f"  Cache: {cache_mgr.CACHE_DIR}")
    info("  数据文件: factor_raw, factor_std")
