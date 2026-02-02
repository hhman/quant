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
from core.factor_analysis import ext_out_3std, z_score


def calculate_factors(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """计算因子并保存到缓存。

    Parameters:
    -----------
    market : str
        股票池名称 (如 csi300)
    start_date : str
        开始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    factor_formulas : list[str]
        因子表达式列表
    provider_uri : str
        Qlib数据目录

    Returns:
    --------
    None
    """
    cache_mgr = CacheManager(market, start_date, end_date)

    print(f"初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\nStep1参数:")
    print(f"  市场: {market}")
    print(f"  日期: [{start_date}, {end_date}]")
    print(f"  因子数: {len(factor_formulas)}")

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
        print(f"  删除全NaN列: {dropped_cols}")
        factor_df = factor_df.drop(columns=dropped_cols)

    valid_factor_cols = factor_df.columns.to_list()

    print("  保存缓存...")

    factor_raw = factor_df.copy()
    cache_mgr.write_dataframe(factor_raw, "factor_raw")
    print(f"    factor_raw: {factor_raw.shape}")

    print("  去极值...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: ext_out_3std(x, valid_factor_cols)
    )

    print("  标准化...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: z_score(x, valid_factor_cols)
    )

    cache_mgr.write_dataframe(factor_df, "factor_std")
    print(f"    factor_std: {factor_df.shape}")

    print("\nStep1完成!")
    print(f"  Cache: {cache_mgr.CACHE_DIR}")
    print("  数据文件: factor_raw, factor_std")
