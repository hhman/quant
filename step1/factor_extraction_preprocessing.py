#!/usr/bin/env python3
"""
Step1:
qlib
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
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
    periods = {"1d": 1, "1w": 5, "1m": 20}
    max_lag = max(periods.values())

    returns_market = "all"

    buffer_days = max_lag + 10
    end_date_extended = (
        pd.Timestamp(end_date) + pd.Timedelta(days=buffer_days)
    ).strftime("%Y-%m-%d")

    cache_mgr = CacheManager(market, start_date, end_date)

    print(f"初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\nStep1参数:")
    print(f"  市场: {market}")
    print(f"  日期: [{start_date}, {end_date}]")
    print(f"  扩展日期: [{start_date}, {end_date_extended}] (buffer {buffer_days}天)")
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

    print("  计算收益率...")
    ret_map = {
        f"ret_{label}": f"Ref($close, -{lag})/$close - 1"
        for label, lag in periods.items()
    }

    ret_instruments = D.instruments(market=returns_market)

    ret_df = D.features(
        instruments=ret_instruments,
        fields=ret_map.values(),
        start_time=start_date,
        end_time=end_date_extended,
        freq="day",
    )
    ret_df.columns = ret_map.keys()

    ret_df = ret_df[ret_df.index.get_level_values("datetime") <= pd.Timestamp(end_date)]

    cache_mgr.write_dataframe(ret_df, "returns")
    print(
        f"    returns: {ret_df.shape}, 股票数: {len(ret_df.index.get_level_values('instrument').unique())}"
    )

    print("  计算风格因子...")
    all_instruments = D.instruments(market="all")

    total_mv = D.features(
        instruments=all_instruments,
        fields=["$total_mv"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    total_mv["$total_mv"] = (
        total_mv.groupby(level="instrument")["$total_mv"].ffill().bfill()
    )

    industry = D.features(
        instruments=all_instruments,
        fields=["$industry"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    industry["$industry"] = (
        industry.groupby(level="instrument")["$industry"].ffill().bfill()
    )

    float_mv = D.features(
        instruments=all_instruments,
        fields=["$float_mv"],
        start_time=start_date,
        end_date=end_date,
        freq="day",
    )
    float_mv["$float_mv"] = (
        float_mv.groupby(level="instrument")["$float_mv"].ffill().bfill()
    )

    styles_df = pd.concat([total_mv, industry, float_mv], axis=1)
    cache_mgr.write_dataframe(styles_df, "styles")
    print(f"    styles: {styles_df.shape}")

    print("\n数据质量检查:")
    factor_missing_rate = factor_df.isna().mean().mean()
    ret_missing_rate = ret_df.isna().mean().mean()
    total_mv_missing_rate = total_mv["$total_mv"].isna().mean()
    industry_missing_rate = industry["$industry"].isna().mean()
    float_mv_missing_rate = float_mv["$float_mv"].isna().mean()

    print(f"  因子缺失率: {factor_missing_rate:.2%}")
    print(f"  收益率缺失率: {ret_missing_rate:.2%}")
    print(f"  总市值缺失率: {total_mv_missing_rate:.2%}")
    print(f"  行业缺失率: {industry_missing_rate:.2%}")
    print(f"  流通市值缺失率: {float_mv_missing_rate:.2%}")

    MISSING_RATE_THRESHOLD = 0.8
    if total_mv_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(
            f"总市值缺失率 {total_mv_missing_rate:.2%} 超过阈值 {MISSING_RATE_THRESHOLD:.2%}"
        )
    if industry_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(
            f"行业缺失率 {industry_missing_rate:.2%} 超过阈值 {MISSING_RATE_THRESHOLD:.2%}"
        )
    if float_mv_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(
            f"流通市值缺失率 {float_mv_missing_rate:.2%} 超过阈值 {MISSING_RATE_THRESHOLD:.2%}"
        )

    print("\n因子统计:")
    factor_stats = pd.DataFrame(
        {
            "mean": factor_df.mean(),
            "std": factor_df.std(),
            "min": factor_df.min(),
            "max": factor_df.max(),
            "na_ratio": factor_df.isna().mean(),
        }
    )
    print(factor_stats.head(10))
    if len(factor_stats) > 10:
        print(f"  ... 共{len(factor_stats)}个因子")

    print("\nStep1完成!")
    print(f"  Cache: {cache_mgr.CACHE_DIR}")
    print("  数据文件: factor_raw, factor_std, returns, styles")
