"""计算收益率和风格因子数据。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from utils.cache_manager import CacheManager


def calculate_returns_and_styles(
    start_date: str,
    end_date: str,
    provider_uri: str,
) -> None:
    """计算收益率和风格因子（全市场）。

    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        provider_uri: Qlib数据目录
    """
    periods = {"1d": 1, "1w": 5, "1m": 20}
    max_lag = max(periods.values())
    buffer_days = max_lag + 10
    end_date_extended = (
        pd.Timestamp(end_date) + pd.Timedelta(days=buffer_days)
    ).strftime("%Y-%m-%d")

    cache_mgr = CacheManager("all", start_date, end_date)

    print(f"初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\n计算收益率...")
    ret_map = {
        f"ret_{label}": f"Ref($close, -{lag})/$close - 1"
        for label, lag in periods.items()
    }

    ret_instruments = D.instruments(market="all")

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
        end_time=end_date,
        freq="day",
    )
    float_mv["$float_mv"] = (
        float_mv.groupby(level="instrument")["$float_mv"].ffill().bfill()
    )

    styles_df = pd.concat([total_mv, industry, float_mv], axis=1)
    cache_mgr.write_dataframe(styles_df, "styles")
    print(f"    styles: {styles_df.shape}")

    print("\n完成!")
    print(f"  Cache: {cache_mgr.CACHE_DIR}")
