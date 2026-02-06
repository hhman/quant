#!/usr/bin/env python3
"""
Step3:

cache
qlib
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils.cache_manager import CacheManager
from core.factor_analysis import factor_return_industry_marketcap


def calculate_returns(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """计算因子收益率回归系数。

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

    print("\nStep3: 收益率计算")

    print("  读取缓存...")
    factor_std = cache_mgr.read_dataframe("factor_std")
    styles_df = cache_mgr.read_dataframe("styles")
    ret_df = cache_mgr.read_dataframe("returns")

    data = factor_std.join(styles_df, how="left")
    data = data.join(ret_df, how="left")

    print(f"    合并后数据: {data.shape}")
    print(
        f"    因子股票数: {len(factor_std.index.get_level_values('instrument').unique())}"
    )
    print(
        f"    收益率股票数: {len(ret_df.index.get_level_values('instrument').unique())}"
    )
    print(
        f"    合并后股票数: {len(data.index.get_level_values('instrument').unique())}"
    )

    factor_cols = [col for col in factor_std.columns if col in factor_formulas]
    ret_cols = [col for col in ret_df.columns if col.startswith("ret_")]

    if not factor_cols:
        print(f"  错误: 未找到因子 {factor_formulas} 在缓存中")
        print(
            f"  可用因子: {[col for col in factor_std.columns if col not in ['$total_mv', '$industry', '$float_mv']]}"
        )
        sys.exit(1)

    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [
        col for col in ret_cols + required_style_cols if col not in data.columns
    ]
    if missing_cols:
        print(f"  错误: 缺失列: {missing_cols}")
        sys.exit(1)

    needed_cols = factor_cols + ret_cols + required_style_cols
    data = data[needed_cols]

    print(f"    因子数: {len(factor_cols)}")
    print(f"    收益率列数: {len(ret_cols)}")
    print(f"    风格列: {required_style_cols}")

    print("  执行回归...")
    coef_list = []
    t_list = []
    for dt in data.index.get_level_values("datetime").unique():
        daily_group = (
            data.xs(dt, level="datetime")
            .assign(datetime=dt)
            .set_index("datetime", append=True)
            .reorder_levels(["instrument", "datetime"])
        )
        result = factor_return_industry_marketcap(
            daily_group,
            factor_list=factor_cols,
            ret_list=ret_cols,
            total_mv_col="$total_mv",
            industry_col="$industry",
            float_mv_col="$float_mv",
        )
        coef_df, t_df = result
        coef_list.append(coef_df)
        t_list.append(t_df)

    if coef_list and t_list:
        coef_all = pd.concat(coef_list, axis=0)
        t_all = pd.concat(t_list, axis=0)

        cache_mgr.write_dataframe(coef_all, "return_coef")
        print(f"    return_coef: {coef_all.shape}")

        cache_mgr.write_dataframe(t_all, "return_tval")
        print(f"    return_tval: {t_all.shape}")

        def _coef_summary(series: pd.Series) -> pd.Series:
            """计算回归系数的统计摘要。

            Args:
                series: 回归系数序列

            Returns:
                包含均值和t统计量的Series
            """
            s = series.dropna()
            if s.empty:
                return pd.Series(dtype=float)
            mean = s.mean()
            std = s.std()
            t_test = mean / std * np.sqrt(len(s)) if std != 0 else np.nan
            return pd.Series(
                {
                    "mean_coef": mean,
                    "t_stat": t_test,
                }
            )

        def _t_summary(series: pd.Series) -> pd.Series:
            """计算t统计量的摘要信息。

            Args:
                series: t统计量序列

            Returns:
                包含t统计量各项指标的Series
            """
            s = series.dropna()
            if s.empty:
                return pd.Series(dtype=float)
            t_mean = s.mean()
            t_std = s.std()
            abs_mean = s.abs().mean()
            gt2_rate = (s.abs() > 2).sum() / len(s)
            t_mean_over_std = t_mean / t_std if t_std != 0 else np.nan
            return pd.Series(
                {
                    "mean_abs_t": abs_mean,
                    "pct_abs_t_gt_2": gt2_rate,
                    "mean_t": t_mean,
                    "mean_t_over_std": t_mean_over_std,
                }
            )

        coef_summary = pd.DataFrame(
            {col: _coef_summary(coef_all[col]) for col in coef_all.columns}
        ).T
        t_summary = pd.DataFrame(
            {col: _t_summary(t_all[col]) for col in t_all.columns}
        ).T

        cache_mgr.write_summary(coef_summary, "return_coef")
        cache_mgr.write_summary(t_summary, "return_tval")
        print("    Excel文件已保存")

        print("\nStep3完成!")
    else:
        print("  错误: 回归结果为空")
        sys.exit(1)
