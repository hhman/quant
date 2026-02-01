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
    """


    Parameters:
    -----------
    market : str

    start_date : str
         (YYYY-MM-DD)
    end_date : str
         (YYYY-MM-DD)
    factor_formulas : list[str]

    provider_uri : str
        Qlib

    Returns:
    --------
    None
    """
    # cache
    cache_mgr = CacheManager(market, start_date, end_date)

    print("\n Step3: ")

    #
    print(" ...")
    factor_std = cache_mgr.read_dataframe("factor_std")
    styles_df = cache_mgr.read_dataframe("styles")
    ret_df = cache_mgr.read_dataframe("returns")

    #
    # : joinconcat,(all)
    # factor_stdstyles_dfcsi300, ret_dfall
    data = factor_std.join(styles_df, how="left")
    data = data.join(ret_df, how="left")  # left joinall

    print(f"   : {data.shape}")
    print(f"    - : {len(factor_std.index.get_level_values('instrument').unique())}")
    print(f"    - : {len(ret_df.index.get_level_values('instrument').unique())}")
    print(f"    - : {len(data.index.get_level_values('instrument').unique())}")

    #
    factor_cols = [col for col in factor_std.columns if col in factor_formulas]
    ret_cols = [col for col in ret_df.columns if col.startswith("ret_")]

    #
    if not factor_cols:
        print(f" :  {factor_formulas} cache")
        print(
            f"  cache: {[col for col in factor_std.columns if col not in ['$total_mv', '$industry', '$float_mv']]}"
        )
        sys.exit(1)

    # step1$total_mv$log_mv
    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [
        col for col in ret_cols + required_style_cols if col not in data.columns
    ]
    if missing_cols:
        print(f" : : {missing_cols}")
        sys.exit(1)

    #
    needed_cols = factor_cols + ret_cols + required_style_cols
    data = data[needed_cols]

    print(f"   : {len(factor_cols)}")
    print(f"   : {len(ret_cols)}")
    print(f"   : {required_style_cols}")

    #
    print("  ...")
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
        print(f"   : return_coef ({coef_all.shape})")

        cache_mgr.write_dataframe(t_all, "return_tval")
        print(f"   : return_tval ({t_all.shape})")

        # 内部辅助函数：计算回归系数的统计摘要
        # 内部辅助函数：计算t统计量的摘要
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

        #
        start_compact = start_date.replace("-", "")
        end_compact = end_date.replace("-", "")
        coef_summary.to_excel(
            f".cache/{market}_{start_compact}_{end_compact}__return_coef_summary.xlsx"
        )
        t_summary.to_excel(
            f".cache/{market}_{start_compact}_{end_compact}__return_tval_summary.xlsx"
        )
        print("   ")

        print("\n Step3!")
    else:
        print(" : ")
        sys.exit(1)
