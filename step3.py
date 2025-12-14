import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from factor_func import (
    factor_return_regression,
    summarize_factor_return,
)
from config import *


if __name__ == "__main__":
    provider_uri = "/Users/hm/Desktop/workspace/output/qlib_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    instruments = D.instruments(market=market)
    
    total_mv = D.features(instruments=instruments, fields=["$total_mv"], start_time=start_time, end_time=end_time, freq="day")

    total_mv["$total_mv"] = (
        total_mv
        .groupby(level="instrument")["$total_mv"]
        .ffill().bfill()
    )

    eps = 1e-8
    log_mv = pd.DataFrame()
    log_mv["$log_mv"] = np.log(total_mv["$total_mv"] + eps)

    industry = D.features(instruments=instruments, fields=["$industry"], start_time=start_time, end_time=end_time, freq="day")

    industry["$industry"] = (
        industry
        .groupby(level="instrument")["$industry"]
        .ffill().bfill()
    )

    float_mv = D.features(instruments=instruments, fields=["$float_mv"], start_time=start_time, end_time=end_time, freq="day")

    float_mv["$float_mv"] = (
        float_mv
        .groupby(level="instrument")["$float_mv"]
        .ffill().bfill()
    )

    factor_df = pd.read_parquet("cache/factor_去极值标准化.parquet")
    ret_df = pd.read_parquet("cache/ret.parquet")

    merged = (
        factor_df
        .join(ret_df, how="left")        
        .join(log_mv, how="left")
        .join(industry, how="left")
        .join(float_mv, how="left")
    )

    coef_list = []
    t_list = []
    for dt in merged.index.get_level_values("datetime").unique():
        daily_group = (
            merged.xs(dt, level="datetime")
            .assign(datetime=dt)
            .set_index("datetime", append=True)
            .reorder_levels(["instrument", "datetime"])
        )
        coef_df, t_df = factor_return_regression(
            daily_group,
            factor_list=factor_df.columns.to_list(),
            ret_list=ret_df.columns.to_list(),
            continuous_styles=["$log_mv"],
            categorical_styles=["$industry"],
            weight_col="$float_mv",
        )
        coef_df["datetime"] = dt
        t_df["datetime"] = dt
        coef_df = coef_df.set_index("datetime")
        t_df = t_df.set_index("datetime")
        coef_list.append(coef_df)
        t_list.append(t_df)

    if coef_list and t_list:
        coef_all = pd.concat(coef_list, axis=0)
        t_all = pd.concat(t_list, axis=0)
        coef_all.to_parquet("cache/factor_回归收益率.parquet", compression="snappy", index=True)
        t_all.to_parquet("cache/factor_回归t值.parquet", compression="snappy", index=True)
        print(coef_all)
        print(t_all)

        coef_summary, t_summary = summarize_factor_return(coef_all, t_all)
        coef_summary.to_excel("cache/factor_回归收益率_summary.xlsx")
        t_summary.to_excel("cache/factor_回归t值_summary.xlsx")
        print(coef_summary)
        print(t_summary)
    else:
        print("未生成回归结果，因样本不足或全部回归失败")
