import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from factor_func import (
    summarize_ic,
    summarize_group_return,
    summarize_autocorr,
    summarize_turnover,
)
from config import *


if __name__ == "__main__":
    provider_uri = "/Users/hm/Desktop/workspace/output/qlib_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    factor_df = pd.read_parquet("cache/factor_行业市值中性化.parquet")
    ret_df = pd.read_parquet("cache/ret.parquet")

    if factor_df.empty or ret_df.empty:
        print("因子或收益数据为空，跳过评估")
        raise SystemExit(0)

    merged_df = factor_df.join(ret_df, how="left")
    factor_list = list(factor_df.columns)
    ret_list = list(ret_df.columns)

    # IC / RankIC
    ic_df, ric_df, ic_summary, ric_summary = summarize_ic(merged_df, factor_list=factor_list, ret_list=ret_list)
    ic_df.to_parquet("cache/factor_ic.parquet", compression="snappy", index=True)
    ric_df.to_parquet("cache/factor_rank_ic.parquet", compression="snappy", index=True)
    ic_summary.to_excel("cache/factor_ic_summary.xlsx", index=True)
    ric_summary.to_excel("cache/factor_rank_ic_summary.xlsx", index=True)
    print("IC summary:\n", ic_summary)
    print("RankIC summary:\n", ric_summary)

    # 分组收益
    group_daily_df, group_summary = summarize_group_return(
        merged_df,
        factor_list=factor_list,
        ret_list=ret_list,
        quantile=0.2,
    )
    group_daily_df.to_parquet("cache/factor_group_return.parquet", compression="snappy", index=True)
    group_summary.to_excel("cache/factor_group_return_summary.xlsx", index=True)
    print("Group return summary:\n", group_summary)

    # 自相关
    ac_df, ac_summary = summarize_autocorr(
        merged_df,
        factor_list=factor_list,
        lag=1,
    )
    ac_df.to_parquet("cache/factor_autocorr.parquet", compression="snappy", index=True)
    ac_summary.to_excel("cache/factor_autocorr_summary.xlsx", index=True)
    print("Autocorr summary:\n", ac_summary)

    # 换手率
    turnover_daily_df, turnover_summary = summarize_turnover(
        merged_df,
        factor_list=factor_list,
        N=5,
        lag=1,
    )
    turnover_daily_df.to_parquet("cache/factor_turnover.parquet", compression="snappy", index=True)
    turnover_summary.to_excel("cache/factor_turnover_summary.xlsx", index=True)
    print("Turnover summary:\n", turnover_summary)
