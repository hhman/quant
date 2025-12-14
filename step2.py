import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from factor_func import (
    universal_neutralization,
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

    merged = (
        factor_df
        .join(log_mv, how="left")
        .join(industry, how="left")
        .join(float_mv, how="left")
    )

    result_list = []
    for dt in merged.index.get_level_values("datetime").unique():
        daily_group = (
            merged.xs(dt, level="datetime")
            .assign(datetime=dt)
            .set_index("datetime", append=True)
            .reorder_levels(["instrument", "datetime"])
        )
        daily = universal_neutralization(
            daily_group,
            factor_list=factor_df.columns.to_list(),
            continuous_styles=["$log_mv"],
            categorical_styles=["$industry"],
            weight_col="$float_mv",
        )
        result_list.append(daily)


    if result_list:
        result = pd.concat(result_list, axis=0)
        result = result.sort_index(level=["instrument", "datetime"])
        result.to_parquet("cache/factor_行业市值中性化.parquet", compression="snappy", index=True)
        print(result)
    else:
        print("未生成中性化结果，因样本不足或全部回归失败")
