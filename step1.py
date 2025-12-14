import qlib
from qlib.constant import REG_CN
from qlib.data import D
from factor_func import (
    ext_out_3std,
    z_score,
)
from config import *


if __name__ == "__main__":
    provider_uri = "/Users/hm/Desktop/workspace/output/qlib_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    instruments = D.instruments(market=market)

    factor_df = D.features(
        instruments=instruments,
        fields=factor_fields,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    factor_df.columns = factor_names

    # 丢弃全空因子列，避免无效列进入后续流程
    dropped_cols = [col for col in factor_df.columns if factor_df[col].isna().all()]
    if dropped_cols:
        print(f"因子全空，已丢弃: {dropped_cols}")
        factor_df = factor_df.drop(columns=dropped_cols)
    valid_factor_cols = factor_df.columns.tolist()

    factor_df.to_parquet("cache/factor_原始.parquet", compression="snappy", index=True)
    print(factor_df)

    # 去极值
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(lambda x: ext_out_3std(x, valid_factor_cols))

    # 标准化
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(lambda x: z_score(x, valid_factor_cols))

    factor_df.to_parquet("cache/factor_去极值标准化.parquet", compression="snappy", index=True)
    print(factor_df)

    # 收益率
    ret_df = D.features(instruments=instruments, fields=ret_map.values(), start_time=start_time, end_time=end_time, freq="day")
    ret_df.columns = ret_map.keys()

    ret_df.to_parquet("cache/ret.parquet", compression="snappy", index=True)
    print(ret_df)
