import pandas as pd
from pathlib import Path
from factor_func import (
    ext_out_mad,
    ext_out_3std,
    z_score,
    universal_neutralization,
    summarize_ic,
    summarize_group_return,
    summarize_autocorr,
    summarize_turnover,
    save_performance_graphs,
)

import qlib
from qlib.constant import REG_CN
from qlib.data import D


market = "csi300"
start_time = "2020-01-01"
end_time = "2024-12-31"
# 日/周/月对齐的持有期（按交易日约数）
periods = {"1d": 1, "1w": 5, "1m": 20}

# 映射：列名 -> 表达式
factor_map = {"qtul5": "Quantile($close, 5, 0.8)/$close"}
return_map = {f"ret_{label}": f"Ref($close, -{lag})/$close - 1" for label, lag in periods.items()}
style_continuous_map = {"log_mv": "Log($total_mv)"}
style_categorical_map = {"industry": "$industry_id"}

# 供后续使用的列名与 qlib 字段
factor_cols = list(factor_map.keys())
return_cols = list(return_map.keys())
style_continuous_cols = list(style_continuous_map.keys())
style_categorical_cols = list(style_categorical_map.keys())
fields = (
    list(factor_map.values())
    + list(return_map.values())
    + list(style_continuous_map.values())
    + list(style_categorical_map.values())
)


if __name__ == "__main__":
    provider_uri = "/Users/hm/Desktop/workspace/output/qlib_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    instruments = D.instruments(market=market)
    
    df = D.features(instruments=instruments, fields=fields, start_time=start_time, end_time=end_time, freq="day")
    df.columns = factor_cols + return_cols + style_continuous_cols + style_categorical_cols

    # 去极值
    if False:
        df = df.groupby(level="datetime", group_keys=False).apply(lambda x: ext_out_mad(x, factor_cols))
    else:
        df = df.groupby(level="datetime", group_keys=False).apply(lambda x: ext_out_3std(x, factor_cols))

    # 标准化
    df = df.groupby(level="datetime", group_keys=False).apply(lambda x: z_score(x, factor_cols))

    # 中性化
    df = df.groupby(level="datetime", group_keys=False).apply(
        lambda x: universal_neutralization(x, factor_cols, return_cols, style_continuous_cols, style_categorical_cols)
    )

    for factor_col in factor_cols:
        for return_col in return_cols:
            # 构造评分/标签
            pred_label = df[[factor_col, return_col]].rename(columns={factor_col: "score", return_col: "label"})

            # IC / Rank IC 及衍生统计
            ic, ric, ic_summary, ric_summary = summarize_ic(
                pred_label["score"], pred_label["label"], date_col="datetime", dropna=True
            )
            # 分组收益（多空、纯多头）
            group_daily, group_summary = summarize_group_return(pred_label, quantile=0.2)
            # 信号自相关
            ac_series, ac_summary = summarize_autocorr(pred_label["score"], lag=1)
            # Top/Bottom 换手率
            turnover_daily, turnover_summary = summarize_turnover(pred_label, N=5, lag=1)

            # 生成性能图
            output_dir = Path(f"output/{factor_col}_{return_col}")
            save_performance_graphs(pred_label, output_dir)

            # 统一打印
            print(f"因子 {factor_col} / 收益 {return_col} 的评估：")
            print("IC 摘要：")
            print(ic_summary)
            print("Rank IC 摘要：")
            print(ric_summary)
            print("分组收益（多空/多头）摘要：")
            print(group_summary)
            print("得分自相关摘要：")
            print(ac_summary)
            print("Top/Bottom 换手率摘要：")
            print(turnover_summary)
            print("性能图已保存至目录:", (output_dir).resolve())
