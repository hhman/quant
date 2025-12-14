import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, pred_autocorr
from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import (
    report_graph,
    risk_analysis_graph,
    score_ic_graph
)
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest import backtest


def ext_out_mad(group: pd.DataFrame, factor_list: List[str]) -> pd.DataFrame:
    """Median absolute deviation outlier removal."""
    for factor_name in factor_list:
        factor = group[factor_name]
        median = factor.median()
        mad = (factor - median).abs().median()
        edge_up = median + 3 * mad
        edge_low = median - 3 * mad
        factor = factor.astype("float64")
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)
        group[factor_name] = factor.astype("float32")
    return group


def ext_out_3std(group: pd.DataFrame, factor_list: List[str], noise_std: float = 1e-10) -> pd.DataFrame:
    """3-sigma 异常值移除并添加噪音确保唯一的分箱边界。"""
    for factor_name in factor_list:
        factor = group[factor_name]
        noise = np.random.normal(0, noise_std, size=len(factor))
        factor += noise
        factor = factor.astype(float)
        edge_up = factor.mean() + 3 * factor.std()
        edge_low = factor.mean() - 3 * factor.std()
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)
        group[factor_name] = factor
    return group


def z_score(group: pd.DataFrame, factor_list: List[str]) -> pd.DataFrame:
    """Z-score standardization."""
    for factor_name in factor_list:
        factor = group[factor_name]
        if factor.std() != 0:
            group[factor_name] = (factor - factor.mean()) / factor.std()
        else:
            group[factor_name] = np.nan
    return group


def universal_neutralization(
    group: pd.DataFrame,
    factor_list: List[str],
    continuous_styles: List[str],
    categorical_styles: List[str],
    weight_col: str,
    eps: float = 1e-8,  # 用于防止权重为 0 或极小值
) -> pd.DataFrame:
    """
    【单日截面】通用中性化函数（WLS 版，带权重安全保护）

    通过一次多元线性回归，将因子值对连续/分类风格的影响剔除，返回仅包含因子列的新 DataFrame（原始 group 不改写）。

    入参类型：
        group: pd.DataFrame 截面数据，index 通常为 MultiIndex(datetime, instrument)
        factor_list: List[str] 需要中性化的因子列名
        continuous_styles: List[str] 连续型风格列名
        categorical_styles: List[str] 分类型风格列名
        weight_col: str WLS 权重列名（取平方根后使用）
        eps: float 权重下限，防止 0 或过小

    返回：
        pd.DataFrame 仅保留 factor_list，对应残差（同索引）
    """
    if not factor_list:
        return pd.DataFrame(index=group.index)

    # 仅保留因子列，生成新的 DataFrame（一次性拷贝，避免碎片化）
    result = group[factor_list].copy()

    # ---------- 处理连续风格 ----------
    X_continuous = group[continuous_styles].astype("float64")

    # ---------- 处理分类风格 ----------
    X_dummies = pd.DataFrame(index=group.index)
    for col in categorical_styles:
        dummies = pd.get_dummies(group[col], drop_first=True, prefix=col, dtype=float)
        X_dummies = pd.concat([X_dummies, dummies], axis=1)

    # ---------- 合并连续和分类风格 ----------
    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    # ---------- 准备回归数据（含权重） ----------
    # clip(eps) 防止流通市值为 0 或极小值
    reg_data = pd.concat(
        [
            group[factor_list].astype("float64"),                   # 因变量
            X_styles,                                               # 自变量
            np.sqrt(group[weight_col].astype("float64").clip(lower=eps)).rename("_w"),  # WLS 权重
        ],
        axis=1,
    ).dropna()  # 丢弃缺失值

    # ---------- 样本数保护 ----------
    if len(reg_data) < len(X_styles.columns) + 2:
        dt = group.index.get_level_values("datetime")[0] if "datetime" in group.index.names else "unknown_dt"
        print(f"[neutralize] {dt} 样本 {len(reg_data)} 少于所需 {len(X_styles.columns) + 2}，跳过中性化")
        return result

    # ---------- 构建回归自变量矩阵 ----------
    X_reg = reg_data[X_styles.columns]
    X_reg = sm.add_constant(X_reg, has_constant="add")  # 添加截距项
    w = reg_data["_w"]  # WLS 权重

    # ---------- 对每个因子做加权回归 ----------
    for factor_name in factor_list:
        Y_reg = reg_data[factor_name]
        try:
            model = sm.WLS(Y_reg, X_reg, weights=w).fit()  # 加权最小二乘回归
            residual = model.resid.astype(group[factor_name].dtype, copy=False)  # 保持原数据类型
            result.loc[residual.index, factor_name] = residual  # 写回新 DataFrame
        except Exception as e:
            dt = group.index.get_level_values("datetime")[0] if "datetime" in group.index.names else "unknown_dt"
            print(f"[neutralize] {dt} 忽略 {factor_name} 的回归失败（如数据共线性、奇异矩阵等）：{e}；保留原值")
            continue

    # ---------- 返回中性化结果 ----------
    return result


def factor_return_regression(
    group: pd.DataFrame,
    factor_list: List[str],
    ret_list: List[str],
    continuous_styles: List[str],
    categorical_styles: List[str],
    weight_col: str,
    eps: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    【单日截面】单因子 × 多持仓周期因子收益回归（WLS 版）

    对每个因子 factor 和每个持仓周期 ret_col 做截面回归：
        r_{i,T+h} = X_{factor,T} factor_{i,T} + Σ_k B_{k,T} style_{k,i,T} + Σ_j X_{j,T} industry_{j,i,T} + μ_{i,T}

    入参类型：
        group: pd.DataFrame 截面数据
        factor_list: List[str] 因子列名
        ret_list: List[str] 收益列名
        continuous_styles / categorical_styles: List[str] 连续/分类风格列名
        weight_col: str 权重列名（取平方根后使用）
        eps: float 权重下限

    返回：
        Tuple[pd.DataFrame, pd.DataFrame]
            coef_df: 单行（RangeIndex），列名 {factor}_{ret_col}
            t_df:    单行（RangeIndex），列名 {factor}_{ret_col}
    """
    if not factor_list or not ret_list:
        empty = pd.DataFrame()
        return empty, empty

    # ---------- 处理连续风格 ----------
    X_continuous = group[continuous_styles].astype("float64")

    # ---------- 处理分类风格 ----------
    X_dummies = pd.DataFrame(index=group.index)
    for col in categorical_styles:
        dummies = pd.get_dummies(
            group[col],
            drop_first=True,
            prefix=col,
            dtype=float,
        )
        X_dummies = pd.concat([X_dummies, dummies], axis=1)

    # ---------- 合并连续和分类风格 ----------
    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    # ---------- WLS 权重 ----------
    w = np.sqrt(group[weight_col].astype("float64").clip(lower=eps))

    coef_row = {}
    t_row = {}

    # ---------- 外层循环：因子 ----------
    for factor_name in factor_list:
        # 当前因子列 + 风格
        base_X = pd.concat([group[[factor_name]].astype("float64"), X_styles], axis=1)

        # ---------- 内层循环：持仓周期 ----------
        for ret_col in ret_list:
            # 若当日该周期收益全空，静默跳过
            if group[ret_col].dropna().empty:
                continue
            reg_data = pd.concat(
                [group[[ret_col]].astype("float64"), base_X, w.rename("_w")],
                axis=1,
            ).dropna()

            # 样本数保护
            n_params = base_X.shape[1] + 1  # 因子+连续+分类+截距
            if len(reg_data) < n_params + 1:
                dt = group.index.get_level_values("datetime")[0] if "datetime" in group.index.names else "unknown_dt"
                print(f"[factor_ret] {dt} 跳过 {factor_name} 对 {ret_col} 的回归：样本 {len(reg_data)} < 所需 {n_params + 1}")
                continue

            X_reg = sm.add_constant(reg_data.drop(columns=[ret_col, "_w"]))
            Y_reg = reg_data[ret_col]
            w_reg = reg_data["_w"]

            try:
                model = sm.WLS(Y_reg, X_reg, weights=w_reg).fit()
                col_name = f"{factor_name}_{ret_col}"
                coef_row[col_name] = model.params.get(factor_name, np.nan)
                t_row[col_name] = model.tvalues.get(factor_name, np.nan)
            except Exception as e:
                # 异常保护，保留 NaN 并提示
                dt = group.index.get_level_values("datetime")[0] if "datetime" in group.index.names else "unknown_dt"
                print(f"[factor_ret] {dt} 忽略 {factor_name} 对 {ret_col} 的回归失败（{e}），保留 NaN")
                col_name = f"{factor_name}_{ret_col}"
                coef_row[col_name] = np.nan
                t_row[col_name] = np.nan

    coef_df = pd.DataFrame([coef_row])
    t_df = pd.DataFrame([t_row])
    return coef_df, t_df


def summarize_factor_return(
    coef_df: pd.DataFrame,
    t_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对因子收益回归的系数 / t 值序列做汇总指标。

    入参：
        coef_df: pd.DataFrame，系数序列（索引为 datetime，列为 {factor}_{ret}）
        t_df: pd.DataFrame，t 值序列（索引为 datetime，列为 {factor}_{ret}）

    返回：
        Tuple[pd.DataFrame, pd.DataFrame]，分别对应系数和 t 序列的汇总：
            系数：因子收益率均值、因子收益率序列 t 检验等
            t 值：|t|均值、|t|>2 占比、t 均值、t 均值/标准差等
    """
    def _coef_summary(series: pd.Series) -> pd.Series:
        s = series.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        mean = s.mean()
        std = s.std()
        t_test = mean / std * np.sqrt(len(s)) if std != 0 else np.nan  # 序列均值的 t 检验
        return pd.Series(
            {
                "因子收益率均值": mean,
                "因子收益率序列t检验": t_test,
            }
        )

    def _t_summary(series: pd.Series) -> pd.Series:
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
                "|t|均值": abs_mean,
                "|t|>2占比": gt2_rate,
                "t均值": t_mean,
                "t均值/标准差": t_mean_over_std,
            }
        )

    coef_summary = pd.DataFrame({col: _coef_summary(coef_df[col]) for col in coef_df.columns}).T
    t_summary = pd.DataFrame({col: _t_summary(t_df[col]) for col in t_df.columns}).T
    return coef_summary, t_summary

def summarize_ic(
    group: pd.DataFrame,
    factor_list: List[str],
    ret_list: List[str],
    *,
    date_col: str = "datetime",
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    针对多个因子 × 多个收益列计算 IC / RankIC 及摘要指标。

    入参：
        group: pd.DataFrame，包含因子列与收益列
        factor_list: List[str] 因子列名
        ret_list: List[str] 收益列名
        date_col: str 日期列名（用于 calc_ic）
        dropna: bool 是否在 calc_ic 中丢弃缺失

    返回：
        ic_map: {(fac, ret): IC 序列}
        ric_map: {(fac, ret): RankIC 序列}
        ic_summary_df: 各 (fac, ret) 的 IC 汇总
        ric_summary_df: 各 (fac, ret) 的 RankIC 汇总
    """

    def _summary(series: pd.Series) -> pd.Series:
        s = pd.Series(series).dropna()
        if s.empty:
            return pd.Series(dtype=float)
        mean = s.mean()
        std = s.std()
        icir = mean / std if std != 0 else np.nan
        win_rate = (s > 0).sum() / len(s)
        gt002_rate = (s.abs() > 0.02).sum() / len(s)
        return pd.Series(
            {
                "IC序列均值": mean,
                "IC序列标准差": std,
                "IR比率": icir,
                "IC>0占比": win_rate,
                "|IC|>0.02占比": gt002_rate,
            }
        )

    ic_series = {}
    ric_series = {}
    ic_summaries = []
    ric_summaries = []
    for fac in factor_list:
        for ret_col in ret_list:
            ic, ric = calc_ic(group[fac], group[ret_col], date_col=date_col, dropna=dropna)
            key = (fac, ret_col)
            col_name = f"{fac}_{ret_col}"
            ic_series[col_name] = ic
            ric_series[col_name] = ric
            ic_summaries.append(_summary(ic).rename(col_name))
            ric_summaries.append(_summary(ric).rename(col_name))

    ic_df = pd.concat(ic_series, axis=1)
    ric_df = pd.concat(ric_series, axis=1)
    ic_summary_df = pd.DataFrame(ic_summaries)
    ric_summary_df = pd.DataFrame(ric_summaries)
    return ic_df, ric_df, ic_summary_df, ric_summary_df


def summarize_group_return(
    group: pd.DataFrame,
    factor_list: List[str],
    ret_list: List[str],
    *,
    quantile: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    封装分组收益与多空收益的计算（多因子 × 多持仓周期）。

    入参：
        group: pd.DataFrame，包含因子与收益列
        factor_list: List[str] 因子列名
        ret_list: List[str] 收益列名
        quantile: float 分组分位数

    返回：
        daily_df: pd.DataFrame，多层列 {factor}_{ret} 下含 long_short/long_avg
        summary_df: pd.DataFrame，各列分组收益的汇总指标
    """

    def _summary(series: pd.Series) -> pd.Series:
        s = series.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        return pd.Series(
            {
                "mean": s.mean(),
                "std": s.std(),
                "cum_return": s.sum(),
                "win_rate": (s > 0).sum() / len(s),
                "count": len(s),
            }
        )

    daily_blocks = {}
    summary_rows = []
    for fac in factor_list:
        for ret_col in ret_list:
            long_short, long_avg = calc_long_short_return(group[fac], group[ret_col], quantile=quantile)
            daily = pd.DataFrame({"long_short": long_short, "long_avg": long_avg})
            col_prefix = f"{fac}_{ret_col}"
            daily_blocks[col_prefix] = daily
            summary_rows.append(_summary(daily["long_short"]).rename(f"{col_prefix}_long_short"))
            summary_rows.append(_summary(daily["long_avg"]).rename(f"{col_prefix}_long_avg"))

    daily_df = pd.concat(daily_blocks, axis=1)
    summary_df = pd.DataFrame(summary_rows)
    return daily_df, summary_df


def summarize_autocorr(
    group: pd.DataFrame,
    factor_list: List[str],
    *,
    lag: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算预测得分的自相关序列及摘要（多因子）。

    返回：
        ac_df: pd.DataFrame，列为因子名，对应自相关序列
        summary_df: pd.DataFrame，自相关汇总指标
    """

    def _summary(series: pd.Series) -> pd.DataFrame:
        s = series.dropna()
        if s.empty:
            return pd.DataFrame(columns=["mean", "std", "count"])
        summary = pd.Series({"mean": s.mean(), "std": s.std(), "count": len(s)}, name="AC")
        return summary.to_frame().T

    ac_series = {}
    summary_rows = []
    for fac in factor_list:
        ac = pred_autocorr(group[fac], lag=lag, inst_col="instrument", date_col="datetime")
        ac_series[fac] = ac
        summary_rows.append(_summary(ac).rename(index={"AC": fac}).iloc[0])
    ac_df = pd.concat(ac_series, axis=1)
    summary_df = pd.DataFrame(summary_rows)
    return ac_df, summary_df


def summarize_turnover(
    group: pd.DataFrame,
    factor_list: List[str],
    *,
    N: int = 5,
    lag: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于 qlib 的 Top/Bottom turnover 逻辑（多因子）。

    返回：
        daily_df: pd.DataFrame，多层列为因子名下的 top/bottom 换手率
        summary_df: pd.DataFrame，按因子汇总换手率均值/波动
    """

    def _compute(score_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pred = group[[score_col]].copy()
        pred["score_last"] = pred.groupby(level="instrument", group_keys=False)[score_col].shift(lag)

        def _top_turnover(x: pd.DataFrame) -> float:
            top_now = x.nlargest(len(x) // N, columns=score_col).index
            top_prev = x.nlargest(len(x) // N, columns="score_last").index
            return 1 - top_now.isin(top_prev).sum() / (len(x) // N)

        def _bottom_turnover(x: pd.DataFrame) -> float:
            bot_now = x.nsmallest(len(x) // N, columns=score_col).index
            bot_prev = x.nsmallest(len(x) // N, columns="score_last").index
            return 1 - bot_now.isin(bot_prev).sum() / (len(x) // N)

        top = pred.groupby(level="datetime", group_keys=False).apply(_top_turnover)
        bottom = pred.groupby(level="datetime", group_keys=False).apply(_bottom_turnover)
        daily = pd.DataFrame({"top": top, "bottom": bottom})

        def _summary(series: pd.Series) -> pd.Series:
            s = series.dropna()
            if s.empty:
                return pd.Series(dtype=float)
            return pd.Series({"mean": s.mean(), "std": s.std(), "count": len(s)})

        summary = pd.DataFrame({col: _summary(daily[col]) for col in daily.columns}).T
        summary.index = [f"{score_col}_{idx}" for idx in summary.index]
        return daily, summary

    daily_blocks = {}
    summary_rows = []
    for fac in factor_list:
        daily, summary = _compute(fac)
        daily_blocks[fac] = daily
        summary_rows.append(summary)
    daily_df = pd.concat(daily_blocks, axis=1)
    summary_df = pd.concat(summary_rows, axis=0)
    return daily_df, summary_df


def save_performance_graphs(
    pred_label: pd.DataFrame,
    output_dir: Path,
    graph_names: Optional[List[str]] = None,
) -> None:
    """生成性能图并保存为 html 文件。"""
    names = graph_names or ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        figs = model_performance_graph(pred_label, graph_names=[name], show_notebook=False)
        for j, fig in enumerate(figs, start=1):
            filename = f"{name}.html" if len(figs) == 1 else f"{name}_{j}.html"
            fig.write_html(str(out_dir / filename))
    figs = score_ic_graph(pred_label, show_notebook=False)
    for i, fig in enumerate(figs, start=1):
        name = "score_ic.html" if len(figs) == 1 else f"score_ic_{i}.html"
        fig.write_html(str(output_dir / name))


def run_backtest(
    signal_series: pd.Series,
    hold_thresh: int,
    output_dir: Path,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, dict]], Dict[str, Tuple[pd.DataFrame, object]]]:
    """运行回测并将回测图表输出到指定目录。"""

    # 检查索引
    if signal_series.index.names != ["instrument", "datetime"]:
        raise ValueError("signal_series 的索引必须为 (instrument, datetime) 的 MultiIndex")

    # 转换与排序
    signal_series = signal_series.swaplevel().sort_index()

    # 时间范围
    dt_index = signal_series.index.get_level_values("datetime")
    start_time = dt_index.min().strftime("%Y-%m-%d")
    backtest_end = (dt_index.max() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")

    # 策略
    strategy = TopkDropoutStrategy(
        signal=signal_series,
        topk=5,
        n_drop=0,
        hold_thresh=hold_thresh,
    )

    # 执行器
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        verbose=True,
    )

    # 运行回测
    portfolio_dict, indicator_dict = backtest(
        start_time=start_time,
        end_time=backtest_end,
        strategy=strategy,
        executor=executor,
        benchmark="SH000300",
        account=100_000_000,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )

    # 保存回测图表
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_normal_df, positions_normal = portfolio_dict["1day"]
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(
        report_normal_df["return"] - report_normal_df["bench"], freq="day"
    )
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"], freq="day"
    )
    analysis_df = pd.concat(analysis)

    # --- 1) report_graph ---
    figs = report_graph(report_normal_df, show_notebook=False)
    for i, fig in enumerate(figs, start=1):
        name = "report_graph.html" if len(figs) == 1 else f"report_graph_{i}.html"
        fig.write_html(str(output_dir / name))

    # --- 2) risk_analysis_graph ---
    figs = risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    for i, fig in enumerate(figs, start=1):
        name = "risk_analysis.html" if len(figs) == 1 else f"risk_analysis_{i}.html"
        fig.write_html(str(output_dir / name))

    return portfolio_dict, indicator_dict
