import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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


def ext_out_mad(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    Median absolute deviation outlier removal.
    """
    for f in factor_list:
        factor = group[f]
        median = factor.median()
        mad = (factor - median).abs().median()
        edge_up = median + 3 * mad
        edge_low = median - 3 * mad
        factor = factor.astype("float64")
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)
        group[f] = factor.astype("float32")
    return group


def ext_out_3std(group: pd.DataFrame, factor_list: list, noise_std: float = 1e-10) -> pd.DataFrame:
    """
    3-sigma 异常值移除并添加噪音确保唯一的分箱边界。
    """
    for f in factor_list:
        factor = group[f]
        noise = np.random.normal(0, noise_std, size=len(factor))
        factor += noise
        factor = factor.astype(float)
        edge_up = factor.mean() + 3 * factor.std()
        edge_low = factor.mean() - 3 * factor.std()
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)
        group[f] = factor
    return group


def z_score(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    Z-score standardization.
    """
    for f in factor_list:
        factor = group[f]
        if factor.std() != 0:
            group[f] = (factor - factor.mean()) / factor.std()
        else:
            group[f] = np.nan
    return group


def universal_neutralization(
    group: pd.DataFrame,
    factors: Optional[List[str]],
    returns: Optional[List[str]],
    continuous_styles: List[str],
    categorical_styles: List[str],
) -> pd.DataFrame:
    """
    【单日截面】通用中性化函数：通过一次多元线性回归，同时剥离连续和分类风格的影响。
    """
    targets = list(factors or []) + list(returns or [])
    if not targets:
        return group

    X_continuous = group[continuous_styles].astype("float64")

    X_dummies = pd.DataFrame(index=group.index)
    for col in categorical_styles:
        dummies = pd.get_dummies(group[col], drop_first=True, prefix=col, dtype=float)
        X_dummies = pd.concat([X_dummies, dummies], axis=1)

    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    reg_data = pd.concat([group[targets].astype("float64"), X_styles], axis=1).dropna()
    if len(reg_data) < len(X_styles.columns) + 2:
        return group

    X_reg = reg_data[X_styles.columns]
    X_reg = sm.add_constant(X_reg, has_constant="add")

    for col in targets:
        Y_reg = reg_data[col]
        try:
            model = sm.OLS(Y_reg, X_reg).fit()
            residual = model.resid.astype(group[col].dtype, copy=False)
            group.loc[residual.index, col] = residual
        except Exception:
            print(f"忽略 {col} 的回归失败（如数据共线性、奇异矩阵等），保留原值")
            continue

    return group


def summarize_ic(
    pred: pd.Series,
    label: pd.Series,
    *,
    date_col: str = "datetime",
    dropna: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """计算 IC / Rank IC 序列及其统计指标。"""
    ic, ric = calc_ic(pred, label, date_col=date_col, dropna=dropna)

    def _summary(series: pd.Series) -> pd.Series:
        s = pd.Series(series).dropna()
        if s.empty:
            return pd.Series(dtype=float)
        mean = s.mean()
        std = s.std()
        icir = mean / std if std != 0 else np.nan
        t_value = icir * np.sqrt(len(s)) if std != 0 else np.nan
        win_rate = (s > 0).sum() / len(s)
        return pd.Series(
            {
                "mean": mean,
                "std": std,
                "icir": icir,
                "t_value": t_value,
                "win_rate": win_rate,
                "count": len(s),
            }
        )

    ic_summary = _summary(ic).rename("IC").to_frame().T
    ric_summary = _summary(ric).rename("RankIC").to_frame().T
    return ic, ric, ic_summary, ric_summary


def summarize_group_return(pred_label: pd.DataFrame, *, quantile: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """封装分组收益与多空收益的计算，返回日度序列与摘要表。"""
    long_short, long_avg = calc_long_short_return(pred_label["score"], pred_label["label"], quantile=quantile)
    daily = pd.DataFrame({"long_short": long_short, "long_avg": long_avg})

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

    summary = pd.DataFrame({col: _summary(daily[col]) for col in daily.columns}).T
    return daily, summary


def summarize_autocorr(pred: pd.Series, *, lag: int = 1) -> tuple[pd.Series, pd.DataFrame]:
    """计算预测得分的自相关序列及摘要。"""
    ac = pred_autocorr(pred, lag=lag, inst_col="instrument", date_col="datetime")

    def _summary(series: pd.Series) -> pd.DataFrame:
        s = series.dropna()
        if s.empty:
            return pd.DataFrame(columns=["mean", "std", "count"])
        summary = pd.Series({"mean": s.mean(), "std": s.std(), "count": len(s)}, name="AC")
        return summary.to_frame().T

    return ac, _summary(ac)


def summarize_turnover(pred_label: pd.DataFrame, *, N: int = 5, lag: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基于 qlib 的 Top/Bottom turnover 逻辑，输出日度换手率序列与摘要。"""
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument", group_keys=False)["score"].shift(lag)

    def _top_turnover(x: pd.DataFrame) -> float:
        top_now = x.nlargest(len(x) // N, columns="score").index
        top_prev = x.nlargest(len(x) // N, columns="score_last").index
        return 1 - top_now.isin(top_prev).sum() / (len(x) // N)

    def _bottom_turnover(x: pd.DataFrame) -> float:
        bot_now = x.nsmallest(len(x) // N, columns="score").index
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
    return daily, summary


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
