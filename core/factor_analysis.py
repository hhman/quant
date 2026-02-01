import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from typing import List, Optional, Tuple

from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, pred_autocorr
from qlib.contrib.report.analysis_model.analysis_model_performance import (
    model_performance_graph,
)
from qlib.contrib.report.analysis_position import score_ic_graph


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


def ext_out_3std(
    group: pd.DataFrame, factor_list: List[str], noise_std: float = 1e-10
) -> pd.DataFrame:
    """3-sigma"""
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


def neutralize_industry_marketcap(
    group: pd.DataFrame,
    factor_list: List[str],
    total_mv_col: str = "$total_mv",
    industry_col: str = "$industry",
    float_mv_col: str = "$float_mv",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """行业市值中性化。

    使用WLS回归对因子进行中性化：factor ~ log(total_mv) + industry_dummies，
    权重为sqrt(float_mv)

    Args:
        group: pd.DataFrame index(instrument, datetime)
        factor_list: List[str] 因子列表
        total_mv_col: str 总市值列名
        industry_col: str 行业列名
        float_mv_col: str 流通市值列名
        eps: float 防止log(0)的小常数

    Returns:
        pd.DataFrame 中性化后的因子值
    """
    if not factor_list:
        return pd.DataFrame(index=group.index)

    dt = (
        group.index.get_level_values("datetime")[0]
        if "datetime" in group.index.names
        else "unknown_dt"
    )

    log_mv = np.log(group[total_mv_col].astype("float64").clip(lower=eps))
    X_continuous = pd.DataFrame({"log_mv": log_mv}, index=group.index)

    X_dummies = pd.get_dummies(
        group[industry_col], drop_first=True, prefix=industry_col, dtype=float
    )
    if X_dummies.empty:
        print(f"[neutralize_industry_marketcap] {dt} 行业虚拟变量为空")

    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    w_all = np.sqrt(group[float_mv_col].astype("float64").clip(lower=eps)).rename("_w")
    w_mean = w_all.mean()
    if w_mean <= eps:
        w_mean = 1.0
    w_all = (w_all / w_mean).rename("_w")

    result = group[factor_list].copy()
    result.loc[:, :] = np.nan

    for factor_name in factor_list:
        reg_data = pd.concat(
            [group[[factor_name]].astype("float64"), X_styles, w_all],
            axis=1,
        )
        before_drop = len(reg_data)
        reg_data = reg_data.dropna()
        if before_drop > 0:
            drop_ratio = (before_drop - len(reg_data)) / before_drop
            if drop_ratio > 0.1:
                print(
                    f"[neutralize_industry_marketcap] {dt} {factor_name} dropna比例 {before_drop - len(reg_data)}/{before_drop} ({drop_ratio:.1%})"
                )

        n_params = len(X_styles.columns) + 2
        if len(reg_data) < n_params:
            print(
                f"[neutralize_industry_marketcap] {dt} {factor_name} 样本不足 {len(reg_data)} < {n_params} 返回NaN"
            )
            continue

        X_reg = sm.add_constant(reg_data[X_styles.columns], has_constant="add")
        Y_reg = reg_data[factor_name]
        w = reg_data["_w"]
        try:
            model = sm.WLS(Y_reg, X_reg, weights=w).fit()
            residual = model.resid.astype(group[factor_name].dtype, copy=False)
            result.loc[residual.index, factor_name] = residual

            if model.rsquared < 0.1:
                print(f"[neutralize] {dt} {factor_name} R²={model.rsquared:.3f}")
            elif model.rsquared < 0.05:
                print(f"[neutralize] {dt} {factor_name} R²={model.rsquared:.3f}")
        except Exception as e:
            print(
                f"[neutralize_industry_marketcap] {dt} {factor_name} 回归失败: {e} 返回NaN"
            )

    return result


def factor_return_industry_marketcap(
    group: pd.DataFrame,
    factor_list: List[str],
    ret_list: List[str],
    total_mv_col: str = "$total_mv",
    industry_col: str = "$industry",
    float_mv_col: str = "$float_mv",
    eps: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算因子的收益率回归系数。

    对每个因子和收益组合进行回归：ret = β0 + β1*factor + β2*log(total_mv) + Σβj*industry_j + ε
    使用WLS回归，权重为sqrt(float_mv)

    Args:
        group: pd.DataFrame index(instrument, datetime)
        factor_list: List[str] 因子列表
        ret_list: List[str] 收益率列表
        total_mv_col: str 总市值列名
        industry_col: str 行业列名
        float_mv_col: str 流通市值列名
        eps: float 防止log(0)的小常数

    Returns:
        coef_df: datetime索引，列为{factor}_{ret}的回归系数
        t_df: datetime索引，列为{factor}_{ret}的t统计量
    """
    if not factor_list or not ret_list:
        return pd.DataFrame(), pd.DataFrame()

    dt = (
        group.index.get_level_values("datetime")[0]
        if "datetime" in group.index.names
        else "unknown_dt"
    )

    log_mv = np.log(group[total_mv_col].astype("float64").clip(lower=eps))
    X_continuous = pd.DataFrame({"log_mv": log_mv}, index=group.index)

    X_dummies = pd.get_dummies(
        group[industry_col], drop_first=True, prefix=industry_col, dtype=float
    )
    if X_dummies.empty:
        print(f"[factor_return] {dt} 行业虚拟变量为空")

    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    w_all = np.sqrt(group[float_mv_col].astype("float64").clip(lower=eps)).rename("_w")
    w_mean = w_all.mean()
    if w_mean <= eps:
        w_mean = 1.0
    w_all = (w_all / w_mean).rename("_w")

    coef_row = {}
    t_row = {}

    for factor_name in factor_list:
        base_X = pd.concat([group[[factor_name]].astype("float64"), X_styles], axis=1)

        for ret_col in ret_list:
            if group[ret_col].dropna().empty:
                continue

            reg_data = pd.concat(
                [group[[ret_col]].astype("float64"), base_X, w_all],
                axis=1,
            )
            before_drop = len(reg_data)
            reg_data = reg_data.dropna()
            if before_drop > 0:
                drop_ratio = (before_drop - len(reg_data)) / before_drop
                if drop_ratio > 0.1:
                    print(
                        f"[factor_return] {dt} {factor_name}-{ret_col} dropna比例 {before_drop - len(reg_data)}/{before_drop} ({drop_ratio:.1%})"
                    )

            n_params = base_X.shape[1] + 1
            col_name = f"{factor_name}_{ret_col}"
            if len(reg_data) < n_params:
                print(
                    f"[factor_return] {dt} 因子 {factor_name} 收益 {ret_col} 样本不足 {len(reg_data)} < {n_params} 返回NaN"
                )
                coef_row[col_name] = np.nan
                t_row[col_name] = np.nan
                continue

            X_reg = sm.add_constant(reg_data.drop(columns=[ret_col, "_w"]))
            Y_reg = reg_data[ret_col]
            w_reg = reg_data["_w"]

            try:
                model = sm.WLS(Y_reg, X_reg, weights=w_reg).fit()
                coef_row[col_name] = model.params.get(factor_name, np.nan)
                t_row[col_name] = model.tvalues.get(factor_name, np.nan)

                if model.rsquared < 0.1:
                    print(
                        f"[factor_return] {dt} {factor_name}-{ret_col} R²={model.rsquared:.3f}"
                    )
            except Exception as e:
                print(
                    f"[factor_return] {dt} {factor_name} 收益 {ret_col} 回归失败: {e} 返回NaN"
                )
                coef_row[col_name] = np.nan
                t_row[col_name] = np.nan

    coef_df = pd.DataFrame([coef_row], index=[dt])
    t_df = pd.DataFrame([t_row], index=[dt])

    return coef_df, t_df


def summarize_ic(
    group: pd.DataFrame,
    factor_list: List[str],
    ret_list: List[str],
    *,
    date_col: str = "datetime",
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算IC和RankIC及其统计摘要。

    Args:
        group: pd.DataFrame MultiIndex (instrument, datetime)
        factor_list: List[str] 因子列表
        ret_list: List[str] 收益率列表
        date_col: str 日期列名，传给calc_ic
        dropna: bool 是否删除NaN值，传给calc_ic

    Returns:
        ic_df: datetime索引，列为{factor}_{ret}的IC序列
        ric_df: datetime索引，列为{factor}_{ret}的RankIC序列
        ic_summary_df: (factor, ret)为索引的IC统计摘要
        ric_summary_df: (factor, ret)为索引的RankIC统计摘要
    """

    def _summary(series: pd.Series) -> pd.Series:
        """计算IC序列的统计摘要。

        Args:
            series: IC值序列

        Returns:
            包含IC均值、标准差、IR、t统计量、胜率等指标的Series
        """
        s = pd.Series(series).dropna()
        if s.empty:
            return pd.Series(dtype=float)
        mean = s.mean()
        std = s.std()
        T = len(s)
        t_stat = mean / std * np.sqrt(T) if std != 0 else np.nan
        icir = mean / std if std != 0 else np.nan
        win_rate = (s > 0).sum() / len(s)
        gt002_rate = (s.abs() > 0.02).sum() / len(s)
        return pd.Series(
            {
                "IC Mean": mean,
                "IC Std": std,
                "IR": icir,
                "IC t": t_stat,
                "IC>0": win_rate,
                "|IC|>0.02": gt002_rate,
            }
        )

    ic_series = {}
    ric_series = {}
    ic_summaries = []
    ric_summaries = []
    for fac in factor_list:
        for ret_col in ret_list:
            ic, ric = calc_ic(
                group[fac], group[ret_col], date_col=date_col, dropna=dropna
            )
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
    """计算分组收益率及其统计摘要。

    Args:
        group: pd.DataFrame MultiIndex (instrument, datetime)
        factor_list: List[str] 因子列表
        ret_list: List[str] 收益率列表
        quantile: float 分组分位数

    Returns:
        daily_df: datetime索引，多级列{factor}_{ret}包含long_short和long_avg
        summary_df: 统计摘要DataFrame
    """

    def _summary(series: pd.Series) -> pd.Series:
        """计算收益率序列的统计摘要。

        Args:
            series: 收益率序列

        Returns:
            包含均值、标准差、累计收益、胜率等指标的Series
        """
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
            long_short, long_avg = calc_long_short_return(
                group[fac], group[ret_col], quantile=quantile
            )
            daily = pd.DataFrame({"long_short": long_short, "long_avg": long_avg})
            col_prefix = f"{fac}_{ret_col}"
            daily_blocks[col_prefix] = daily
            summary_rows.append(
                _summary(daily["long_short"]).rename(f"{col_prefix}_long_short")
            )
            summary_rows.append(
                _summary(daily["long_avg"]).rename(f"{col_prefix}_long_avg")
            )

    daily_df = pd.concat(daily_blocks, axis=1)
    summary_df = pd.DataFrame(summary_rows)
    return daily_df, summary_df


def summarize_autocorr(
    group: pd.DataFrame,
    factor_list: List[str],
    *,
    lag: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算因子自相关性及其统计摘要。

    Args:
        group: pd.DataFrame MultiIndex (instrument, datetime)
        factor_list: List[str] 因子列表
        lag: int 滞后期数

    Returns:
        ac_df: datetime索引，列为各因子的自相关序列
        summary_df: 因子为索引的统计摘要
    """

    def _summary(series: pd.Series) -> pd.Series:
        """计算自相关序列的统计摘要。

        Args:
            series: 自相关序列

        Returns:
            包含均值、标准差、样本数的Series
        """
        s = series.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        return pd.Series({"mean": s.mean(), "std": s.std(), "count": len(s)})

    ac_series = {}
    summary_rows = []
    for fac in factor_list:
        ac = pred_autocorr(
            group[fac], lag=lag, inst_col="instrument", date_col="datetime"
        )
        ac_series[fac] = ac
        summary_rows.append(_summary(ac).rename(fac))
    ac_df = pd.concat(ac_series, axis=1)
    summary_df = pd.DataFrame(summary_rows).T
    return ac_df, summary_df


def summarize_turnover(
    group: pd.DataFrame,
    factor_list: List[str],
    *,
    N: int = 5,
    lag: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算因子换手率（Top/Bottom N档）。

    Args:
        group: pd.DataFrame MultiIndex (instrument, datetime)
        factor_list: List[str] 因子列表
        N: int 分组数
        lag: int 滞后期数

    Returns:
        daily_df: datetime索引，多级列为{factor}/(top, bottom)的换手率
        summary_df: 统计摘要DataFrame
    """

    def _compute(score_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """计算换手率。

        Args:
            score_col: 得分列名

        Returns:
            (daily_df, summary_df) 日度换手率和汇总统计
        """
        pred = group[[score_col]].copy()
        pred["score_last"] = pred.groupby(level="instrument", group_keys=False)[
            score_col
        ].shift(lag)

        def _top_turnover(x: pd.DataFrame) -> float:
            """计算顶部N档的换手率。

            Args:
                x: 分组后的DataFrame

            Returns:
                换手率
            """
            k = len(x) // N
            if k == 0:
                return np.nan
            top_now = x.nlargest(k, columns=score_col).index
            top_prev = x.nlargest(k, columns="score_last").index
            return 1 - top_now.isin(top_prev).sum() / k

        def _bottom_turnover(x: pd.DataFrame) -> float:
            """计算底部N档的换手率。

            Args:
                x: 分组后的DataFrame

            Returns:
                换手率
            """
            k = len(x) // N
            if k == 0:
                return np.nan
            bot_now = x.nsmallest(k, columns=score_col).index
            bot_prev = x.nsmallest(k, columns="score_last").index
            return 1 - bot_now.isin(bot_prev).sum() / k

        top = pred.groupby(level="datetime", group_keys=False).apply(_top_turnover)
        bottom = pred.groupby(level="datetime", group_keys=False).apply(
            _bottom_turnover
        )
        daily = pd.DataFrame({"top": top, "bottom": bottom})

        def _summary(series: pd.Series) -> pd.Series:
            """计算换手率序列的统计摘要。

            Args:
                series: 换手率序列

            Returns:
                包含均值、标准差、样本数的Series
            """
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
    merged_df: pd.DataFrame,
    factor_list: List[str],
    ret_list: List[str],
    output_dir: Path,
    graph_names: Optional[List[str]] = None,
) -> None:
    """生成绩效分析图表HTML。

    Args:
        merged_df: pd.DataFrame MultiIndex数据
        factor_list: List[str] 因子列表
        ret_list: List[str] 收益率列表
        output_dir: Path 输出目录
        graph_names: List[str] 图表名称，默认["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
    """
    names = graph_names or ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for factor_name in factor_list:
        for ret_col in ret_list:
            pred_label = merged_df[[factor_name, ret_col]].copy()
            pred_label.columns = ["score", "label"]

            subdir = out_dir / f"{factor_name}_{ret_col}"
            subdir.mkdir(parents=True, exist_ok=True)

            for name in names:
                try:
                    figs = model_performance_graph(
                        pred_label, graph_names=[name], show_notebook=False
                    )
                    for j, fig in enumerate(figs, start=1):
                        filename = (
                            f"{name}.html" if len(figs) == 1 else f"{name}_{j}.html"
                        )
                        fig.write_html(str(subdir / filename))
                except Exception as e:
                    print(
                        f"[save_performance_graphs] {factor_name}-{ret_col} {name}: {e}"
                    )

            try:
                figs = score_ic_graph(pred_label, show_notebook=False)
                for i, fig in enumerate(figs, start=1):
                    name = "score_ic.html" if len(figs) == 1 else f"score_ic_{i}.html"
                    fig.write_html(str(subdir / name))
            except Exception as e:
                print(
                    f"[save_performance_graphs] {factor_name}-{ret_col} score_ic: {e}"
                )

    print(f"图表已保存: {out_dir}")
