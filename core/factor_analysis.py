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
    """


    factor ~ log(total_mv) + industry_dummiesWLSsqrt(float_mv)

    Args:
        group: pd.DataFrame index(instrument, datetime)
        factor_list: List[str]
        total_mv_col: str
        industry_col: str
        float_mv_col: str
        eps: float 0

    Returns:
        pd.DataFrame
    """
    if not factor_list:
        return pd.DataFrame(index=group.index)

    dt = (
        group.index.get_level_values("datetime")[0]
        if "datetime" in group.index.names
        else "unknown_dt"
    )

    # 1)
    log_mv = np.log(group[total_mv_col].astype("float64").clip(lower=eps))
    X_continuous = pd.DataFrame({"log_mv": log_mv}, index=group.index)

    #
    X_dummies = pd.get_dummies(
        group[industry_col], drop_first=True, prefix=industry_col, dtype=float
    )
    if X_dummies.empty:
        print(f"[neutralize_industry_marketcap] {dt} ")

    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    #
    w_all = np.sqrt(group[float_mv_col].astype("float64").clip(lower=eps)).rename("_w")
    w_mean = w_all.mean()
    if w_mean <= eps:
        w_mean = 1.0
    w_all = (w_all / w_mean).rename("_w")

    # 2) dtype
    result = group[factor_list].copy()
    # NaN
    result.loc[:, :] = np.nan

    # 3) X_stylesw_all
    for factor_name in factor_list:
        #  +  +
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
                    f"[neutralize_industry_marketcap] {dt} {factor_name} dropna  {before_drop - len(reg_data)}/{before_drop} ({drop_ratio:.1%})"
                )

        n_params = len(X_styles.columns) + 2  #  +  +
        if len(reg_data) < n_params:
            print(
                f"[neutralize_industry_marketcap] {dt} {factor_name}  {len(reg_data)}  {n_params} NaN"
            )
            # result[factor_name] NaN
            continue

        X_reg = sm.add_constant(reg_data[X_styles.columns], has_constant="add")
        Y_reg = reg_data[factor_name]
        w = reg_data["_w"]
        try:
            model = sm.WLS(Y_reg, X_reg, weights=w).fit()
            residual = model.resid.astype(group[factor_name].dtype, copy=False)
            result.loc[residual.index, factor_name] = residual

            #
            if model.rsquared < 0.1:
                print(f"[neutralize] {dt} {factor_name} R²={model.rsquared:.3f}")
            elif model.rsquared < 0.05:
                print(f"[neutralize] {dt} {factor_name} R²={model.rsquared:.3f}")
        except Exception as e:
            print(f"[neutralize_industry_marketcap] {dt} {factor_name} {e} NaN")
            # result[factor_name] NaN

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
    """


    factorret_col
        ret = β0 + β1*factor + β2*log(total_mv) + Σβj*industry_j + ε
    WLSsqrt(float_mv)

    Args:
        group: pd.DataFrame index(instrument, datetime)
        factor_list: List[str]
        ret_list: List[str]
        total_mv_col: str
        industry_col: str
        float_mv_col: str
        eps: float 0

    Returns:
        coef_df: datetime{factor}_{ret}
        t_df: tdatetime{factor}_{ret}
    """
    if not factor_list or not ret_list:
        return pd.DataFrame(), pd.DataFrame()

    dt = (
        group.index.get_level_values("datetime")[0]
        if "datetime" in group.index.names
        else "unknown_dt"
    )

    # 1)
    log_mv = np.log(group[total_mv_col].astype("float64").clip(lower=eps))
    X_continuous = pd.DataFrame({"log_mv": log_mv}, index=group.index)

    #
    X_dummies = pd.get_dummies(
        group[industry_col], drop_first=True, prefix=industry_col, dtype=float
    )
    if X_dummies.empty:
        print(f"[factor_return] {dt} ")

    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    #
    w_all = np.sqrt(group[float_mv_col].astype("float64").clip(lower=eps)).rename("_w")
    w_mean = w_all.mean()
    if w_mean <= eps:
        w_mean = 1.0
    w_all = (w_all / w_mean).rename("_w")

    # 2)  ×
    coef_row = {}
    t_row = {}

    for factor_name in factor_list:
        # Xret_col
        base_X = pd.concat([group[[factor_name]].astype("float64"), X_styles], axis=1)

        for ret_col in ret_list:
            if group[ret_col].dropna().empty:
                continue

            #
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
                        f"[factor_return] {dt} {factor_name}-{ret_col} dropna  {before_drop - len(reg_data)}/{before_drop} ({drop_ratio:.1%})"
                    )

            n_params = base_X.shape[1] + 1  #  +  +
            col_name = f"{factor_name}_{ret_col}"
            if len(reg_data) < n_params:
                print(
                    f"[factor_return] {dt}  {factor_name}  {ret_col}  {len(reg_data)} <  {n_params} NaN"
                )
                coef_row[col_name] = np.nan
                t_row[col_name] = np.nan
                continue

            #
            X_reg = sm.add_constant(reg_data.drop(columns=[ret_col, "_w"]))
            Y_reg = reg_data[ret_col]
            w_reg = reg_data["_w"]

            try:
                model = sm.WLS(Y_reg, X_reg, weights=w_reg).fit()
                coef_row[col_name] = model.params.get(factor_name, np.nan)
                t_row[col_name] = model.tvalues.get(factor_name, np.nan)

                #
                if model.rsquared < 0.1:
                    print(
                        f"[factor_return] {dt} {factor_name}-{ret_col} R²={model.rsquared:.3f}"
                    )
            except Exception as e:
                print(f"[factor_return] {dt} {factor_name}  {ret_col} {e} NaN")
                coef_row[col_name] = np.nan
                t_row[col_name] = np.nan

    # 3) DataFrame
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
    """
    ×  IC / RankIC


       group: pd.DataFrame
       factor_list: List[str]
       ret_list: List[str]
       date_col: str  calc_ic
       dropna: bool  calc_ic


       ic_map: {(fac, ret): IC }
       ric_map: {(fac, ret): RankIC }
       ic_summary_df:  (fac, ret)  IC
       ric_summary_df:  (fac, ret)  RankIC
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
    """
    ×


       group: pd.DataFrame
       factor_list: List[str]
       ret_list: List[str]
       quantile: float


       daily_df: pd.DataFrame {factor}_{ret}  long_short/long_avg
       summary_df: pd.DataFrame
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
    """



    ac_df: pd.DataFrame
    summary_df: pd.DataFrame
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
    """
    qlib  Top/Bottom turnover


       daily_df: pd.DataFrame top/bottom
       summary_df: pd.DataFrame/
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
    """
     html ×

    Args:
        merged_df:
        factor_list:
        ret_list:
        output_dir:
        graph_names:  ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
    """
    names = graph_names or ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ×
    for factor_name in factor_list:
        for ret_col in ret_list:
            # ×
            # pred_labelDataFrame
            pred_label = merged_df[[factor_name, ret_col]].copy()
            pred_label.columns = ["score", "label"]  # qlib

            # factor_ret/
            subdir = out_dir / f"{factor_name}_{ret_col}"
            subdir.mkdir(parents=True, exist_ok=True)

            #
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

            # score_ic
            try:
                figs = score_ic_graph(pred_label, show_notebook=False)
                for i, fig in enumerate(figs, start=1):
                    name = "score_ic.html" if len(figs) == 1 else f"score_ic_{i}.html"
                    fig.write_html(str(subdir / name))
            except Exception as e:
                print(
                    f"[save_performance_graphs] {factor_name}-{ret_col} score_ic: {e}"
                )

    print(f"   : {out_dir}")
