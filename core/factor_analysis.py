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


def neutralize_industry_marketcap(
    group: pd.DataFrame,
    factor_list: List[str],
    total_mv_col: str = "$total_mv",
    industry_col: str = "$industry",
    float_mv_col: str = "$float_mv",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    【单日截面】行业市值中性化（专用版本，内存优化）

    对每个因子执行：factor ~ log(total_mv) + industry_dummies，WLS权重为sqrt(float_mv)

    Args:
        group: pd.DataFrame 单日截面数据，index为(instrument, datetime)
        factor_list: List[str] 需要中性化的因子列名
        total_mv_col: str 总市值列名（将在函数内取对数）
        industry_col: str 行业分类列名（将转换为哑变量）
        float_mv_col: str 流通市值列名（将在函数内取平方根作为权重）
        eps: float 防止权重为0的保护值

    Returns:
        pd.DataFrame 仅包含中性化后的因子列（残差）
    """
    if not factor_list:
        return pd.DataFrame(index=group.index)

    dt = (
        group.index.get_level_values("datetime")[0]
        if "datetime" in group.index.names
        else "unknown_dt"
    )

    # 1) 预先构造风格变量（一次性计算，所有因子共享）
    log_mv = np.log(group[total_mv_col].astype("float64").clip(lower=eps))
    X_continuous = pd.DataFrame({"log_mv": log_mv}, index=group.index)

    # 行业哑变量
    X_dummies = pd.get_dummies(
        group[industry_col], drop_first=True, prefix=industry_col, dtype=float
    )
    if X_dummies.empty:
        print(
            f"[neutralize_industry_marketcap] {dt} 行业只有一个类别，回归仅使用对数市值"
        )

    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    # 权重（一次性计算）
    w_all = np.sqrt(group[float_mv_col].astype("float64").clip(lower=eps)).rename("_w")
    w_mean = w_all.mean()
    if w_mean <= eps:
        w_mean = 1.0
    w_all = (w_all / w_mean).rename("_w")

    # 2) 预分配结果容器（保持每列的原始dtype）
    result = group[factor_list].copy()
    # 用NaN填充（避免保留原始数据）
    result.loc[:, :] = np.nan

    # 3) 按因子回归（复用X_styles和w_all）
    for factor_name in factor_list:
        # 构造回归数据（只包含当前因子 + 风格 + 权重）
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
                    f"[neutralize_industry_marketcap] {dt} {factor_name} dropna 丢弃 {before_drop - len(reg_data)}/{before_drop} ({drop_ratio:.1%})"
                )

        n_params = len(X_styles.columns) + 2  # 截距 + 因子 + 风格
        if len(reg_data) < n_params:
            print(
                f"[neutralize_industry_marketcap] {dt} {factor_name} 样本 {len(reg_data)} 少于所需 {n_params}，置 NaN"
            )
            # result[factor_name] 已经是NaN，无需操作
            continue

        X_reg = sm.add_constant(reg_data[X_styles.columns], has_constant="add")
        Y_reg = reg_data[factor_name]
        w = reg_data["_w"]
        try:
            model = sm.WLS(Y_reg, X_reg, weights=w).fit()
            residual = model.resid.astype(group[factor_name].dtype, copy=False)
            result.loc[residual.index, factor_name] = residual

            # 回归质量诊断
            if model.rsquared < 0.1:
                print(
                    f"[neutralize] {dt} {factor_name} R²={model.rsquared:.3f}，拟合质量较低"
                )
            elif model.rsquared < 0.05:
                print(
                    f"[neutralize] {dt} {factor_name} R²={model.rsquared:.3f}，拟合质量很差"
                )
        except Exception as e:
            print(
                f"[neutralize_industry_marketcap] {dt} {factor_name} 回归失败：{e}，置 NaN"
            )
            # result[factor_name] 已经是NaN，无需操作

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
    【单日截面】因子收益回归（专用版本，内存优化）

    对每个因子factor和每个持仓周期ret_col做截面回归：
        ret = β0 + β1*factor + β2*log(total_mv) + Σβj*industry_j + ε
    WLS权重：sqrt(float_mv)

    Args:
        group: pd.DataFrame 单日截面数据，index为(instrument, datetime)
        factor_list: List[str] 因子列名
        ret_list: List[str] 收益率列名
        total_mv_col: str 总市值列名（将在函数内取对数）
        industry_col: str 行业分类列名（将转换为哑变量）
        float_mv_col: str 流通市值列名（将在函数内取平方根作为权重）
        eps: float 防止权重为0的保护值

    Returns:
        coef_df: 系数时间序列（datetime索引，列名{factor}_{ret}）
        t_df: t值时间序列（datetime索引，列名{factor}_{ret}）
    """
    if not factor_list or not ret_list:
        return pd.DataFrame(), pd.DataFrame()

    dt = (
        group.index.get_level_values("datetime")[0]
        if "datetime" in group.index.names
        else "unknown_dt"
    )

    # 1) 预先构造风格变量（所有回归共享）
    log_mv = np.log(group[total_mv_col].astype("float64").clip(lower=eps))
    X_continuous = pd.DataFrame({"log_mv": log_mv}, index=group.index)

    # 行业哑变量
    X_dummies = pd.get_dummies(
        group[industry_col], drop_first=True, prefix=industry_col, dtype=float
    )
    if X_dummies.empty:
        print(f"[factor_return] {dt} 行业只有一个类别，回归仅使用对数市值")

    X_styles = pd.concat([X_continuous, X_dummies], axis=1)

    # 权重（一次性计算）
    w_all = np.sqrt(group[float_mv_col].astype("float64").clip(lower=eps)).rename("_w")
    w_mean = w_all.mean()
    if w_mean <= eps:
        w_mean = 1.0
    w_all = (w_all / w_mean).rename("_w")

    # 2) 执行回归：因子 × 收益率
    coef_row = {}
    t_row = {}

    for factor_name in factor_list:
        # 构造包含该因子的X（所有ret_col共享）
        base_X = pd.concat([group[[factor_name]].astype("float64"), X_styles], axis=1)

        for ret_col in ret_list:
            if group[ret_col].dropna().empty:
                continue

            # 构造回归数据
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
                        f"[factor_return] {dt} {factor_name}-{ret_col} dropna 丢弃 {before_drop - len(reg_data)}/{before_drop} ({drop_ratio:.1%})"
                    )

            n_params = base_X.shape[1] + 1  # 截距 + 因子 + 风格
            col_name = f"{factor_name}_{ret_col}"
            if len(reg_data) < n_params:
                print(
                    f"[factor_return] {dt} 跳过 {factor_name} 对 {ret_col} 的回归：样本 {len(reg_data)} < 所需 {n_params}，置 NaN"
                )
                coef_row[col_name] = np.nan
                t_row[col_name] = np.nan
                continue

            # 回归
            X_reg = sm.add_constant(reg_data.drop(columns=[ret_col, "_w"]))
            Y_reg = reg_data[ret_col]
            w_reg = reg_data["_w"]

            try:
                model = sm.WLS(Y_reg, X_reg, weights=w_reg).fit()
                coef_row[col_name] = model.params.get(factor_name, np.nan)
                t_row[col_name] = model.tvalues.get(factor_name, np.nan)

                # 回归质量诊断
                if model.rsquared < 0.1:
                    print(
                        f"[factor_return] {dt} {factor_name}-{ret_col} R²={model.rsquared:.3f}，拟合质量较低"
                    )
            except Exception as e:
                print(
                    f"[factor_return] {dt} {factor_name} 对 {ret_col} 的回归失败（{e}），置 NaN"
                )
                coef_row[col_name] = np.nan
                t_row[col_name] = np.nan

    # 3) 构造结果DataFrame
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
        T = len(s)
        t_stat = mean / std * np.sqrt(T) if std != 0 else np.nan
        icir = mean / std if std != 0 else np.nan
        win_rate = (s > 0).sum() / len(s)
        gt002_rate = (s.abs() > 0.02).sum() / len(s)
        return pd.Series(
            {
                "IC序列均值": mean,
                "IC序列标准差": std,
                "IR比率": icir,
                "IC t统计量": t_stat,
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
    计算预测得分的自相关序列及摘要（多因子）。

    返回：
        ac_df: pd.DataFrame，列为因子名，对应自相关序列
        summary_df: pd.DataFrame，自相关汇总指标
    """

    def _summary(series: pd.Series) -> pd.Series:
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
    基于 qlib 的 Top/Bottom turnover 逻辑（多因子）。

    返回：
        daily_df: pd.DataFrame，多层列为因子名下的 top/bottom 换手率
        summary_df: pd.DataFrame，按因子汇总换手率均值/波动
    """

    def _compute(score_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pred = group[[score_col]].copy()
        pred["score_last"] = pred.groupby(level="instrument", group_keys=False)[
            score_col
        ].shift(lag)

        def _top_turnover(x: pd.DataFrame) -> float:
            k = len(x) // N
            if k == 0:
                return np.nan
            top_now = x.nlargest(k, columns=score_col).index
            top_prev = x.nlargest(k, columns="score_last").index
            return 1 - top_now.isin(top_prev).sum() / k

        def _bottom_turnover(x: pd.DataFrame) -> float:
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
    生成性能图并保存为 html 文件（支持多因子×多周期）。

    Args:
        merged_df: 合并后的数据，包含因子列和收益率列
        factor_list: 因子列名列表
        ret_list: 收益率列名列表
        output_dir: 输出目录
        graph_names: 要生成的图表类型列表，默认为 ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
    """
    names = graph_names or ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 为每个因子×收益率组合生成图表
    for factor_name in factor_list:
        for ret_col in ret_list:
            # 构造单因子×单收益率的数据
            # pred_label格式要求：包含因子列和收益率列的DataFrame
            pred_label = merged_df[[factor_name, ret_col]].copy()
            pred_label.columns = ["score", "label"]  # qlib要求的列名

            # 创建子目录：factor_ret/
            subdir = out_dir / f"{factor_name}_{ret_col}"
            subdir.mkdir(parents=True, exist_ok=True)

            # 生成性能图表
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
                        f"[save_performance_graphs] {factor_name}-{ret_col} 生成{name}图表失败: {e}"
                    )

            # 生成score_ic图表
            try:
                figs = score_ic_graph(pred_label, show_notebook=False)
                for i, fig in enumerate(figs, start=1):
                    name = "score_ic.html" if len(figs) == 1 else f"score_ic_{i}.html"
                    fig.write_html(str(subdir / name))
            except Exception as e:
                print(
                    f"[save_performance_graphs] {factor_name}-{ret_col} 生成score_ic图表失败: {e}"
                )

    print(f"  ✓ 性能图表已保存到: {out_dir}")
