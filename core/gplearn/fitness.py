"""GP 适应度函数模块。

提供因子挖掘所需的适应度函数，使用 Rank IC 作为评价指标。
"""

import pandas as pd

from .common.registry import register_fitness, get_fitness, list_registered_fitness
from .common.decorators import with_panel_convert


@register_fitness(name="rank_ic", stopping_criteria=0.03)
@with_panel_convert()
def rank_ic_fitness(
    y_true_panel: pd.DataFrame,
    y_pred_panel: pd.DataFrame,
) -> float:
    """Rank IC 适应度函数。

    Args:
        y_true_panel: 真实值面板，形状为 (n_dates, n_instruments)
        y_pred_panel: 预测值面板，形状为 (n_dates, n_instruments)

    Returns:
        加权平均 Rank IC，范围 [-1, 1]
    """
    ic_series = y_pred_panel.corrwith(y_true_panel, axis=1, method="spearman")
    n_samples_per_date = y_pred_panel.notna().sum(axis=1)
    ic_mean = (ic_series * n_samples_per_date).sum() / n_samples_per_date.sum()
    return ic_mean


__all__ = [
    "register_fitness",
    "get_fitness",
    "list_registered_fitness",
]
