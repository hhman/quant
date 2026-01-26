"""
Gplearn 适应度函数库

提供基于面板数据的适应度计算函数。
"""

import pandas as pd

from .common.registry import register_fitness, get_fitness, list_registered_fitness
from .common.decorators import with_panel_convert


# ==================== 适应度函数 ====================


@register_fitness(name="rank_ic")
@with_panel_convert(min_samples=100)
def rank_ic_fitness(
    y_true_panel: pd.DataFrame,
    y_pred_panel: pd.DataFrame,
) -> float:
    """
    Rank IC 适应度函数

    计算预测值与真实值的 Spearman 相关系数的均值。
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
