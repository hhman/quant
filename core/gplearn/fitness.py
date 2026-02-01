"""
Gplearn


"""

import pandas as pd

from .common.registry import register_fitness, get_fitness, list_registered_fitness
from .common.decorators import with_panel_convert


# ====================  ====================


@register_fitness(name="rank_ic")
@with_panel_convert(min_samples=100)
def rank_ic_fitness(
    y_true_panel: pd.DataFrame,
    y_pred_panel: pd.DataFrame,
) -> float:
    """
    Rank IC

     Spearman
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
