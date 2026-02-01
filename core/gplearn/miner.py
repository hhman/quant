"""
GP

 SymbolicTransformer
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from .common.registry import get_all_operators
from .common.state import global_state
from .common.panel import flatten_features, flatten_target, calc_boundaries
from .config import GPConfig

try:
    from gplearn.genetic import SymbolicTransformer
except ImportError:
    raise ImportError(" gplearn: pip install gplearn")


class FactorMiner:
    """



    -  MultiIndex DataFrame
    -
    """

    def __init__(
        self,
        features: List[str],
        target: str,
        gp_config: GPConfig = None,
        random_state: int = None,
    ) -> None:
        """初始化因子挖掘器。

        Args:
            features: 特征列表
            target: 目标变量名
            gp_config: GP配置对象
            random_state: 随机种子
        """
        self.features = features
        self.target = target
        self.gp_config = gp_config or GPConfig()
        self.random_state = random_state
        self._transformer = None

    def run(
        self,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> List[str]:
        """
         +

        Args:
            features_df: (n_samples, n_features) MultiIndex DataFrame
            target_df: (n_samples, 1) MultiIndex DataFrame

        Returns:

        """
        X, y, index, boundaries = self._prepare_data(features_df, target_df)
        self._train(X, y, index, boundaries)
        return self._export()

    def _prepare_data(
        self,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, pd.MultiIndex, List[int]]:
        """
         GP

        Args:
            features_df: (n_samples, n_features) MultiIndex DataFrame
            target_df: (n_samples, 1) MultiIndex DataFrame

        Returns:
            (X, y, index, boundaries)
        """
        common_index = features_df.index.intersection(target_df.index)

        if len(common_index) == 0:
            raise ValueError(" index")

        if len(common_index) < len(features_df.index):
            print(f"   {len(features_df.index) - len(common_index)} ")

        features_filtered = features_df.loc[common_index]
        target_filtered = target_df.loc[common_index]

        target_col = target_filtered.columns[0]
        valid_mask = ~target_filtered[target_col].isna()

        features_clean = features_filtered[valid_mask].fillna(0)
        target_clean = target_filtered[valid_mask]

        if len(target_clean) == 0:
            raise ValueError("")

        dropped = len(valid_mask) - valid_mask.sum()
        if dropped > 0:
            print(f"   {dropped}  NaN  {len(target_clean)} ")

        index = features_clean.index

        X = flatten_features(features_clean)
        y = flatten_target(target_clean)
        boundaries = calc_boundaries(index)

        return X, y, index, boundaries

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        index: pd.MultiIndex,
        boundaries: List[int],
    ) -> None:
        """"""
        transformer = self._setup_training_env(X, y, index, boundaries)
        self._execute_train(transformer, X, y)
        self._transformer = transformer

    def _setup_training_env(
        self,
        X: np.ndarray,
        y: np.ndarray,
        index: pd.MultiIndex,
        boundaries: List[int],
    ) -> SymbolicTransformer:
        """


        Args:
            X:
            y:
            index: MultiIndex
            boundaries:

        Returns:
            SymbolicTransformer
        """
        with global_state(index, boundaries):
            function_set = get_all_operators()
            return self._create_transformer(function_set)

    def _create_transformer(
        self,
        function_set: List,
    ) -> SymbolicTransformer:
        """
         SymbolicTransformer

        Args:
            function_set:

        Returns:
            SymbolicTransformer
        """
        params = self.gp_config.to_dict()
        params["random_state"] = self.random_state

        return SymbolicTransformer(
            function_set=function_set, feature_names=self.features, **params
        )

    def _execute_train(
        self,
        transformer: SymbolicTransformer,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """


        Args:
            transformer: SymbolicTransformer
            X:
            y:
        """
        params = transformer.get_params()
        print(
            f"  : population={params['population_size']}, "
            f"generations={params['generations']}, "
            f"n_jobs={params['n_jobs']}"
        )
        transformer.fit(X, y)

    def _export(self) -> List[str]:
        """"""
        if self._transformer is None:
            raise RuntimeError("")

        expressions = []
        for program_list in self._transformer._programs:
            for program in program_list:
                expressions.append(str(program))

        return expressions


__all__ = ["FactorMiner"]
