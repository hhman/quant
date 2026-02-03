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
        index = features_df.index

        X = flatten_features(features_df)
        y = flatten_target(target_df)
        boundaries = calc_boundaries(index)

        return X, y, index, boundaries

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        index: pd.MultiIndex,
        boundaries: List[int],
    ) -> None:
        """训练GP模型。

        Args:
            X: 特征数组
            y: 目标数组
            index: MultiIndex
            boundaries: 边界索引列表
        """
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
        """创建 SymbolicTransformer。

        Args:
            function_set: 算子集合

        Returns:
            SymbolicTransformer 实例
        """
        from gplearn.fitness import make_fitness
        from .common.registry import _get_fitness_raw, _get_fitness_meta

        params = self.gp_config.to_dict()
        params["random_state"] = self.random_state

        metric_name = params.pop("metric", "rank_ic")

        meta = _get_fitness_meta(metric_name)
        stopping_criteria = meta.get("stopping_criteria", 0.0)
        params["stopping_criteria"] = stopping_criteria

        fitness_func = _get_fitness_raw(metric_name)
        custom_metric = make_fitness(function=fitness_func, greater_is_better=True)
        params["metric"] = custom_metric

        return SymbolicTransformer(
            function_set=function_set, feature_names=self.features, **params
        )

    def _execute_train(
        self,
        transformer: SymbolicTransformer,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """执行GP训练。

        Args:
            transformer: SymbolicTransformer
            X: 特征数组
            y: 目标数组
        """
        transformer.fit(X, y)

    def _export(self) -> List[str]:
        """导出GP生成的表达式。

        Returns:
            表达式字符串列表

        Raises:
            RuntimeError: 当模型未训练时
        """
        if self._transformer is None:
            raise RuntimeError("模型尚未训练，请先调用run方法")

        expressions = []
        for program_list in self._transformer._programs:
            for program in program_list:
                expressions.append(str(program))

        return expressions


__all__ = ["FactorMiner"]
