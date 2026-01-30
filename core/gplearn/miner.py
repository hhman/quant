"""
GP 挖掘模块

封装 SymbolicTransformer，提供遗传算法因子挖掘功能。
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
    raise ImportError("请先安装 gplearn: pip install gplearn")


class FactorMiner:
    """
    遗传算法因子挖掘器

    职责：
    - 接收 MultiIndex DataFrame，自动展平后训练
    - 训练完成后自动导出表达式
    """

    def __init__(
        self,
        features: List[str],
        target: str,
        gp_config: GPConfig = None,
        random_state: int = None,
    ):
        """
        初始化挖掘器

        Args:
            features: 特征名称列表
            target: 目标表达式（用于记录）
            gp_config: GP 配置对象
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
        执行挖掘流程：训练 + 导出表达式

        Args:
            features_df: (n_samples, n_features) MultiIndex DataFrame
            target_df: (n_samples, 1) MultiIndex DataFrame

        Returns:
            因子表达式字符串列表
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
        准备 GP 训练数据

        Args:
            features_df: (n_samples, n_features) MultiIndex DataFrame
            target_df: (n_samples, 1) MultiIndex DataFrame

        Returns:
            (X, y, index, boundaries)
        """
        common_index = features_df.index.intersection(target_df.index)

        if len(common_index) == 0:
            raise ValueError("特征和标签无共同 index！请检查数据源。")

        if len(common_index) < len(features_df.index):
            print(f"  标签数据缺失 {len(features_df.index) - len(common_index)} 个样本")

        features_filtered = features_df.loc[common_index]
        target_filtered = target_df.loc[common_index]

        target_col = target_filtered.columns[0]
        valid_mask = ~target_filtered[target_col].isna()

        features_clean = features_filtered[valid_mask].fillna(0)
        target_clean = target_filtered[valid_mask]

        if len(target_clean) == 0:
            raise ValueError("清洗后无有效数据！请检查数据源。")

        dropped = len(valid_mask) - valid_mask.sum()
        if dropped > 0:
            print(
                f"  删除了 {dropped} 个目标为 NaN 的样本（剩余 {len(target_clean)} 个）"
            )

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
        """训练模型"""
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
        准备训练环境

        Args:
            X: 特征数组
            y: 标签数组
            index: MultiIndex
            boundaries: 边界索引列表

        Returns:
            SymbolicTransformer 实例
        """
        with global_state(index, boundaries):
            function_set = get_all_operators()
            return self._create_transformer(function_set)

    def _create_transformer(
        self,
        function_set: List,
    ) -> SymbolicTransformer:
        """
        创建 SymbolicTransformer 实例

        Args:
            function_set: 算子函数列表

        Returns:
            SymbolicTransformer 实例
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
        执行训练

        Args:
            transformer: SymbolicTransformer 实例
            X: 特征数组
            y: 标签数组
        """
        params = transformer.get_params()
        print(
            f"  训练参数: population={params['population_size']}, "
            f"generations={params['generations']}, "
            f"n_jobs={params['n_jobs']}"
        )
        transformer.fit(X, y)

    def _export(self) -> List[str]:
        """导出因子表达式"""
        if self._transformer is None:
            raise RuntimeError("模型未训练")

        expressions = []
        for program_list in self._transformer._programs:
            for program in program_list:
                expressions.append(str(program))

        return expressions


__all__ = ["FactorMiner"]
