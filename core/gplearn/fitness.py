"""
Gplearn 适应度函数库

本模块提供三维面板数据的适应度计算函数，支持在遗传算法中
直接计算 Rank IC、加权 IC 和复合适应度。

核心机制：
1. 接收扁平化预测结果
2. 应用边界删除
3. 还原为三维面板
4. 计算多日期的横截面相关系数

使用示例：
    >>> fitness = RankICFitness(window_size=10)
    >>> score = fitness.compute(y_true, y_pred, index, boundary_indices)
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from abc import ABC, abstractmethod

from .exceptions import FitnessCalculationError


class BaseFitness(ABC):
    """
    适应度函数基类

    所有适应度函数必须继承此类并实现 compute 方法。
    """

    def __init__(
        self,
        window_size: int = 10,
        min_samples: int = 100,
    ):
        """
        初始化适应度函数

        Args:
            window_size: 边界删除窗口大小
            min_samples: 计算适应度所需的最小样本数
        """
        self.window_size = window_size
        self.min_samples = min_samples

    @abstractmethod
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        index: pd.MultiIndex,
        boundary_indices: List[int],
    ) -> float:
        """
        计算适应度分数

        Args:
            y_true: 真实目标值（扁平化）
            y_pred: 预测值（扁平化）
            index: MultiIndex (instrument, datetime)
            boundary_indices: 边界索引列表

        Returns:
            适应度分数

        Raises:
            FitnessCalculationError: 计算失败
        """
        pass

    def _apply_boundary_deletion(
        self,
        data: np.ndarray,
        boundary_indices: List[int],
    ) -> np.ndarray:
        """
        应用边界删除

        Args:
            data: 扁平数据
            boundary_indices: 边界索引列表

        Returns:
            删除边界污染后的数据
        """
        if not boundary_indices:
            return data

        data_clean = data.copy()

        for boundary_idx in boundary_indices:
            try:
                end_idx = min(boundary_idx + self.window_size, len(data_clean))
                data_clean[boundary_idx:end_idx] = np.nan
            except IndexError:
                continue

        return data_clean

    def _restore_to_panel(
        self,
        flat_data: np.ndarray,
        index: pd.MultiIndex,
    ) -> pd.DataFrame:
        """
        将扁平数据还原为三维面板

        Args:
            flat_data: 扁平数据
            index: MultiIndex

        Returns:
            面板数据 (n_dates, n_stocks)
        """
        df = pd.DataFrame(flat_data, index=index, columns=["value"])
        panel = df.unstack(level=0)["value"]
        return panel


class RankICFitness(BaseFitness):
    """
    Rank IC 适应度函数

    计算预测值与真实值的 Spearman 相关系数的均值。

    核心优势：
    - 对异常值鲁棒
    - 适合因子挖掘场景
    - 反映排序能力而非绝对值

    计算公式：
        Rank IC = mean(corrwith(y_pred, y_true, method='spearman'))
    """

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        index: pd.MultiIndex,
        boundary_indices: List[int],
    ) -> float:
        """
        计算 Rank IC

        Args:
            y_true: 真实目标值
            y_pred: 预测值
            index: MultiIndex
            boundary_indices: 边界索引

        Returns:
            Rank IC 均值

        Raises:
            FitnessCalculationError: 计算失败
        """
        # 1. 应用边界删除
        y_pred_clean = self._apply_boundary_deletion(y_pred, boundary_indices)

        # 2. 构造 DataFrame
        df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred_clean,
            },
            index=index,
        )

        # 3. 还原为面板
        y_true_panel = df["y_true"].unstack(level=0)
        y_pred_panel = df["y_pred"].unstack(level=0)

        # 4. 删除全 NaN 列
        valid_mask = ~y_pred_panel.isna().all(axis=0)
        y_true_panel = y_true_panel.loc[:, valid_mask]
        y_pred_panel = y_pred_panel.loc[:, valid_mask]

        # 5. 检查样本量
        if y_true_panel.shape[1] < self.min_samples:
            raise FitnessCalculationError(
                f"样本量不足：{y_true_panel.shape[1]} < {self.min_samples}"
            )

        # 6. 计算横截面 Rank IC
        try:
            ic_series = y_pred_panel.corrwith(y_true_panel, axis=1, method="spearman")
            ic_mean = ic_series.mean()

            # 处理 NaN
            if pd.isna(ic_mean):
                return 0.0

            return ic_mean

        except Exception as e:
            raise FitnessCalculationError(f"Rank IC 计算失败: {e}")


class WeightedICFitness(BaseFitness):
    """
    加权 IC 适应度函数

    计算预测值与真实值的 Pearson 相关系数的均值，
    可选地使用市值加权。

    适用场景：
    - 关注绝对值相关性
    - 需要市值加权的大市值股偏好

    计算公式：
        Weighted IC = mean(weight * corrwith(y_pred, y_true, method='pearson'))
    """

    def __init__(
        self,
        window_size: int = 10,
        min_samples: int = 100,
        use_weight: bool = False,
        weight_col: str = "$total_mv",
    ):
        """
        初始化加权 IC 适应度函数

        Args:
            window_size: 边界删除窗口大小
            min_samples: 最小样本数
            use_weight: 是否使用市值加权
            weight_col: 权重列名
        """
        super().__init__(window_size, min_samples)
        self.use_weight = use_weight
        self.weight_col = weight_col

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        index: pd.MultiIndex,
        boundary_indices: List[int],
        weight_data: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算加权 IC

        Args:
            y_true: 真实目标值
            y_pred: 预测值
            index: MultiIndex
            boundary_indices: 边界索引
            weight_data: 权重数据（扁平化）

        Returns:
            加权 IC 均值

        Raises:
            FitnessCalculationError: 计算失败
        """
        # 1. 应用边界删除
        y_pred_clean = self._apply_boundary_deletion(y_pred, boundary_indices)

        # 2. 构造 DataFrame
        df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred_clean,
            },
            index=index,
        )

        if self.use_weight and weight_data is not None:
            df["weight"] = weight_data

        # 3. 还原为面板
        y_true_panel = df["y_true"].unstack(level=0)
        y_pred_panel = df["y_pred"].unstack(level=0)

        # 4. 删除全 NaN 列
        valid_mask = ~y_pred_panel.isna().all(axis=0)
        y_true_panel = y_true_panel.loc[:, valid_mask]
        y_pred_panel = y_pred_panel.loc[:, valid_mask]

        # 5. 检查样本量
        if y_true_panel.shape[1] < self.min_samples:
            raise FitnessCalculationError(
                f"样本量不足：{y_true_panel.shape[1]} < {self.min_samples}"
            )

        # 6. 计算横截面 IC
        try:
            ic_series = y_pred_panel.corrwith(y_true_panel, axis=1, method="pearson")

            # 7. 应用权重
            if self.use_weight and "weight" in df.columns:
                weight_panel = df["weight"].unstack(level=0).loc[:, valid_mask]
                # 标准化权重
                weight_normalized = weight_panel.div(weight_panel.sum(axis=1), axis=0)
                # 加权平均
                ic_mean = (ic_series * weight_normalized.mean(axis=1)).sum()
            else:
                ic_mean = ic_series.mean()

            # 处理 NaN
            if pd.isna(ic_mean):
                return 0.0

            return ic_mean

        except Exception as e:
            raise FitnessCalculationError(f"加权 IC 计算失败: {e}")


class CompositeICFitness(BaseFitness):
    """
    复合 IC 适应度函数

    综合考虑 Rank IC 和加权 IC，提供更稳健的适应度评估。

    组合方式：
        Composite IC = w1 * Rank IC + w2 * Weighted IC

    优势：
    - 兼顾排序能力和绝对值相关性
    - 降低单一指标的风险
    - 可调节权重适应不同场景
    """

    def __init__(
        self,
        window_size: int = 10,
        min_samples: int = 100,
        rank_weight: float = 0.5,
        weighted_weight: float = 0.5,
    ):
        """
        初始化复合 IC 适应度函数

        Args:
            window_size: 边界删除窗口大小
            min_samples: 最小样本数
            rank_weight: Rank IC 权重
            weighted_weight: 加权 IC 权重
        """
        super().__init__(window_size, min_samples)
        self.rank_weight = rank_weight
        self.weighted_weight = weighted_weight

        # 初始化子适应度函数
        self.rank_ic = RankICFitness(window_size, min_samples)
        self.weighted_ic = WeightedICFitness(window_size, min_samples)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        index: pd.MultiIndex,
        boundary_indices: List[int],
        weight_data: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算复合 IC

        Args:
            y_true: 真实目标值
            y_pred: 预测值
            index: MultiIndex
            boundary_indices: 边界索引
            weight_data: 权重数据（扁平化）

        Returns:
            复合 IC 均值

        Raises:
            FitnessCalculationError: 计算失败
        """
        try:
            # 计算 Rank IC
            rank_ic_score = self.rank_ic.compute(
                y_true, y_pred, index, boundary_indices
            )

            # 计算加权 IC
            weighted_ic_score = self.weighted_ic.compute(
                y_true, y_pred, index, boundary_indices, weight_data
            )

            # 复合加权
            composite_score = (
                self.rank_weight * rank_ic_score
                + self.weighted_weight * weighted_ic_score
            )

            return composite_score

        except Exception as e:
            raise FitnessCalculationError(f"复合 IC 计算失败: {e}")
