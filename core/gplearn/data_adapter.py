"""
Gplearn 数据适配器

负责将 Qlib 多维面板数据展平为 Gplearn 可接受的二维格式，
并在适应度计算时还原为三维数据。

核心机制：
1. 数据展平：三维面板 → 二维扁平数据
2. 边界记录：记录每只股票的起始位置
3. 数据还原：二维扁平 → 三维面板（unstack）

使用示例：
    >>> adapter = GplearnDataAdapter(
    ...     base_features=['$close', '$volume', '$total_mv'],
    ...     target_col='ret_1d'
    ... )
    >>> X_flat, y_flat, index = adapter.prepare_training_data(panel_data)
    >>> boundary_indices = adapter.get_boundary_indices(index)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from .config import DEFAULT_WINDOW_SIZE
from .exceptions import GplearnDataError


class GplearnDataAdapter:
    """
    Gplearn 数据适配器

    职责：
    - 将 Qlib MultiIndex DataFrame 展平为二维数组
    - 记录每只股票的边界索引
    - 支持数据还原（unstack）
    """

    def __init__(
        self,
        base_features: List[str],
        target_col: str = "ret_1d",
        window_size: int = DEFAULT_WINDOW_SIZE,
    ):
        """
        初始化数据适配器

        Args:
            base_features: 基础特征列表（如 ['$close', '$volume']）
            target_col: 目标列名（如 'ret_1d'）
            window_size: 时间序列窗口大小
        """
        self.base_features = base_features
        self.target_col = target_col
        self.window_size = window_size

        # 边界索引（每只股票的起始位置）
        self.boundary_indices = []

    def prepare_training_data(
        self,
        panel_data: pd.DataFrame,
        dropna: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
        """
        准备训练数据

        将三维面板数据展平为二维数组，并记录边界索引

        Args:
            panel_data: MultiIndex DataFrame (instrument, datetime)
            dropna: 是否删除缺失值

        Returns:
            X_flat: 特征矩阵 (n_samples, n_features)
            y_flat: 目标向量 (n_samples,)
            index: MultiIndex (用于还原)

        Raises:
            GplearnDataError: 数据准备失败
        """
        # 验证输入数据
        if not isinstance(panel_data.index, pd.MultiIndex):
            raise GplearnDataError("输入数据必须是 MultiIndex DataFrame")

        if panel_data.index.names != ["instrument", "datetime"]:
            raise GplearnDataError(
                f"MultiIndex 级别顺序必须是 ['instrument', 'datetime']，"
                f"实际为 {panel_data.index.names}"
            )

        # 验证特征列存在
        missing_features = set(self.base_features) - set(panel_data.columns)
        if missing_features:
            raise GplearnDataError(f"缺少特征列: {missing_features}")

        if self.target_col not in panel_data.columns:
            raise GplearnDataError(f"缺少目标列: {self.target_col}")

        # 提取数据
        try:
            feature_data = panel_data[self.base_features]
            target_data = panel_data[self.target_col]
            index = panel_data.index
        except Exception as e:
            raise GplearnDataError(f"数据提取失败: {e}")

        # 处理缺失值
        if dropna:
            valid_mask = ~(feature_data.isna().any(axis=1) | target_data.isna())
            feature_data = feature_data[valid_mask]
            target_data = target_data[valid_mask]
            index = index[valid_mask]

            if len(index) == 0:
                raise GplearnDataError("删除缺失值后无数据")

        # 展平为二维数组
        X_flat = feature_data.values
        y_flat = target_data.values

        # 记录边界索引
        self.boundary_indices = self._find_boundary_indices(index)

        return X_flat, y_flat, index

    def _find_boundary_indices(self, index: pd.MultiIndex) -> List[int]:
        """
        找到每只股票的起始位置（边界索引）

        Args:
            index: MultiIndex (instrument, datetime)

        Returns:
            边界索引列表（每只股票第一个样本的位置）
        """
        if len(index) == 0:
            return []

        # 获取标的代码列表
        instruments = index.get_level_values(0)

        # 找到标的变更的位置
        boundary_indices = []
        for i in range(1, len(instruments)):
            if instruments[i] != instruments[i - 1]:
                boundary_indices.append(i)

        return boundary_indices

    def get_boundary_indices(self) -> List[int]:
        """
        获取边界索引

        Returns:
            边界索引列表
        """
        return self.boundary_indices.copy()

    def restore_to_panel(
        self,
        flat_data: np.ndarray,
        index: pd.MultiIndex,
    ) -> pd.DataFrame:
        """
        将扁平数据还原为三维面板

        Args:
            flat_data: 扁平数据 (n_samples,)
            index: MultiIndex (instrument, datetime)

        Returns:
            面板数据 (n_dates, n_stocks)
        """
        # 构造 DataFrame
        df = pd.DataFrame(flat_data, index=index)

        # 还原为面板：unstack('instrument') -> index=日期, columns=股票
        panel = df.unstack(level=0)

        return panel

    def apply_boundary_deletion(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        应用边界删除（删除每只股票前 N 天数据）

        目的：去除时间序列算子产生的跨股票污染

        Args:
            data: 扁平数据 (n_samples,)
            window_size: 窗口大小（默认使用 self.window_size）

        Returns:
            删除边界污染后的数据
        """
        if window_size is None:
            window_size = self.window_size

        if not self.boundary_indices:
            return data

        # 复制数据（避免修改原数据）
        # 转换为浮点类型以支持 NaN
        data_clean = data.copy().astype(float)

        # 删除每只股票的前 window 天数据
        for boundary_idx in self.boundary_indices:
            try:
                # 删除从边界开始的 window 个样本
                end_idx = min(boundary_idx + window_size, len(data_clean))
                data_clean[boundary_idx:end_idx] = np.nan
            except IndexError:
                # 边界超出数据范围，跳过
                continue

        return data_clean
