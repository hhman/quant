"""
配置管理模块

使用 dataclass 管理系统配置参数，提供类型安全和不可变性保证。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


# ==================== 默认配置常量 ====================

# 默认特征字段
DEFAULT_FEATURES = [
    "$close",
    "$open",
    "$high",
    "$low",
    "$volume",
    "$amount",
    "$vwap",
]

# 默认目标标签
DEFAULT_TARGET = "Ref($close, -1)/$close - 1"


# ==================== 数据配置 ====================


@dataclass(frozen=True)
class DataConfig:
    """
    数据加载与清洗配置

    Attributes:
        features: 特征字段列表
        target: 目标标签表达式
        fillna_price: 价格类特征填充策略 ("ffill", "zero", "mean", "drop")
        fillna_volume: 成交量类特征填充策略 ("ffill", "zero", "mean", "drop")
        price_columns: 价格类字段列表（用于识别）
        volume_columns: 成交量类字段列表（用于识别）
    """

    features: List[str] = field(default_factory=lambda: DEFAULT_FEATURES)
    target: str = DEFAULT_TARGET
    fillna_price: str = "ffill"  # 价格类：前向填充
    fillna_volume: str = "zero"  # 成交量类：填充 0

    # 特征类型分类（用于识别）
    price_columns: List[str] = field(
        default_factory=lambda: ["$close", "$open", "$high", "$low", "$vwap"]
    )
    volume_columns: List[str] = field(default_factory=lambda: ["$volume", "$amount"])


# ==================== 遗传算法配置 ====================


@dataclass(frozen=True)
class GPConfig:
    """
    Gplearn 遗传算法参数配置

    Attributes:
        population_size: 种群大小
        generations: 迭代代数
        hall_of_fame: 精英保留数量
        n_components: 最终输出的因子数量
        tournament_size: 锦标赛选择大小
        p_crossover: 交叉概率
        p_subtree_mutation: 子树变异概率
        p_hoist_mutation: 提升（hoist）变异概率
        p_point_mutation: 点变异概率
        p_point_replace: 点替换概率
        const_range: 常数范围
        init_depth: 初始树深度范围 (min, max)
        init_method: 初始化方法 ("half and half", "grow", "full")
        max_samples: 最大采样比例
        n_jobs: 并行任务数（支持多线程）
        verbose: 输出详细程度
        random_state: 随机种子
    """

    population_size: int = 20
    generations: int = 2
    hall_of_fame: int = 5
    n_components: int = 3
    tournament_size: int = 3
    p_crossover: float = 0.7
    p_subtree_mutation: float = 0.1
    p_hoist_mutation: float = 0.05
    p_point_mutation: float = 0.1
    p_point_replace: float = 0.05
    const_range: tuple = (-1.0, 1.0)
    init_depth: tuple = (2, 4)
    init_method: str = "half and half"
    max_samples: float = 1.0
    n_jobs: int = 4
    verbose: int = 1
    random_state: int = None
    stopping_criteria: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（用于传递给 SymbolicTransformer）

        Returns:
            参数字典
        """
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "hall_of_fame": self.hall_of_fame,
            "n_components": self.n_components,
            "tournament_size": self.tournament_size,
            "p_crossover": self.p_crossover,
            "p_subtree_mutation": self.p_subtree_mutation,
            "p_hoist_mutation": self.p_hoist_mutation,
            "p_point_mutation": self.p_point_mutation,
            "p_point_replace": self.p_point_replace,
            "const_range": self.const_range,
            "init_depth": self.init_depth,
            "init_method": self.init_method,
            "max_samples": self.max_samples,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "stopping_criteria": self.stopping_criteria,
        }


# ==================== 预设配置 ====================


def get_default_data_config() -> DataConfig:
    """获取默认数据配置"""
    return DataConfig()


def get_default_gp_config() -> GPConfig:
    """获取默认 GP 配置"""
    return GPConfig()


def get_fast_test_config() -> GPConfig:
    """
    获取快速测试配置（用于开发和调试）

    减小种群大小和迭代次数，加快训练速度
    """
    return GPConfig(
        population_size=20,
        generations=2,
        hall_of_fame=5,
        n_components=2,
        tournament_size=3,
    )


# ==================== 导出 ====================

__all__ = [
    # 配置类
    "DataConfig",
    "GPConfig",
    # 默认值
    "DEFAULT_FEATURES",
    "DEFAULT_TARGET",
    # 工厂函数
    "get_default_data_config",
    "get_default_gp_config",
    "get_fast_test_config",
]
