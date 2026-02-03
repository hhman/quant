"""GP配置模块，定义数据配置和GP算法配置的数据类。"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


# ==================== 默认配置 ====================

DEFAULT_FEATURES = [
    "$close",
    "$open",
    "$high",
    "$low",
    "$volume",
    "$amount",
]

DEFAULT_TARGET = "Ref($close, -1)/$close - 1"


# ==================== 数据配置 ====================


@dataclass(frozen=True)
class DataConfig:
    """数据配置类。

    Attributes:
        features: 特征字段列表
        target: 目标字段表达式
    """

    features: List[str] = field(default_factory=lambda: DEFAULT_FEATURES)
    target: str = DEFAULT_TARGET


# ==================== GP算法配置 ====================


@dataclass(frozen=True)
class GPConfig:
    """GP遗传算法配置类。

    Attributes:
        population_size: 种群大小
        generations: 迭代代数
        hall_of_fame: 保留的精英个体数量
        n_components: 最终输出的组件数量
        tournament_size: 锦标赛选择规模
        p_crossover: 交叉概率
        p_subtree_mutation: 子树变异概率
        p_hoist_mutation: 提升变异概率
        p_point_mutation: 点变异概率
        p_point_replace: 点替换概率
        const_range: 常量范围
        init_depth: 初始树深度范围(min, max)
        init_method: 初始化方法（half and half/grow/full）
        max_samples: 最大采样比例
        n_jobs: 并行任务数
        verbose: 详细输出级别
        random_state: 随机种子
        metric: 适应度函数名称（从注册表获取 stopping_criteria）
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
    metric: str = "rank_ic"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于传递给SymbolicTransformer。

        Returns:
            配置字典
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
        }


# ==================== 配置工厂函数 ====================


def get_fast_test_config() -> GPConfig:
    """获取快速测试配置（用于调试和测试）。

    Returns:
        快速测试配置对象
    """
    return GPConfig(
        population_size=20,
        generations=2,
        hall_of_fame=5,
        n_components=3,
        tournament_size=3,
    )


def get_production_config() -> GPConfig:
    """获取生产级配置（用于正式因子挖掘）。

    最大化探索能力，适合最终因子挖掘，计算时间较长。

    Returns:
        生产级配置对象
    """
    return GPConfig(
        population_size=500,
        generations=20,
        hall_of_fame=20,
        n_components=10,
        tournament_size=10,
    )


# ==================== 导出列表 ====================

__all__ = [
    # 数据类
    "DataConfig",
    "GPConfig",
    # 默认值
    "DEFAULT_FEATURES",
    "DEFAULT_TARGET",
    # 工厂函数
    "get_fast_test_config",
    "get_production_config",
]
