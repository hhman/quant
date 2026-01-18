"""
Gplearn 因子挖掘器

本模块是因子挖掘系统的核心类，整合数据适配、算子库、适应度函数
和表达式转换器，实现端到端的因子挖掘功能。

核心流程：
1. 数据准备：加载 Qlib 数据，展平为二维数组
2. 初始化种群：使用自定义算子库构建初始种群
3. 遗传进化：调用 Gplearn 进行进化训练
4. 表达式转换：将最优个体转换为 Qlib 表达式
5. 结果输出：返回因子表达式和适应度分数

使用示例：
    >>> from core.gplearn import GplearnFactorMiner
    >>> miner = GplearnFactorMiner(
    ...     market='csi300',
    ...     start_date='2023-01-01',
    ...     end_date='2024-12-31',
    ...     base_features=['$close', '$volume', '$total_mv']
    ... )
    >>> formulas = miner.mine_factors()
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

try:
    from gplearn.genetic import SymbolicRegressor
    from gplearn.fitness import _Fitness
except ImportError:
    raise ImportError("请先安装 gplearn: pip install gplearn")

from .data_adapter import GplearnDataAdapter
from .fitness import RankICFitness, BaseFitness
from .operators import get_time_series_operators, set_window_size
from .exceptions import GplearnDataError
from .config import (
    DEFAULT_POPULATION_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_HALL_OF_FAME,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_JOBS,
    DEFAULT_VERBOSE,
    DEFAULT_RANDOM_STATE,
)
from .constants import FitnessType


class GplearnFactorMiner:
    """
    Gplearn 因子挖掘器

    职责：
    - 加载和预处理数据
    - 配置遗传算法参数
    - 执行因子挖掘
    - 转换和输出结果
    """

    def __init__(
        self,
        market: str,
        start_date: str,
        end_date: str,
        base_features: List[str],
        target_col: str = "ret_1d",
        window_size: int = DEFAULT_WINDOW_SIZE,
        fitness_type: str = FitnessType.RANK_IC,
        # 遗传算法参数
        population_size: int = DEFAULT_POPULATION_SIZE,
        generations: int = DEFAULT_GENERATIONS,
        n_components: int = DEFAULT_N_COMPONENTS,
        hall_of_fame: int = DEFAULT_HALL_OF_FAME,
        max_depth: int = DEFAULT_MAX_DEPTH,
        # 并行和随机种子
        n_jobs: int = DEFAULT_N_JOBS,
        verbose: int = DEFAULT_VERBOSE,
        random_state: int = DEFAULT_RANDOM_STATE,
        # Qlib 配置
        qlib_provider_uri: str = "/Users/hm/Desktop/workspace/.cache/qlib_data",
        qlib_region: str = "cn",
    ):
        """
        初始化因子挖掘器

        Args:
            market: 市场标识（如 'csi300'）
            start_date: 训练开始日期
            end_date: 训练结束日期
            base_features: 基础特征列表（如 ['$close', '$volume']）
            target_col: 目标列名（如 'ret_1d'）
            window_size: 时间序列窗口大小
            fitness_type: 适应度类型（'rank_ic', 'weighted_ic', 'composite_ic'）
            population_size: 种群大小
            generations: 进化代数
            n_components: 输出因子数量
            hall_of_fame: 候选池大小
            max_depth: 最大树深度
            n_jobs: 并行度
            verbose: 详细程度（0=静默，1=进度，2=详细）
            random_state: 随机种子
            qlib_provider_uri: Qlib 数据路径
            qlib_region: Qlib 区域（'cn' 或 'us'）
        """
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.base_features = base_features
        self.target_col = target_col
        self.window_size = window_size
        self.fitness_type = fitness_type

        # 遗传算法参数
        self.population_size = population_size
        self.generations = generations
        self.n_components = n_components
        self.hall_of_fame = hall_of_fame
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        # Qlib 配置
        self.qlib_provider_uri = qlib_provider_uri
        self.qlib_region = qlib_region

        # 初始化组件
        self.adapter = GplearnDataAdapter(
            base_features=base_features,
            target_col=target_col,
            window_size=window_size,
        )

        # 初始化适应度函数（降低最小样本数要求）
        # 训练数据可能经过边界删除后样本量较少
        self.fitness_func = self._create_fitness_function(fitness_type)
        self.fitness_func.min_samples = max(5, len(base_features) * 2)

        # 初始化算子库
        set_window_size(window_size)
        self.operators = get_time_series_operators(window_size)

        # 训练数据
        self.X_train = None
        self.y_train = None
        self.train_index = None

    def _create_fitness_function(self, fitness_type: str) -> BaseFitness:
        """
        创建适应度函数

        Args:
            fitness_type: 适应度类型

        Returns:
            适应度函数实例
        """
        if fitness_type == FitnessType.RANK_IC:
            return RankICFitness(window_size=self.window_size)
        else:
            raise ValueError(f"不支持的适应度类型: {fitness_type}")

    def load_data(self) -> pd.DataFrame:
        """
        直接从 Qlib 加载训练数据

        Returns:
            MultiIndex DataFrame (instrument × datetime)

        Raises:
            GplearnDataError: 数据加载失败
        """
        try:
            import qlib
            from qlib.data import D

            # 初始化 Qlib（仅首次需要）
            # 使用 try-except 检测是否已初始化
            try:
                qlib.get_instance()
            except Exception:
                qlib.init(
                    provider_uri=self.qlib_provider_uri,
                    region=self.qlib_region,
                )

            # 从文件读取股票列表
            instruments_file = f"{self.qlib_provider_uri}/instruments/{self.market}.txt"
            try:
                with open(instruments_file, "r") as f:
                    lines = f.readlines()
                    instruments = [line.strip().split("\t")[0] for line in lines]
            except FileNotFoundError:
                raise GplearnDataError(f"找不到股票池文件: {instruments_file}")

            # 加载基础特征（OHLCV 等不需要缓存）
            df = D.features(
                instruments=instruments,
                fields=self.base_features,
                freq="day",
            )

            # 展平列名（如果有多层列索引）
            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(0)

            # 计算目标变量
            if self.target_col == "ret_1d":
                # 次日收益率
                df[self.target_col] = (
                    df["$close"].groupby("instrument").shift(-1) / df["$close"] - 1
                )
            else:
                raise ValueError(f"不支持的目标列: {self.target_col}")

            # 筛选日期范围
            df = df.loc[(slice(None), slice(self.start_date, self.end_date)), :]

            # 删除包含 NaN 的行（目标列的 shift(-1) 会导致最后一天为 NaN）
            df = df.dropna()

            if len(df) == 0:
                raise GplearnDataError(
                    f"筛选后无数据：{self.start_date} - {self.end_date}"
                )

            if self.verbose > 0:
                print(
                    f"    加载 {len(df.index.get_level_values('instrument').unique())} 只股票"
                )
                print(
                    f"    日期范围: {df.index.get_level_values('datetime').min()} ~ {df.index.get_level_values('datetime').max()}"
                )
                print(f"    数据形状: {df.shape}")

            return df

        except Exception as e:
            raise GplearnDataError(f"数据加载失败: {e}")

    def prepare_training_data(
        self,
        panel_data: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
        """
        准备训练数据

        Args:
            panel_data: 面板数据

        Returns:
            (X_flat, y_flat, index)
        """
        try:
            X_flat, y_flat, index = self.adapter.prepare_training_data(
                panel_data,
                dropna=True,
            )

            self.X_train = X_flat
            self.y_train = y_flat
            self.train_index = index

            return X_flat, y_flat, index

        except Exception as e:
            raise GplearnDataError(f"训练数据准备失败: {e}")

    def _custom_fitness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        自定义适应度函数（适配 Gplearn 接口）

        Args:
            y_true: 真实值
            y_pred: 预测值
            sample_weight: 样本权重（未使用）

        Returns:
            适应度分数
        """
        try:
            # 获取边界索引
            boundary_indices = self.adapter.get_boundary_indices()

            # 计算适应度
            score = self.fitness_func.compute(
                y_true=y_true,
                y_pred=y_pred,
                index=self.train_index,
                boundary_indices=boundary_indices,
            )

            return score

        except Exception as e:
            # 适应度计算失败时返回最低分
            if self.verbose > 0:
                print(f"适应度计算失败: {e}")
            return -np.inf

    def mine_factors(self) -> List[Dict[str, any]]:
        """
        执行因子挖掘

        Returns:
            因子列表，每个因子包含：
            - expression: Qlib 表达式
            - fitness: 适应度分数
            - raw_program: Gplearn 原始程序

        Raises:
            GplearnDataError: 数据准备失败
            FitnessCalculationError: 适应度计算失败
        """
        print("=" * 60)
        print("Gplearn 因子挖掘")
        print("=" * 60)
        print(f"市场: {self.market}")
        print(f"时间范围: {self.start_date} - {self.end_date}")
        print(f"基础特征: {', '.join(self.base_features)}")
        print(f"适应度类型: {self.fitness_type}")
        print(f"窗口大小: {self.window_size}")
        print(f"种群大小: {self.population_size}")
        print(f"进化代数: {self.generations}")
        print("=" * 60)

        # 1. 加载数据
        print("\n[1/5] 加载数据...")
        panel_data = self.load_data()
        print(f"    数据形状: {panel_data.shape}")

        # 2. 准备训练数据
        print("\n[2/5] 准备训练数据...")
        X_flat, y_flat, index = self.prepare_training_data(panel_data)
        print(f"    扁平数据形状: {X_flat.shape}")
        print(f"    边界数量: {len(self.adapter.get_boundary_indices())}")

        # 3. 初始化 Gplearn
        print("\n[3/5] 初始化遗传算法...")

        # 创建自定义适应度对象
        custom_fitness = _Fitness(
            function=self._custom_fitness,
            greater_is_better=True,  # Rank IC 越大越好
        )

        est = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=20,
            init_depth=(2, min(6, self.max_depth)),
            function_set=self.operators,
            metric=custom_fitness,
            parsimony_coefficient=0.01,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            warm_start=False,
        )

        # 4. 训练
        print("\n[4/5] 开始进化训练...")
        est.fit(X_flat, y_flat)

        # 5. 提取结果
        print("\n[5/5] 提取因子...")

        factors = []

        # SymbolicRegressor 只返回一个最优程序
        # 要获取多个因子，需要多次运行或使用不同的随机种子
        program = est._program

        try:
            # 提取 Gplearn 表达式
            gplearn_expr = program.__str__()

            # 替换特征占位符为实际名称
            readable_expr = self._replace_feature_names(gplearn_expr)

            # 计算适应度
            y_pred = program.execute(X_flat)
            fitness = self._custom_fitness(y_flat, y_pred)

            factors.append(
                {
                    "rank": 1,
                    "expression": readable_expr,  # 使用可读的表达式
                    "gplearn_expression": gplearn_expr,  # 保留 Gplearn 原始格式
                    "program": program,  # 保存 program 对象用于计算
                    "fitness": fitness,
                    "depth": program.depth_,
                    "length": program.length_,
                }
            )

        except Exception as e:
            print(f"    警告：因子提取失败: {e}")

        print(f"\n成功挖掘 {len(factors)} 个因子")

        # 打印 Top 5
        print("\n" + "=" * 60)
        print("Top 因子:")
        print("=" * 60)
        for factor in factors[:5]:
            print(f"\nRank {factor['rank']}:")
            print(f"  表达式: {factor['expression']}")
            print(f"  适应度: {factor['fitness']:.4f}")
            print(f"  深度: {factor['depth']}, 长度: {factor['length']}")

        return factors

    def _replace_feature_names(self, expression: str) -> str:
        """
        将表达式中的特征占位符（X0, X1, ...）替换为实际特征名称

        Args:
            expression: Gplearn 表达式（如 'min(momentum(X2))'）

        Returns:
            替换后的可读表达式（如 'min(momentum($total_mv))'）
        """
        import re

        result = expression
        for i, feature in enumerate(self.base_features):
            # 替换 Xi 为实际特征名（保留 $ 前缀）
            pattern = rf"\bX{i}\b"
            result = re.sub(pattern, feature, result)

        return result

    def save_factors(
        self,
        factors: List[Dict[str, any]],
        output_path: str,
    ) -> None:
        """
        保存因子到文件

        Args:
            factors: 因子列表
            output_path: 输出路径
        """
        # 保存为 CSV
        df = pd.DataFrame(factors)
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"\n因子已保存至: {output_path}")
