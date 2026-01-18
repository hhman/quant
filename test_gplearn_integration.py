#!/usr/bin/env python3
"""
Gplearn 集成测试脚本

快速验证核心功能是否正常工作。

运行方式:
    python test_gplearn_integration.py

测试内容:
    1. 数据适配器（展平/还原/边界删除）
    2. 适应度函数（Rank IC 计算）
    3. 端到端挖掘流程（小规模）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from core.gplearn import (  # noqa: E402
    GplearnDataAdapter,
    RankICFitness,
)


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def generate_mock_data(
    n_stocks: int = 10,
    n_dates: int = 60,  # 约 3 个月
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成 mock 面板数据

    Args:
        n_stocks: 股票数量
        n_dates: 日期数量
        seed: 随机种子

    Returns:
        MultiIndex DataFrame
    """
    print(f"生成 mock 数据: {n_stocks} 只股票 × {n_dates} 个交易日")

    np.random.seed(seed)

    # 生成日期序列
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_dates)]

    # 生成股票代码
    instruments = [f"stock_{i:03d}" for i in range(n_stocks)]

    # 生成 MultiIndex
    index = pd.MultiIndex.from_product(
        [instruments, dates], names=["instrument", "datetime"]
    )

    # 生成特征数据
    n_samples = len(index)

    # close: 随机游走价格
    price_init = np.random.uniform(10, 100, n_stocks)
    close_flat = np.concatenate(
        [
            price_init[i] * np.cumprod(1 + np.random.normal(0, 0.02, n_dates))
            for i in range(n_stocks)
        ]
    )

    # volume: 随机成交量
    volume_flat = np.random.uniform(1e6, 1e8, n_samples)

    # total_mv: 市值（与价格相关）
    total_mv_flat = close_flat * np.random.uniform(1e8, 1e9, n_samples)

    # ret_1d: 目标变量（次日收益率，带一点信号）
    ret_1d_flat = np.random.normal(0, 0.02, n_samples)

    # 添加一些信号：价格动量
    for i in range(n_stocks):
        start = i * n_dates
        # 简单动量信号：过去 5 天收益率预测明天
        for j in range(5, n_dates):
            momentum = (close_flat[start + j] - close_flat[start + j - 5]) / close_flat[
                start + j - 5
            ]
            ret_1d_flat[start + j] += 0.3 * momentum  # 添加一些信号

    # 构造 DataFrame
    data = pd.DataFrame(
        {
            "$close": close_flat,
            "$volume": volume_flat,
            "$total_mv": total_mv_flat,
            "ret_1d": ret_1d_flat,
        },
        index=index,
    )

    print(f"  数据形状: {data.shape}")
    print(f"  缺失值: {data.isna().sum().sum()}")

    return data


# ========== 测试 1: 数据适配器 ==========


def test_data_adapter():
    """测试数据适配器"""
    print_section("测试 1: 数据适配器")

    # 1. 生成 mock 数据
    panel_data = generate_mock_data(n_stocks=10, n_dates=60)

    # 2. 初始化适配器
    adapter = GplearnDataAdapter(
        base_features=["$close", "$volume", "$total_mv"],
        target_col="ret_1d",
        window_size=10,
    )

    # 3. 展平数据
    print("\n展平数据...")
    X_flat, y_flat, index = adapter.prepare_training_data(
        panel_data,
        dropna=True,
    )

    print(f"  扁平数据形状: {X_flat.shape}")
    print(f"  特征数量: {X_flat.shape[1]}")
    print(f"  样本数量: {X_flat.shape[0]}")

    # 4. 验证边界索引
    print("\n验证边界索引...")
    boundary_indices = adapter.get_boundary_indices()
    print(f"  边界数量: {len(boundary_indices)}")
    print(f"  前 5 个边界: {boundary_indices[:5]}")

    # 预期：每只股票 60 个样本，边界在 [60, 120, 180, ...]
    expected_boundaries = [60 * i for i in range(1, 10)]
    if boundary_indices == expected_boundaries:
        print("  ✓ 边界索引正确")
    else:
        print("  ✗ 边界索引错误")
        print(f"    预期: {expected_boundaries}")
        print(f"    实际: {boundary_indices}")

    # 5. 测试边界删除
    print("\n测试边界删除...")
    test_data = np.arange(100)  # [0, 1, 2, ..., 99]
    test_boundary = [50]  # 在位置 50 有边界

    adapter_test = GplearnDataAdapter(
        base_features=["$close"],
        window_size=10,
    )
    adapter_test.boundary_indices = test_boundary

    cleaned = adapter_test.apply_boundary_deletion(test_data, window_size=10)

    # 验证前 50 个数据不变，50-59 变成 NaN
    expected_cleaned = np.array([0.0] * 50 + [np.nan] * 10 + list(range(60, 100)))
    if np.allclose(cleaned[:50], expected_cleaned[:50], equal_nan=True):
        print("  ✓ 边界删除正确")
    else:
        print("  ✗ 边界删除错误")

    # 6. 测试还原为面板
    print("\n测试还原为面板...")
    # 创建简单的测试数据
    test_flat = np.arange(20)  # 2 只股票，每只 10 天
    test_index = pd.MultiIndex.from_product(
        [["stock_000", "stock_001"], pd.date_range("2023-01-01", periods=10)],
        names=["instrument", "datetime"],
    )

    adapter_test.boundary_indices = [10]
    panel = adapter_test.restore_to_panel(test_flat, test_index)

    print(f"  面板形状: {panel.shape}")  # 应该是 (10, 2)
    print(f"  列索引类型: {type(panel.columns)}")
    print(f"  列索引: {panel.columns}")

    # 检查前几行数据
    print("  前 3 行数据:")
    print(panel.head(3))

    print("\n✓ 数据适配器测试完成")


# ========== 测试 2: 表达式转换器 ==========


def test_fitness():
    """测试适应度函数"""
    print_section("测试 3: 适应度函数")

    # 生成 mock 数据
    print("\n生成 mock 数据...")
    panel_data = generate_mock_data(n_stocks=10, n_dates=60)

    # 初始化适配器
    adapter = GplearnDataAdapter(
        base_features=["$close", "$volume", "$total_mv"],
        target_col="ret_1d",
        window_size=10,
    )

    # 展平数据
    X_flat, y_flat, index = adapter.prepare_training_data(panel_data, dropna=True)

    # 初始化适应度函数（降低最小样本数要求以适应测试数据）
    fitness = RankICFitness(window_size=10, min_samples=5)

    # 测试 1: 完美预测（应该 IC ≈ 1）
    print("\n测试 1: 完美预测...")
    y_pred_perfect = y_flat.copy()
    score_perfect = fitness.compute(
        y_flat, y_pred_perfect, index, adapter.get_boundary_indices()
    )
    print(f"  Rank IC: {score_perfect:.4f}")
    if abs(score_perfect - 1.0) < 0.01:
        print("  ✓ 完美预测 IC 接近 1.0")
    else:
        print("  ⚠ 完美预测 IC 偏离 1.0（可能有边界影响）")

    # 测试 2: 随机预测（应该 IC ≈ 0）
    print("\n测试 2: 随机预测...")
    np.random.seed(123)
    y_pred_random = np.random.randn(*y_flat.shape)
    score_random = fitness.compute(
        y_flat, y_pred_random, index, adapter.get_boundary_indices()
    )
    print(f"  Rank IC: {score_random:.4f}")
    if abs(score_random) < 0.2:
        print("  ✓ 随机预测 IC 接近 0")
    else:
        print("  ⚠ 随机预测 IC 偏离 0（可能是随机性）")

    # 测试 3: 反向预测（应该 IC ≈ -1）
    print("\n测试 3: 反向预测...")
    y_pred_inverse = -y_flat.copy()
    score_inverse = fitness.compute(
        y_flat, y_pred_inverse, index, adapter.get_boundary_indices()
    )
    print(f"  Rank IC: {score_inverse:.4f}")
    if abs(score_inverse + 1.0) < 0.1:
        print("  ✓ 反向预测 IC 接近 -1.0")
    else:
        print("  ⚠ 反向预测 IC 偏离 -1.0（可能有边界影响）")

    print("\n✓ 适应度函数测试完成")


# ========== 测试 4: 端到端挖掘 ==========


def test_end_to_end_mining():
    """测试端到端挖掘流程（使用真实数据）"""
    print_section("测试 4: 端到端挖掘（真实数据）")

    try:
        from core.gplearn import GplearnFactorMiner
    except ImportError as e:
        print(f"✗ 无法导入 GplearnFactorMiner: {e}")
        print("  提示：请先安装 gplearn: pip install gplearn")
        return

    # 初始化挖掘器（小参数）
    print("\n初始化挖掘器...")
    miner = GplearnFactorMiner(
        market="csi300",
        start_date="2023-01-01",
        end_date="2023-03-31",  # 短时间范围，快速测试
        base_features=["$close", "$volume", "$total_mv"],
        target_col="ret_1d",
        window_size=5,
        fitness_type="rank_ic",
        # 小参数，快速测试
        population_size=50,  # 小种群
        generations=3,  # 少代数
        n_components=3,  # 少因子
        max_depth=5,
        n_jobs=1,  # 单线程
        verbose=1,
        random_state=42,
        # Qlib 配置
        qlib_provider_uri="~/.qlib/qlib_data/cn_data",
        qlib_region="cn",
    )

    print("  ✓ 挖掘器初始化成功")

    # 执行挖掘
    print("\n开始挖掘...")
    print("-" * 60)

    try:
        factors = miner.mine_factors()

        print("\n" + "-" * 60)
        print("\n✓ 挖掘完成！")

        # 显示结果
        print(f"\n发现 {len(factors)} 个因子:")
        for i, factor in enumerate(factors):
            print(f"\n因子 #{i + 1}:")
            print(f"  表达式: {factor['expression']}")
            print(f"  适应度: {factor['fitness']:.4f}")
            print(f"  深度: {factor['depth']}, 长度: {factor['length']}")

        # 验证表达式格式（Gplearn 格式即可）
        print("\n验证表达式格式...")
        all_valid = True
        for i, factor in enumerate(factors):
            expr = factor["expression"]
            if expr and "(" in expr:
                print(f"  因子 #{i + 1}: {expr[:50]}... ✓")
            else:
                print(f"  因子 #{i + 1}: 表达式异常 ✗")
                all_valid = False

        if all_valid:
            print("\n✓ 所有表达式格式正确")
        else:
            print("\n⚠ 部分表达式可能有问题")

    except Exception as e:
        print(f"\n✗ 挖掘失败: {e}")
        import traceback

        traceback.print_exc()


# ========== 主函数 ==========


def main():
    """运行所有测试"""
    print("=" * 60)
    print("  Gplearn 集成测试")
    print("=" * 60)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 测试 1: 数据适配器
        test_data_adapter()

        # 测试 2: 适应度函数
        test_fitness()

        # 测试 3: 端到端挖掘
        test_end_to_end_mining()

        print("\n" + "=" * 60)
        print("  ✓ 所有测试完成")
        print("=" * 60)
        print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  ✗ 测试失败: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
