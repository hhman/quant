#!/usr/bin/env python3
"""
Gplearn 因子挖掘 CLI

命令行工具用于执行遗传算法因子挖掘。

使用示例：
    # 基础用法
    python -m factor_mining.gplearn_mining \\
        --market csi300 \\
        --start-date 2023-01-01 \\
        --end-date 2024-12-31 \\
        --features $close $volume $total_mv

    # 使用配置文件
    python -m factor_mining.gplearn_mining --config config.yaml
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.gplearn import GplearnFactorMiner  # noqa: E402
from core.gplearn.config import (  # noqa: E402
    DEFAULT_WINDOW_SIZE,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_JOBS,
    DEFAULT_VERBOSE,
    DEFAULT_RANDOM_STATE,
)
from core.gplearn.constants import FitnessType  # noqa: E402


def parse_args():
    """
    解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Gplearn 遗传算法因子挖掘",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python -m factor_mining.gplearn_mining \\
      --market csi300 \\
      --start-date 2023-01-01 \\
      --end-date 2024-12-31 \\
      --features $close $volume $total_mv

  # 自定义参数
  python -m factor_mining.gplearn_mining \\
      --market csi300 \\
      --start-date 2023-01-01 \\
      --end-date 2024-12-31 \\
      --features $close $volume $total_mv \\
      --window-size 20 \\
      --population-size 2000 \\
      --generations 30

  # 指定输出目录
  python -m factor_mining.gplearn_mining \\
      --market csi300 \\
      --start-date 2023-01-01 \\
      --end-date 2024-12-31 \\
      --features $close $volume $total_mv \\
      --output-dir ./factors
        """,
    )

    # 必填参数
    parser.add_argument(
        "--market", type=str, required=True, help="市场标识（如 csi300, csi500）"
    )

    parser.add_argument(
        "--start-date", type=str, required=True, help="训练开始日期（格式: YYYY-MM-DD）"
    )

    parser.add_argument(
        "--end-date", type=str, required=True, help="训练结束日期（格式: YYYY-MM-DD）"
    )

    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        required=True,
        help="基础特征列表（如 $close $volume $total_mv）",
    )

    # 可选参数
    parser.add_argument(
        "--target-col", type=str, default="ret_1d", help="目标列名（默认: ret_1d）"
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"时间序列窗口大小（默认: {DEFAULT_WINDOW_SIZE}）",
    )

    parser.add_argument(
        "--fitness-type",
        type=str,
        default=FitnessType.RANK_IC,
        choices=[
            FitnessType.RANK_IC,
            FitnessType.WEIGHTED_IC,
            FitnessType.COMPOSITE_IC,
        ],
        help=f"适应度类型（默认: {FitnessType.RANK_IC}）",
    )

    # 遗传算法参数
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help=f"种群大小（默认: {DEFAULT_POPULATION_SIZE}）",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help=f"进化代数（默认: {DEFAULT_GENERATIONS}）",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help=f"输出因子数量（默认: {DEFAULT_N_COMPONENTS}）",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"最大树深度（默认: {DEFAULT_MAX_DEPTH}）",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"并行度（默认: {DEFAULT_N_JOBS}）",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=DEFAULT_VERBOSE,
        choices=[0, 1, 2],
        help=f"详细程度（默认: {DEFAULT_VERBOSE}）",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"随机种子（默认: {DEFAULT_RANDOM_STATE}）",
    )

    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: .cache/factor_mining）",
    )

    parser.add_argument(
        "--qlib-provider-uri",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib 数据路径（默认: ~/.qlib/qlib_data/cn_data）",
    )

    parser.add_argument(
        "--qlib-region",
        type=str,
        default="cn",
        choices=["cn", "us"],
        help="Qlib 区域（默认: cn）",
    )

    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()

    # 构造输出目录
    if args.output_dir is None:
        output_dir = Path(".cache") / "factor_mining"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化挖掘器
    try:
        miner = GplearnFactorMiner(
            market=args.market,
            start_date=args.start_date,
            end_date=args.end_date,
            base_features=args.features,
            target_col=args.target_col,
            window_size=args.window_size,
            fitness_type=args.fitness_type,
            population_size=args.population_size,
            generations=args.generations,
            n_components=args.n_components,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            random_state=args.random_state,
            qlib_provider_uri=args.qlib_provider_uri,
            qlib_region=args.qlib_region,
        )

        # 执行挖掘
        factors = miner.mine_factors()

        # 保存结果
        output_path = (
            output_dir / f"{args.market}_{args.start_date}_{args.end_date}_factors.csv"
        )
        miner.save_factors(factors, output_path)

        print(f"\n✓ 挖掘完成！共发现 {len(factors)} 个因子")
        print(f"✓ 结果已保存至: {output_path}")

        return 0

    except Exception as e:
        print(f"\n✗ 挖掘失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
