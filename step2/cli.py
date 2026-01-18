#!/usr/bin/env python3
"""
Step 2 CLI: 因子中性化
独立的 CLI 入口脚本
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cli_args import (
    parse_date_range,
    validate_market,
    parse_factor_formulas,
    resolve_provider_uri,
)


def parse_args() -> argparse.Namespace:
    """
    定义并解析 CLI 参数

    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="Step 2: 因子中性化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step2/cli.py --market csi300 --start-date 2023-01-01 --end-date 2025-12-31 \\
      --factor-formulas "Ref($close,60)/$close;MA($close,20)"
        """,
    )

    # 核心参数
    parser.add_argument(
        "--market", type=str, default="csi300", help="市场标识 (default: csi300)"
    )

    parser.add_argument(
        "--start-date", type=str, required=True, help="起始日期 (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=str, required=True, help="结束日期 (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--factor-formulas",
        type=str,
        required=True,
        help='因子表达式列表，分号分隔，如: "Ref($close,60)/$close;MA($close,20)"',
    )

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> dict:
    """
    校验和标准化参数

    Args:
        args: 解析后的 CLI 参数

    Returns:
        标准化的参数字典

    Raises:
        ValueError: 当参数验证失败时
    """
    # 验证并标准化市场参数
    market = validate_market(args.market)

    # 验证日期范围
    start_date, end_date = parse_date_range(args.start_date, args.end_date)

    # 解析因子表达式
    factor_formulas = parse_factor_formulas(args.factor_formulas)

    # 获取默认 provider_uri
    provider_uri = resolve_provider_uri()

    return {
        "market": market,
        "start_date": start_date,
        "end_date": end_date,
        "factor_formulas": factor_formulas,
        "provider_uri": provider_uri,
    }


def main():
    """
    主入口函数
    """
    # 1. 解析 CLI 参数
    args = parse_args()

    # 2. 标准化参数
    params = normalize_args(args)

    # 3. 调用核心逻辑函数
    from step2.因子中性化 import neutralize_factors

    neutralize_factors(**params)

    print("\n✅ Step 2 完成!")
    print("   结果已保存到: .cache/")


if __name__ == "__main__":
    main()
