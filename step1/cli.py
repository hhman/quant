#!/usr/bin/env python3
"""Step 1 CLI: 因子计算模块的命令行接口。"""

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
    """解析命令行参数。

    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="Step 1: 因子计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step1/cli.py --market csi300 --start-date 2023-01-01 --end-date 2025-12-31 \\
      --factor-formulas "Ref($close,60)/$close;MA($close,20)"
        """,
    )

    parser.add_argument(
        "--market", type=str, default="csi300", help="市场名称 (default: csi300)"
    )

    parser.add_argument(
        "--start-date", type=str, required=True, help="开始日期 (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=str, required=True, help="结束日期 (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--factor-formulas",
        type=str,
        required=True,
        help='因子公式列表: "Ref($close,60)/$close;MA($close,20)"',
    )

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> dict:
    """规范化并验证命令行参数。

    Args:
        args: 解析后的CLI参数命名空间

    Returns:
        规范化后的参数字典

    Raises:
        ValueError: 参数验证失败
    """
    market = validate_market(args.market)

    start_date, end_date = parse_date_range(args.start_date, args.end_date)

    factor_formulas = parse_factor_formulas(args.factor_formulas)

    provider_uri = resolve_provider_uri()

    return {
        "market": market,
        "start_date": start_date,
        "end_date": end_date,
        "factor_formulas": factor_formulas,
        "provider_uri": provider_uri,
    }


def main():
    args = parse_args()

    params = normalize_args(args)

    from step1.factor_extraction_preprocessing import calculate_factors

    calculate_factors(**params)

    print("Step1完成! 缓存位置: .cache/")


if __name__ == "__main__":
    main()
