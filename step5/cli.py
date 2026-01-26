#!/usr/bin/env python3
"""Step 5 CLI: 遗传算法因子挖掘"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cli_args import (
    parse_date_range,
    validate_market,
    resolve_provider_uri,
)


def parse_args() -> argparse.Namespace:
    """定义并解析 CLI 参数"""
    parser = argparse.ArgumentParser(
        description="Step 5: 遗传算法因子挖掘",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step5/cli.py --market csi300 --start-date 2020-01-01 --end-date 2023-12-31
        """,
    )

    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--random-state", type=int, default=None)

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> dict:
    """校验和标准化参数"""
    market = validate_market(args.market)
    start_date, end_date = parse_date_range(args.start_date, args.end_date)
    provider_uri = resolve_provider_uri()

    return {
        "market": market,
        "start_date": start_date,
        "end_date": end_date,
        "provider_uri": provider_uri,
    }


def main():
    """主入口函数"""
    args = parse_args()
    params = normalize_args(args)

    # 调用业务逻辑
    from step5.遗传算法因子挖掘 import mine_factors_with_gp

    mine_factors_with_gp(
        provider_uri=params["provider_uri"],
        market=params["market"],
        start_date=params["start_date"],
        end_date=params["end_date"],
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
