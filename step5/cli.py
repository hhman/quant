#!/usr/bin/env python3
"""Step 5 CLI:"""

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
    """CLI"""
    parser = argparse.ArgumentParser(
        description="Step 5: ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
:
  python step5/cli.py --market csi300 --start-date 2020-01-01 --end-date 2023-12-31
        """,
    )

    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--random-state", type=int, default=None)

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> dict:
    """"""
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
    """"""
    args = parse_args()
    params = normalize_args(args)

    # 执行GP因子挖掘
    from step5.genetic_algorithm_factor_mining import mine_factors_with_gp

    mine_factors_with_gp(
        provider_uri=params["provider_uri"],
        market=params["market"],
        start_date=params["start_date"],
        end_date=params["end_date"],
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
