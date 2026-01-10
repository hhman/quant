#!/usr/bin/env python3
"""
CLI参数解析器 - 统一的参数体系
所有step脚本使用完全相同的参数
"""
import argparse
from typing import Dict, List, Optional


def parse_common_args() -> argparse.Namespace:
    """
    解析统一的CLI参数
    所有step使用完全相同的参数
    """
    parser = argparse.ArgumentParser(
        description='Quant因子分析流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Step0: 数据预处理 (从CSV到Qlib格式)
  bash step0/step0.sh --start-date 2008-01-01 --end-date 2025-01-01 --raw-data-dir raw_data --verbose

  # Step1: 生成cache（使用因子表达式）
  python step1/因子提取与预处理.py --market csi300 --factor-formulas "Ref($close,60)/$close,MOM($close,20)" --periods 1d,1w,1m

  # Step2: 使用cache中的所有数据
  python step2/因子中性化.py --market csi300

  # Step2: 指定子集表达式（智能匹配）
  python step2/因子中性化.py --market csi300 --factor-formulas "Ref($close,60)/$close" --periods 1d

  # Step2: 指定时间范围
  python step2/因子中性化.py --market csi300 --start-date 2022-01-01 --end-date 2023-01-01

  # Step3: 因子收益率计算
  python step3/因子收益率计算.py --market csi300

  # Step4: 因子绩效评估
  python step4/因子绩效评估.py --market csi300
        """
    )

    # ========== 核心参数 ==========
    parser.add_argument(
        '--market',
        type=str,
        default='csi300',
        help='市场标识 (default: csi300)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,  # None表示使用cache中的值
        help='起始日期 (default: 使用cache中的起始日期)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,  # None表示使用cache中的值
        help='结束日期 (default: 使用cache中的结束日期)'
    )

    parser.add_argument(
        '--factor-formulas',
        type=str,
        default=None,  # None表示使用cache中的所有因子
        help='因子表达式列表，逗号分隔 (default: 使用cache中的所有因子)'
    )

    parser.add_argument(
        '--periods',
        type=str,
        default=None,  # None表示使用cache中的所有周期
        help='收益率周期，逗号分隔，如: 1d,1w,1m (default: 使用cache中的所有周期)'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='cache',
        help='Cache目录路径 (default: cache)'
    )

    parser.add_argument(
        '--provider-uri',
        type=str,
        default='~/data/qlib_data',
        help='Qlib数据路径 (仅step1需要, default: ~/data/qlib_data)'
    )

    # ========== Step0专用参数 ==========
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='raw_data',
        help='原始CSV数据目录 (仅step0使用, default: raw_data)'
    )

    parser.add_argument(
        '--index-dir',
        type=str,
        default=None,
        help='指数CSV目录 (仅step0使用, default: {raw_data_dir}/index)'
    )

    parser.add_argument(
        '--stock-dir',
        type=str,
        default=None,
        help='股票CSV目录 (仅step0使用, default: {raw_data_dir}/stock)'
    )

    parser.add_argument(
        '--finance-dir',
        type=str,
        default=None,
        help='财务CSV目录 (仅step0使用, default: {raw_data_dir}/finance)'
    )

    parser.add_argument(
        '--qlib-src-dir',
        type=str,
        default='qlib_src',
        help='Qlib脚本目录 (仅step0使用, default: qlib_src)'
    )

    # ========== 调试参数 ==========
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行，不实际执行'
    )

    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='强制重新生成cache (仅step1有效)'
    )

    return parser.parse_args()


def parse_periods(periods_str: Optional[str]) -> Optional[Dict[str, int]]:
    """
    解析周期字符串为字典

    Args:
        periods_str: 逗号分隔的周期字符串，如 "1d,1w,1m"

    Returns:
        周期字典，如 {"1d": 1, "1w": 5, "1m": 20}

    Raises:
        ValueError: 当遇到不支持的周期时
    """
    if periods_str is None:
        return None

    period_map = {
        '1d': 1,
        '1w': 5,
        '1m': 20,
        '1q': 60,
        '1y': 252
    }

    periods = {}
    for p in periods_str.split(','):
        p = p.strip()
        if p in period_map:
            periods[p] = period_map[p]
        else:
            raise ValueError(
                f"不支持的周期: {p}，"
                f"支持的周期: {list(period_map.keys())}"
            )

    return periods


def parse_factor_formulas(factor_formulas_str: Optional[str]) -> Optional[List[str]]:
    """
    解析因子表达式字符串为列表

    Args:
        factor_formulas_str: 逗号分隔的因子表达式，如 "Ref($close,60)/$close,MOM($close,20)"

    Returns:
        因子表达式列表
    """
    if factor_formulas_str is None:
        return None

    return [formula.strip() for formula in factor_formulas_str.split(',')]


def normalize_args(args: argparse.Namespace, step: str) -> dict:
    """
    标准化参数

    Args:
        args: CLI参数
        step: 步骤名称 ("step0", "step1", "step2", "step3", "step4")

    Returns:
        标准化的参数字典

    Raises:
        ValueError: 当step1缺少必需参数时
    """
    params = {
        'market': args.market,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'factor_formulas': parse_factor_formulas(args.factor_formulas),
        'periods': parse_periods(args.periods),
        'cache_dir': args.cache_dir,
        'provider_uri': args.provider_uri,
        'verbose': args.verbose,
        'dry_run': args.dry_run,
        'force_regenerate': args.force_regenerate,
        'step': step
    }

    # Step0专用参数 (仅step0使用,其他step忽略)
    if step == "step0":
        params['raw_data_dir'] = args.raw_data_dir

        # 如果未指定子目录,使用默认值
        params['index_dir'] = args.index_dir or f"{args.raw_data_dir}/index"
        params['stock_dir'] = args.stock_dir or f"{args.raw_data_dir}/stock"
        params['finance_dir'] = args.finance_dir or f"{args.raw_data_dir}/finance"
        params['qlib_src_dir'] = args.qlib_src_dir

        # Step0的输出目录 (临时数据,不是最终的cache)
        params['output_dir'] = f"{args.cache_dir}/step0_temp"

    # step1必须明确指定factor_formulas和periods
    if step == "step1":
        if params['factor_formulas'] is None:
            raise ValueError("step1必须使用 --factor-formulas 参数指定因子表达式列表")
        if params['periods'] is None:
            raise ValueError("step1必须使用 --periods 参数指定周期列表")

    return params
