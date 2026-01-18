#!/usr/bin/env python3
"""
共享的 CLI 参数处理工具函数
为 Step 1-4 的 CLI 入口提供统一的参数解析和验证
"""

import re


def parse_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """
    验证并标准化日期范围参数

    Args:
        start_date: 起始日期字符串 (YYYY-MM-DD)
        end_date: 结束日期字符串 (YYYY-MM-DD)

    Returns:
        标准化的日期元组 (start_date, end_date)

    Raises:
        ValueError: 当日期格式不正确或 start_date > end_date 时
    """
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    if not date_pattern.match(start_date):
        raise ValueError(f"起始日期格式不正确: {start_date}，期望格式: YYYY-MM-DD")

    if not date_pattern.match(end_date):
        raise ValueError(f"结束日期格式不正确: {end_date}，期望格式: YYYY-MM-DD")

    if start_date > end_date:
        raise ValueError(f"起始日期不能晚于结束日期: {start_date} > {end_date}")

    return start_date, end_date


def validate_market(market: str) -> str:
    """
    验证市场参数有效性

    Args:
        market: 市场标识 (如: csi300, csi500, csi1000)

    Returns:
        验证后的市场标识（转为小写）

    Raises:
        ValueError: 当市场标识为空时
    """
    if not market or not market.strip():
        raise ValueError("市场标识不能为空")

    market_normalized = market.strip().lower()

    return market_normalized


def parse_periods(periods_str: str) -> dict[str, int]:
    """
    解析周期字符串为字典

    Args:
        periods_str: 逗号分隔的周期字符串，如 "1d,1w,1m"

    Returns:
        周期字典，如 {"1d": 1, "1w": 5, "1m": 20}

    Raises:
        ValueError: 当遇到不支持的周期时
    """
    if not periods_str or not periods_str.strip():
        raise ValueError("周期参数不能为空")

    period_map = {"1d": 1, "1w": 5, "1m": 20, "1q": 60, "1y": 252}

    periods = {}
    for p in periods_str.split(","):
        p = p.strip().lower()
        if p in period_map:
            periods[p] = period_map[p]
        else:
            raise ValueError(
                f"不支持的周期: {p}，支持的周期: {list(period_map.keys())}"
            )

    return periods


def parse_factor_formulas(formulas_str: str) -> list[str]:
    """
    解析因子表达式字符串为列表
    使用分号作为分隔符，支持表达式中的逗号

    Args:
        formulas_str: 分号分隔的因子表达式，如 "Ref($close,60)/$close;MOM($close,20)"

    Returns:
        因子表达式列表

    Raises:
        ValueError: 当因子表达式为空时
    """
    if not formulas_str or not formulas_str.strip():
        raise ValueError("因子表达式不能为空")

    formulas = [formula.strip() for formula in formulas_str.split(";")]

    # 过滤空字符串
    formulas = [f for f in formulas if f]

    if not formulas:
        raise ValueError("因子表达式不能为空")

    return formulas


def resolve_provider_uri() -> str:
    """
    获取默认的 provider_uri

    Returns:
        默认的 Qlib 数据路径字符串
    """
    return ".cache/qlib_data"
