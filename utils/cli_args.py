#!/usr/bin/env python3
"""CLI参数解析工具，提供Step 1-4通用参数验证功能。"""

import re


def parse_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """解析并验证日期范围。

    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)

    Returns:
        验证后的日期元组 (start_date, end_date)

    Raises:
        ValueError: 日期格式错误或start_date > end_date
    """
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    if not date_pattern.match(start_date):
        raise ValueError(f"日期格式错误: {start_date}，应为 YYYY-MM-DD")

    if not date_pattern.match(end_date):
        raise ValueError(f"日期格式错误: {end_date}，应为 YYYY-MM-DD")

    if start_date > end_date:
        raise ValueError(f"日期范围错误: {start_date} > {end_date}")

    return start_date, end_date


def validate_market(market: str) -> str:
    """验证并规范化市场参数。

    Args:
        market: 市场名称（如: csi300, csi500, csi1000）

    Returns:
        规范化后的市场名称（小写）

    Raises:
        ValueError: 市场名称为空
    """
    if not market or not market.strip():
        raise ValueError("市场名称不能为空")

    market_normalized = market.strip().lower()

    return market_normalized


def parse_periods(periods_str: str) -> dict[str, int]:
    """解析周期参数字符串。

    Args:
        periods_str: 周期字符串，如 "1d,1w,1m"

    Returns:
        周期映射字典，如 {"1d": 1, "1w": 5, "1m": 20}

    Raises:
        ValueError: 周期格式错误或包含不支持的周期
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
            raise ValueError(f"不支持的周期: {p}，可选值: {list(period_map.keys())}")

    return periods


def parse_factor_formulas(formulas_str: str) -> list[str]:
    """解析因子公式字符串。

    支持分号或逗号分隔的多个公式，自动过滤空字符串。

    Args:
        formulas_str: 公式字符串，如 "Ref($close,60)/$close;MOM($close,20)"

    Returns:
        因子公式列表

    Raises:
        ValueError: 公式字符串为空或解析后为空列表
    """
    if not formulas_str or not formulas_str.strip():
        raise ValueError("因子公式不能为空")

    formulas = [formula.strip() for formula in formulas_str.split(";")]

    formulas = [f for f in formulas if f]

    if not formulas:
        raise ValueError("因子公式解析失败，列表为空")

    return formulas


def resolve_provider_uri() -> str:
    """解析Qlib数据源路径。

    Returns:
        Qlib数据源目录路径
    """
    return ".cache/qlib_data"
