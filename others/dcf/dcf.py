#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd

pd.set_option("display.max_rows", None)

# ---- 常量配置 ----
FINANCIAL_ROOT = Path("/Users/hm/Desktop/workspace/finance")
TRADING_ROOT = Path("/Users/hm/Desktop/workspace/stock")
UNIT = 1e8  # 将报表金额换算为“亿元”口径，保持与 Excel 模板一致
PROJECTION_YEARS = 10  # 对应 Excel 的 10 年显式期（估值2!H16:Q16）
DEFAULT_GROWTH = 0.08  # 对应“常规”场景的增长率输入（估值2!G16）
DEFAULT_DISCOUNT = 0.08  # 对应贴现率（估值2!K16 与估值2!D10）
TERMINAL_GROWTH = 0.0  # 对应十年以后增长率（估值2!D16）
MAX_REPORT_AGE_DAYS = 365  # 只接受最近约一年内的年报，避免使用陈旧数据


@dataclass
class ScenarioResult:
    name: str
    growth: float
    cashflow_pv: float
    terminal_pv: float
    intrinsic_value: float
    fair_price: float


@dataclass
class ValuationSummary:
    code: str
    name: str
    report_date: pd.Timestamp
    publish_date: pd.Timestamp
    publish_trade_date: pd.Timestamp
    net_profit: float
    shares: float
    fair_price: float
    latest_price: float
    trade_date: pd.Timestamp
    undervaluation_pct: float
    return_since_publish_pct: float


# ---- DCF 计算工具函数 ----
def project_cashflows(
    base_cf: float, growth: float, years: int = PROJECTION_YEARS
) -> List[float]:
    return [base_cf * (1 + growth) ** year for year in range(1, years + 1)]


def discount_cashflows(
    cashflows: Iterable[float],
    discount_rate: float,
    years: int = PROJECTION_YEARS,
) -> List[float]:
    return [
        cf / (1 + discount_rate) ** year for year, cf in enumerate(cashflows, start=1)
    ]


def terminal_value(
    last_cf: float,
    discount_rate: float,
    terminal_growth: float,
    years: int = PROJECTION_YEARS,
) -> float:
    if discount_rate <= terminal_growth:
        raise ValueError("Discount rate must be greater than terminal growth.")
    return (
        last_cf
        * (1 + terminal_growth)
        / ((discount_rate - terminal_growth) * (1 + discount_rate) ** years)
    )


def compute_scenario(
    name: str,
    base_cf: float,
    shares: float,
    growth: float,
    discount_rate: float = DEFAULT_DISCOUNT,
    terminal_growth: float = TERMINAL_GROWTH,
    years: int = PROJECTION_YEARS,
) -> ScenarioResult:
    if shares <= 0:
        raise ValueError("Shares outstanding must be positive.")
    cashflows = project_cashflows(base_cf, growth, years=years)
    discounted = discount_cashflows(cashflows, discount_rate, years=years)
    cashflow_pv = sum(discounted)
    terminal_pv = terminal_value(
        cashflows[-1], discount_rate, terminal_growth, years=years
    )
    intrinsic_value = cashflow_pv + terminal_pv
    fair_price = intrinsic_value / shares
    return ScenarioResult(
        name=name,
        growth=growth,
        cashflow_pv=cashflow_pv,
        terminal_pv=terminal_pv,
        intrinsic_value=intrinsic_value,
        fair_price=fair_price,
    )


# ---- 数据读取与筛选 ----
def iter_general_enterprise_csv(root: Path) -> Iterator[Path]:
    for code_dir in sorted(root.iterdir()):
        if not code_dir.is_dir():
            continue
        code = code_dir.name.lower()
        if code.startswith("bj") or code.startswith("sh68"):
            continue
        for file in code_dir.glob("*_一般企业*.csv"):
            yield file


def load_financial_snapshot(path: Path) -> Optional[Dict[str, float]]:
    target_cols = {"report_date", "publish_date", "R_np_atoopc@xbx"}
    df = pd.read_csv(
        path,
        encoding="gbk",
        skiprows=1,
        usecols=lambda col: col in target_cols,
    )
    df["report_date"] = pd.to_datetime(
        df["report_date"], format="%Y%m%d", errors="coerce"
    )
    df["publish_date"] = pd.to_datetime(
        df["publish_date"], format="%Y-%m-%d", errors="coerce"
    )
    df = df.dropna(subset=["report_date", "publish_date"])
    if df.empty or "R_np_atoopc@xbx" not in df.columns:
        return None

    snapshot = df[df["report_date"].dt.month.eq(12) & df["report_date"].dt.day.eq(31)]
    if snapshot.empty:
        return None
    snapshot = snapshot.sort_values(
        ["report_date", "publish_date"], ascending=[False, False]
    ).drop_duplicates(subset=["report_date"], keep="first")

    latest = snapshot.iloc[0]
    net_profit = latest["R_np_atoopc@xbx"]
    report_date = latest["report_date"].normalize()
    publish_date = latest["publish_date"].normalize()

    if pd.Timestamp.now().normalize() - report_date > pd.Timedelta(
        days=MAX_REPORT_AGE_DAYS
    ):
        return None

    if pd.isna(net_profit):
        return None
    if net_profit <= 0:
        return None

    return {
        "report_date": report_date,
        "publish_date": publish_date,
        "net_profit": float(net_profit),
    }


def load_price_info(
    code: str, root: Path, publish_date: pd.Timestamp
) -> Optional[tuple[float, str, pd.Timestamp, float, pd.Timestamp, float]]:
    price_path = root / f"{code}.csv"
    if not price_path.exists():
        return None
    df = pd.read_csv(
        price_path,
        encoding="gbk",
        skiprows=1,
        usecols=lambda col: col in {"交易日期", "收盘价", "股票名称", "总市值"},
    )
    if df.empty or "收盘价" not in df.columns or "交易日期" not in df.columns:
        return None

    df["交易日期"] = pd.to_datetime(df["交易日期"], errors="coerce")
    df = df.dropna(subset=["交易日期", "收盘价", "总市值"])
    if df.empty:
        return None

    df = df.sort_values("交易日期")

    publish_rows = df[df["交易日期"] >= publish_date]
    if publish_rows.empty:
        return None
    publish_row = publish_rows.iloc[0]
    publish_trade_date = publish_row["交易日期"].normalize()
    publish_price = publish_row["收盘价"]
    if publish_price <= 0:
        return None

    latest_row = df.iloc[-1]
    latest_close = latest_row["收盘价"]
    stock_name = str(latest_row.get("股票名称", ""))
    trade_date = latest_row["交易日期"].normalize()
    if latest_close <= 0 or trade_date <= publish_trade_date:
        return None

    total_market_cap = latest_row["总市值"]
    if total_market_cap <= 0:
        return None

    shares_in_100m = (total_market_cap / latest_close) / UNIT
    if shares_in_100m <= 0:
        return None

    return (
        float(latest_close),
        stock_name,
        trade_date,
        float(shares_in_100m),
        publish_trade_date,
        float(publish_price),
    )


def evaluate_stock(financial_path: Path) -> Optional[ValuationSummary]:
    code = financial_path.stem.split("_")[0].lower()
    snapshot = load_financial_snapshot(financial_path)
    if snapshot is None:
        return None

    price_info = load_price_info(code, TRADING_ROOT, snapshot["publish_date"])
    if price_info is None:
        return None
    (
        latest_price,
        stock_name,
        trade_date,
        shares_in_100m,
        publish_trade_date,
        publish_price,
    ) = price_info

    base_cf = snapshot["net_profit"]
    shares = shares_in_100m * UNIT
    scenario = compute_scenario(
        name="常规",
        base_cf=base_cf,
        shares=shares,
        growth=DEFAULT_GROWTH,
        discount_rate=DEFAULT_DISCOUNT,
        terminal_growth=TERMINAL_GROWTH,
        years=PROJECTION_YEARS,
    )

    return ValuationSummary(
        code=code,
        name=stock_name,
        report_date=pd.Timestamp(snapshot["report_date"]),
        publish_date=pd.Timestamp(snapshot["publish_date"]),
        publish_trade_date=publish_trade_date,
        net_profit=base_cf / UNIT,
        shares=shares / UNIT,
        fair_price=scenario.fair_price,
        latest_price=latest_price,
        trade_date=trade_date,
        undervaluation_pct=(
            (scenario.fair_price - latest_price) / scenario.fair_price
            if scenario.fair_price > 0
            else 0.0
        ),
        return_since_publish_pct=(latest_price / publish_price - 1) * 100,
    )


# ---- 主流程 ----
def main() -> None:
    excel_example = compute_scenario(
        name="Excel示例",
        base_cf=747.34,
        shares=12.56,
        growth=DEFAULT_GROWTH,
        discount_rate=DEFAULT_DISCOUNT,
        terminal_growth=TERMINAL_GROWTH,
        years=PROJECTION_YEARS,
    )
    print(
        "Excel示例场景：基准现金流 747.34，股本 12.56，增长率/贴现率 8%，终值增速 0%，"
        f"合理价格 = {excel_example.fair_price:.2f}"
    )

    valuations: List[ValuationSummary] = []
    for financial_csv in iter_general_enterprise_csv(FINANCIAL_ROOT):
        summary = evaluate_stock(financial_csv)
        if summary is not None:
            valuations.append(summary)

    if not valuations:
        print("未找到可用的一般企业财务数据或对应行情数据。")
        return

    undervalued = [item for item in valuations if item.undervaluation_pct > 0]
    undervalued.sort(key=lambda item: item.undervaluation_pct, reverse=True)

    output_path = Path("valuation_results.csv")
    if undervalued:
        records = [
            {
                "code": item.code,
                "name": item.name or "-",
                "trade_date": item.trade_date.date(),
                "report_date": item.report_date.date(),
                "publish_date": item.publish_date.date(),
                "publish_trade_date": item.publish_trade_date.date(),
                "profit_亿元": round(item.net_profit, 4),
                "shares_亿股": round(item.shares, 4),
                "fair_price": round(item.fair_price, 4),
                "latest_price": round(item.latest_price, 4),
                "undervaluation_pct": round(item.undervaluation_pct * 100, 2),
                "return_since_publish_pct": round(item.return_since_publish_pct, 2),
            }
            for item in undervalued
        ]
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(
            columns=[
                "code",
                "name",
                "trade_date",
                "report_date",
                "publish_date",
                "publish_trade_date",
                "profit_亿元",
                "shares_亿股",
                "fair_price",
                "latest_price",
                "undervaluation_pct",
                "return_since_publish_pct",
            ]
        )

    df.to_csv(output_path, index=False)
    print(df)


if __name__ == "__main__":
    main()
