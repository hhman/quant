from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# =============================================================================
#
# =============================================================================

REQUIRED_COLUMNS = ("stock_code", "report_date", "publish_date")

META_COLUMNS = (
    "stock_code",
    "statement_format",
    "report_date",
    "publish_date",
    "report_type",
)

OUTPUT_COLUMNS = ("stock_code", "date", "period", "field", "value")

INPUT_ENCODING = "gbk"
OUTPUT_ENCODING = "utf-8-sig"


# =============================================================================
#
# =============================================================================


def normalize_exchange_code(code: str) -> str:
    """"""
    if pd.isna(code):
        return code
    code_str = str(code).strip()
    lower = code_str.lower()
    for prefix in ("sz", "sh", "bj"):
        if lower.startswith(prefix):
            return f"{prefix.upper()}{code_str[len(prefix) :]}"
    return code_str


def compute_period(report_dates: pd.Series) -> pd.Series:
    """YYYYQ"""
    ts = pd.to_datetime(report_dates.astype(str), format="%Y%m%d", errors="coerce")
    quarter = ((ts.dt.month - 1) // 3 + 1).astype("Int64")
    return ts.dt.year.astype("Int64") * 100 + quarter


def pivot_to_long(
    df: pd.DataFrame,
    value_cols: Iterable[str],
) -> pd.DataFrame:
    """field/value"""
    melted = df.melt(
        id_vars=["stock_code", "date", "period"],
        value_vars=list(value_cols),
        var_name="field",
        value_name="value",
    )
    melted["date"] = pd.to_datetime(melted["date"], errors="coerce")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["date", "period", "value"])
    melted["period"] = melted["period"].astype(int)
    melted["date"] = melted["date"].dt.strftime("%Y-%m-%d")
    melted["stock_code"] = melted["stock_code"].map(normalize_exchange_code)
    return melted[list(OUTPUT_COLUMNS)]


def iter_csv_files(input_dir: Path) -> list[Path]:
    """CSV"""
    if input_dir.is_file():
        return [input_dir]
    if input_dir.is_dir():
        return sorted(input_dir.rglob("*.csv"))
    return []


# =============================================================================
#
# =============================================================================


def process_financial_data(
    start_date: str,
    end_date: str,
    finance_dir: str,
    output_dir: str,
) -> None:
    """dump_pit.py

    Parameters:
    -----------
    start_date : str
         (YYYY-MM-DD)
    end_date : str
         (YYYY-MM-DD)
    finance_dir : str
         CSV
    output_dir : str

    """
    input_path = Path(finance_dir)
    output_path = Path(output_dir)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    output_path.mkdir(parents=True, exist_ok=True)

    for csv_file in iter_csv_files(input_path):
        df = pd.read_csv(
            csv_file, encoding=INPUT_ENCODING, skiprows=1, low_memory=False
        )
        if df.empty:
            continue
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise KeyError(f"{csv_file} : {missing}")

        df = df.sort_values("publish_date")
        df["period"] = compute_period(df["report_date"])
        df["date"] = pd.to_datetime(df["publish_date"], errors="coerce")
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        df = df.loc[mask]
        if df.empty:
            continue

        value_cols = [
            c
            for c in df.columns
            if c not in META_COLUMNS and c not in {"period", "date"}
        ]
        if not value_cols:
            continue

        long_df = pivot_to_long(df, value_cols)
        long_df = long_df.sort_values(["stock_code", "date", "field", "period"])
        long_df = long_df.drop_duplicates(
            subset=["stock_code", "date", "period", "field", "value"]
        )

        for code, sub in long_df.groupby("stock_code"):
            out_path = output_path / f"{code}.csv"
            sub[list(OUTPUT_COLUMNS)].to_csv(
                out_path, index=False, encoding=OUTPUT_ENCODING
            )

    print(" ")
    print(f"   : {output_path}")
