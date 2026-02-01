from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# =============================================================================
# 数据结构定义
# =============================================================================


@dataclass
class MembershipRecord:
    """成分股成员关系记录"""

    code: str
    start: pd.Timestamp
    end: pd.Timestamp

    def format(self, delimiter: str = ",", uppercase: bool = True) -> str:
        code = self.code.upper() if uppercase else self.code
        return delimiter.join([code, str(self.start.date()), str(self.end.date())])


@dataclass
class ProcessedResult:
    """处理后的数据结果"""

    symbol: str
    dataframe: pd.DataFrame
    filename: str


# =============================================================================
# 常量定义
# =============================================================================

REQUIRED_COLUMNS = (
    "股票代码",
    "股票名称",
    "交易日期",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "前收盘价",
    "成交量",
    "成交额",
    "总市值",
    "流通市值",
    "沪深300成分股",
    "上证50成分股",
    "中证500成分股",
    "中证1000成分股",
    "中证2000成分股",
    "创业板指成分股",
    "新版申万一级行业名称",
)

PRICE_COLUMNS = (
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "前收盘价",
    "总市值",
    "流通市值",
)

STATIC_COLUMNS = ("股票代码", "股票名称")

MEMBERSHIP_COLUMNS = (
    "沪深300成分股",
    "上证50成分股",
    "中证500成分股",
    "中证1000成分股",
    "中证2000成分股",
    "创业板指成分股",
)

MEMBERSHIP_FILE_MAP = {
    "沪深300成分股": "csi300",
    "上证50成分股": "sse50",
    "中证500成分股": "csi500",
    "中证1000成分股": "csi1000",
    "中证2000成分股": "csi2000",
    "创业板指成分股": "csi_gem",
}

INDUSTRY_COLUMN = "新版申万一级行业名称"

COLUMN_RENAME_MAP = {
    "股票代码": "symbol",
    "股票名称": "name",
    "交易日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "前收盘价": "prev_close",
    "成交量": "volume",
    "成交额": "amount",
    "总市值": "total_mv",
    "流通市值": "float_mv",
    "factor": "factor",
}

OUTPUT_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "industry",
    "total_mv",
    "float_mv",
    "factor",
)

MEMBERSHIP_DIR = "instruments"


# =============================================================================
# 辅助函数
# =============================================================================


def normalize_exchange_code(code: str) -> str:
    """将交易所前缀转换为大写，不改变原有顺序，例如 sz000900 -> SZ000900。"""
    if pd.isna(code):
        return code
    code_str = str(code).strip()
    lower_code = code_str.lower()
    for prefix in ("sz", "sh", "bj"):
        if lower_code.startswith(prefix):
            return f"{prefix.upper()}{code_str[len(prefix) :]}"
    return code_str


def normalize_membership_value(value) -> int:
    """将多种表示方式压缩为0/1。"""
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "y", "yes", "是"}:
            return 1
        if value in {"0", "false", "n", "no", "否"}:
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def infer_index_date_column(columns: Iterable[str]) -> str:
    """推断指数文件中的日期列名。"""
    for candidate in ("candle_end_time", "date"):
        if candidate in columns:
            return candidate
    raise ValueError("指数文件缺少可识别的日期列（candle_end_time 或 date）。")


def iter_csv_files(directory: Path) -> Iterable[Path]:
    """遍历目录下的 CSV 文件。"""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


def load_index_dataframe(csv_path: Path) -> pd.DataFrame:
    """加载指数 CSV 文件并标准化列名。"""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df.columns = [str(col).strip().lower() for col in df.columns]
    date_column = infer_index_date_column(df.columns)
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.rename(columns={date_column: "date"})
    symbol_column = "index_code" if "index_code" in df.columns else None
    if symbol_column is None:
        for candidate in ("symbol", "code"):
            if candidate in df.columns:
                symbol_column = candidate
                break
    if symbol_column:
        df["symbol"] = df[symbol_column].astype(str).str.strip()
    else:
        df["symbol"] = csv_path.stem
    df["symbol"] = df["symbol"].map(normalize_exchange_code)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def process_index_file(
    csv_path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> tuple[str, pd.DataFrame] | None:
    """处理单个指数文件。"""
    df = load_index_dataframe(csv_path)
    if df.empty or "date" not in df.columns:
        return None
    df = df.dropna(subset=["date"])
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask]
    if df.empty:
        return None
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")
    df["industry"] = 0
    df["total_mv"] = 0
    df["float_mv"] = 0
    df["factor"] = 1.0
    symbol_value = (
        str(df["symbol"].iloc[0]) if "symbol" in df.columns else csv_path.stem
    )
    columns = [col for col in OUTPUT_COLUMNS if col in df.columns]
    if len(columns) != len(OUTPUT_COLUMNS):
        missing = set(OUTPUT_COLUMNS) - set(columns)
        raise KeyError(f"指数文件缺少必要列: {missing}")
    df = df[columns]
    return symbol_value, df


def fill_static_columns(df: pd.DataFrame) -> None:
    """补齐股票代码、名称等不随时间变化的列（原地修改）。"""
    remaining = [col for col in STATIC_COLUMNS if col in df.columns]
    if remaining:
        df[remaining] = df[remaining].ffill().bfill()


def fill_price_columns(df: pd.DataFrame) -> None:
    """对价格/市值类列进行前向填充（原地修改）。"""
    df[list(PRICE_COLUMNS)] = df[list(PRICE_COLUMNS)].ffill()


def fill_membership_columns(df: pd.DataFrame) -> None:
    """补齐成分股布尔列，缺失视为0（原地修改）。"""
    for column in MEMBERSHIP_COLUMNS:
        df[column] = df[column].ffill().fillna(0).astype(int)


def compute_adjust_factor(df: pd.DataFrame) -> pd.Series:
    """计算后复权因子。

    后复权以开始日期价格为基准，因子从1.0开始累乘。
    Qlib将使用该因子计算后复权价格：后复权价 = 原始价 × 因子。
    """
    close = df["收盘价"]
    prev_close = df["前收盘价"]
    prev_close_shift = close.shift(1)

    ratio = pd.Series(1.0, index=df.index)
    valid_mask = prev_close.notna() & prev_close_shift.notna() & (prev_close != 0)
    ratio.loc[valid_mask] = (
        prev_close_shift[valid_mask] / prev_close[valid_mask]
    ).astype(float)
    ratio.iloc[0] = 1.0
    return ratio.cumprod()


def collect_category(
    code: str,
    series: pd.Series,
    key_builder,
    category_records: dict[str, list[MembershipRecord]],
) -> None:
    """通用的区间切片逻辑，根据key_builder决定输出文件。"""
    if series.empty:
        return
    current_key: str | None = None
    start_ts: pd.Timestamp | None = None
    prev_ts: pd.Timestamp | None = None
    for date, value in series.items():
        key = key_builder(value)
        if key != current_key:
            if current_key is not None and start_ts is not None and prev_ts is not None:
                category_records.setdefault(current_key, []).append(
                    MembershipRecord(code=code, start=start_ts, end=prev_ts)
                )
            current_key = key
            start_ts = date if key is not None else None
        prev_ts = date
    if current_key is not None and start_ts is not None and prev_ts is not None:
        category_records.setdefault(current_key, []).append(
            MembershipRecord(code=code, start=start_ts, end=prev_ts)
        )


def collect_membership_categories(
    code: str,
    df: pd.DataFrame,
    category_records: dict[str, list[MembershipRecord]],
) -> None:
    """遍历成分股列，按1的连续区间生成记录。"""
    for column, filename in MEMBERSHIP_FILE_MAP.items():
        if column not in df.columns:
            continue
        series = df[column].fillna(0).astype(int)
        if series.eq(0).all():
            continue
        collect_category(
            code,
            series,
            lambda value, fname=filename: f"{MEMBERSHIP_DIR}/{fname}.txt"
            if value == 1
            else None,
            category_records,
        )


def map_industry(
    value,
    industry_mapping: dict[str, int],
) -> int:
    """将行业名称映射为稳定的整数ID，空值返回0。"""
    if pd.isna(value) or value == "":
        return 0
    name = str(value).strip()
    if name not in industry_mapping:
        max_id = max(industry_mapping.values()) if industry_mapping else 0
        industry_mapping[name] = max_id + 1
    return industry_mapping[name]


def process_stock_file(
    csv_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    category_records: dict[str, list[MembershipRecord]],
    industry_mapping: dict[str, int],
) -> tuple[str, pd.DataFrame] | None:
    """处理单个股票文件。

    Parameters:
    -----------
    csv_path : Path
        CSV 文件路径
    start_date : pd.Timestamp
        起始日期
    end_date : pd.Timestamp
        结束日期
    category_records : dict
        成分股记录字典（会被修改）
    industry_mapping : dict
        行业映射字典（会被修改）

    Returns:
    --------
    (symbol, dataframe) | None
        股票代码和处理后的 DataFrame
    """
    raw_df = pd.read_csv(
        csv_path,
        encoding="gbk",
        skiprows=1,
        usecols=REQUIRED_COLUMNS,
        parse_dates=["交易日期"],
    )
    raw_df = raw_df.dropna(subset=["交易日期"])
    raw_df = raw_df.sort_values("交易日期")
    raw_df["股票代码"] = raw_df["股票代码"].map(normalize_exchange_code)

    mask = (raw_df["交易日期"] >= start_date) & (raw_df["交易日期"] <= end_date)
    raw_df = raw_df.loc[mask]
    if raw_df.empty:
        return None

    code = raw_df["股票代码"].iloc[0]
    raw_df = raw_df.drop_duplicates(subset="交易日期", keep="last")

    for column in MEMBERSHIP_COLUMNS:
        raw_df[column] = raw_df[column].map(normalize_membership_value)

    industry_series = (
        raw_df.set_index("交易日期")[INDUSTRY_COLUMN].sort_index().ffill().bfill()
    )
    raw_df = raw_df.drop(columns=[INDUSTRY_COLUMN])

    df = raw_df.set_index("交易日期")

    fill_static_columns(df)
    fill_price_columns(df)
    fill_membership_columns(df)
    df["factor"] = compute_adjust_factor(df)
    collect_membership_categories(code, df, category_records)

    aligned_industry = industry_series.reindex(df.index).ffill().bfill()
    df["industry"] = aligned_industry.map(lambda x: map_industry(x, industry_mapping))

    df = df.drop(columns=list(MEMBERSHIP_COLUMNS), errors="ignore")
    df = df.rename(columns=COLUMN_RENAME_MAP)
    symbol_value = code
    if "symbol" in df.columns and not df["symbol"].isna().all():
        symbol_value = str(df["symbol"].iloc[0])
    columns = [col for col in OUTPUT_COLUMNS if col in df.columns]
    if len(columns) != len(OUTPUT_COLUMNS):
        missing = set(OUTPUT_COLUMNS) - set(columns)
        raise KeyError(f"股票文件缺少必要列: {missing}")
    df = df.loc[:, columns]
    return symbol_value, df


def write_processed_results(
    results: list[ProcessedResult],
    output_dir: Path,
) -> None:
    """将处理结果写入 CSV 文件。"""
    for item in results:
        output_path = output_dir / item.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        item.dataframe.to_csv(output_path, encoding="utf-8-sig", index_label="date")


def write_category_files(
    category_records: dict[str, list[MembershipRecord]],
    output_dir: Path,
) -> None:
    """将成分股记录写入文件，并清理无数据的股票池文件。"""
    expected_category_files = [
        f"{MEMBERSHIP_DIR}/{filename}.txt" for filename in MEMBERSHIP_FILE_MAP.values()
    ]
    written_files = set()

    for relative_path, records in category_records.items():
        if not records:
            continue
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_records = sorted(records, key=lambda item: (item.code, item.start))
        delimiter = "\t" if relative_path.startswith(MEMBERSHIP_DIR) else ","
        lines = [record.format(delimiter=delimiter) for record in sorted_records]
        output_path.write_text("\n".join(lines), encoding="utf-8")
        written_files.add(relative_path)

    for relative_path in expected_category_files:
        if relative_path in written_files:
            continue
        target = output_dir / relative_path
        if target.exists():
            target.unlink()


def save_industry_mapping(
    industry_mapping: dict[str, int],
    output_dir: Path,
) -> None:
    """将行业名称与ID映射写盘，便于复现与下游使用。"""
    industry_mapping_file = output_dir / "industry_mapping.json"
    industry_mapping_file.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(industry_mapping.items(), key=lambda kv: kv[1])
    mapping_dict = {name: idx for name, idx in sorted_items}
    industry_mapping_file.write_text(
        json.dumps(mapping_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =============================================================================
# 核心函数
# =============================================================================


def process_stock_data(
    start_date: str,
    end_date: str,
    stock_dir: str,
    index_dir: str,
    output_dir: str,
) -> None:
    """处理股票日线数据：清洗、对齐并输出标准化结果。

    Parameters:
    -----------
    start_date : str
        起始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    stock_dir : str
        股票 CSV 数据目录
    index_dir : str
        指数 CSV 数据目录
    output_dir : str
        输出目录
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    index_path = Path(index_dir)
    stock_path = Path(stock_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    category_records: dict[str, list[MembershipRecord]] = {}
    industry_mapping: dict[str, int] = {}

    index_results = []
    for csv_path in iter_csv_files(index_path):
        result = process_index_file(csv_path, start_ts, end_ts)
        if result is not None:
            symbol, processed_df = result
            index_results.append(
                ProcessedResult(
                    symbol=symbol,
                    dataframe=processed_df,
                    filename=f"{symbol}{csv_path.suffix}",
                )
            )

    stock_results = []
    for csv_path in iter_csv_files(stock_path):
        result = process_stock_file(
            csv_path,
            start_ts,
            end_ts,
            category_records,
            industry_mapping,
        )
        if result is not None:
            symbol, processed_df = result
            stock_results.append(
                ProcessedResult(
                    symbol=symbol,
                    dataframe=processed_df,
                    filename=f"{symbol}{csv_path.suffix}",
                )
            )

    write_processed_results(stock_results, output_path)
    write_processed_results(index_results, output_path)
    save_industry_mapping(industry_mapping, output_path)
    write_category_files(category_records, output_path)

    print(
        f"处理完成: 股票 {len(stock_results)}, 指数 {len(index_results)}, 行业 {len(industry_mapping)}"
    )
    print(f"输出目录: {output_path}")
