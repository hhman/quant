from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class MembershipRecord:
    code: str
    start: pd.Timestamp
    end: pd.Timestamp

    def format(self, delimiter: str = ",", uppercase: bool = True) -> str:
        code = self.code.upper() if uppercase else self.code
        return delimiter.join([code, str(self.start.date()), str(self.end.date())])


@dataclass
class ProcessedResult:
    symbol: str
    dataframe: pd.DataFrame
    filename: str


class CSVDataLoader:
    """将本地CSV股票数据清洗、对齐并输出标准化结果。"""

    REQUIRED_COLUMNS: Sequence[str] = (
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
        "沪深300成分股",
        "上证50成分股",
        "中证500成分股",
        "中证1000成分股",
        "中证2000成分股",
        "创业板指成分股",
        "新版申万一级行业名称",
    )

    PRICE_COLUMNS: Sequence[str] = ("开盘价", "最高价", "最低价", "收盘价", "前收盘价", "总市值")
    VOLUME_COLUMNS: Sequence[str] = ("成交量", "成交额")
    STATIC_COLUMNS: Sequence[str] = ("股票代码", "股票名称")

    MEMBERSHIP_COLUMNS: Sequence[str] = (
        "沪深300成分股",
        "上证50成分股",
        "中证500成分股",
        "中证1000成分股",
        "中证2000成分股",
        "创业板指成分股",
    )

    MEMBERSHIP_FILE_MAP: Dict[str, str] = {
        "沪深300成分股": "csi300",
        "上证50成分股": "sse50",
        "中证500成分股": "csi500",
        "中证1000成分股": "csi1000",
        "中证2000成分股": "csi2000",
        "创业板指成分股": "csi_gem",
    }
    INDUSTRY_COLUMN = "新版申万一级行业名称"
    COLUMN_RENAME_MAP: Dict[str, str] = {
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
    OUTPUT_COLUMNS: Sequence[str] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "industry_id",
        "total_mv",
        "factor",
    )

    def __init__(
        self,
        start_date: str,
        end_date: str,
        index_dir: str,
        stock_dir: str,
        output_dir: str,
    ) -> None:
        """初始化加载器所需的路径与日期范围。"""
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.index_dir = Path(index_dir)
        self.stock_dir = Path(stock_dir)
        self.output_dir = Path(output_dir)
        self.category_records: Dict[str, List[MembershipRecord]] = {}
        self.industry_mapping: Dict[str, int] = {}
        self.next_industry_id = 1
        self.industry_mapping_file = self.output_dir / "industry_mapping.json"
        self.features_dir = "features"
        self.membership_dir = "instruments"
        self.expected_category_files = [
            f"{self.membership_dir}/{filename}.txt" for filename in self.MEMBERSHIP_FILE_MAP.values()
        ]

    # Internal helpers -----------------------------------------------------------
    @staticmethod
    def _infer_index_date_column(columns: Iterable[str]) -> str:
        for candidate in ("candle_end_time", "date"):
            if candidate in columns:
                return candidate
        raise ValueError("指数文件缺少可识别的日期列（candle_end_time 或 date）。")

    def _iter_csv_files(self, directory: Path) -> Iterable[Path]:
        if not directory.exists():
            return []
        return sorted(directory.glob("*.csv"))

    def _load_index_dataframe(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if df.empty:
            return df
        df.columns = [str(col).strip().lower() for col in df.columns]
        date_column = self._infer_index_date_column(df.columns)
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
        df["symbol"] = df["symbol"].map(self._normalize_exchange_code)
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def _process_index_file(self, csv_path: Path) -> Optional[Tuple[str, pd.DataFrame]]:
        df = self._load_index_dataframe(csv_path)
        if df.empty or "date" not in df.columns:
            return None
        df = df.dropna(subset=["date"])
        mask = (df["date"] >= self.start_date) & (df["date"] <= self.end_date)
        df = df.loc[mask]
        if df.empty:
            return None
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
        df = df.set_index("date")
        df["industry_id"] = 0
        df["total_mv"] = 0
        df["factor"] = 1.0
        symbol_value = str(df["symbol"].iloc[0]) if "symbol" in df.columns else csv_path.stem
        columns = [col for col in self.OUTPUT_COLUMNS if col in df.columns]
        if len(columns) != len(self.OUTPUT_COLUMNS):
            missing = set(self.OUTPUT_COLUMNS) - set(columns)
            raise KeyError(f"指数文件缺少必要列: {missing}")
        df = df[columns]
        return symbol_value, df

    def _process_directory(
        self, directory: Path, handler: Callable[[Path], Optional[Tuple[str, pd.DataFrame]]]
    ) -> List[ProcessedResult]:
        results: List[ProcessedResult] = []
        for csv_path in self._iter_csv_files(directory):
            processed = handler(csv_path)
            if processed is None:
                continue
            symbol, processed_df = processed
            results.append(ProcessedResult(symbol=symbol, dataframe=processed_df, filename=f"{symbol}{csv_path.suffix}"))
        return results

    def _process_index_files(self) -> List[ProcessedResult]:
        return self._process_directory(self.index_dir, self._process_index_file)

    def _process_stock_files(self) -> List[ProcessedResult]:
        return self._process_directory(self.stock_dir, self._process_stock_file)

    def _write_processed_results(self, results: Sequence[ProcessedResult]) -> None:
        for item in results:
            output_path = self.output_dir / self.features_dir /item.filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            item.dataframe.to_csv(output_path, encoding="utf-8-sig", index_label="date")

    def _process_stock_file(self, csv_path: Path) -> Optional[Tuple[str, pd.DataFrame]]:
        """读取个股CSV并输出对齐后的DataFrame。"""
        raw_df = pd.read_csv(
            csv_path,
            encoding="gbk",
            skiprows=1,
            usecols=self.REQUIRED_COLUMNS,
            parse_dates=["交易日期"],
        )
        raw_df = raw_df.dropna(subset=["交易日期"])
        raw_df = raw_df.sort_values("交易日期")
        raw_df["股票代码"] = raw_df["股票代码"].map(self._normalize_exchange_code)

        mask = (raw_df["交易日期"] >= self.start_date) & (raw_df["交易日期"] <= self.end_date)
        raw_df = raw_df.loc[mask]
        if raw_df.empty:
            return None

        code = raw_df["股票代码"].iloc[0]
        raw_df = raw_df.drop_duplicates(subset="交易日期", keep="last")

        for column in self.MEMBERSHIP_COLUMNS:
            raw_df[column] = raw_df[column].map(self._normalize_membership_value)

        industry_series = (
            raw_df.set_index("交易日期")[self.INDUSTRY_COLUMN].sort_index().ffill().bfill()
        )
        raw_df = raw_df.drop(columns=[self.INDUSTRY_COLUMN])

        df = raw_df.set_index("交易日期")

        self._fill_static_columns(df)
        self._fill_price_columns(df)
        self._fill_membership_columns(df)
        df["factor"] = self._compute_adjust_factor(df)

        self._collect_membership_categories(code, df)
        aligned_industry = industry_series.reindex(df.index).ffill().bfill()
        df["industry_id"] = aligned_industry.map(self._map_industry_id)

        df = df.drop(columns=list(self.MEMBERSHIP_COLUMNS), errors="ignore")
        df = df.rename(columns=self.COLUMN_RENAME_MAP)
        symbol_value = code
        if "symbol" in df.columns and not df["symbol"].isna().all():
            symbol_value = str(df["symbol"].iloc[0])
        columns = [col for col in self.OUTPUT_COLUMNS if col in df.columns]
        if len(columns) != len(self.OUTPUT_COLUMNS):
            missing = set(self.OUTPUT_COLUMNS) - set(columns)
            raise KeyError(f"股票文件缺少必要列: {missing}")
        df = df.loc[:, columns]
        return symbol_value, df

    def _fill_static_columns(self, df: pd.DataFrame) -> None:
        """补齐股票代码、名称等不随时间变化的列。"""
        remaining = [col for col in self.STATIC_COLUMNS if col in df.columns]
        if remaining:
            df[remaining] = df[remaining].ffill().bfill()

    def _fill_price_columns(self, df: pd.DataFrame) -> None:
        """对价格/市值类列进行前向填充。"""
        df[list(self.PRICE_COLUMNS)] = df[list(self.PRICE_COLUMNS)].ffill()

    def _fill_membership_columns(self, df: pd.DataFrame) -> None:
        """补齐成分股布尔列，缺失视为0。"""
        for column in self.MEMBERSHIP_COLUMNS:
            df[column] = df[column].ffill().fillna(0).astype(int)


    def _compute_adjust_factor(self, df: pd.DataFrame) -> pd.Series:
        """基于前收/昨日收盘计算复权因子（后复权）。"""
        close = df["收盘价"]
        prev_close = df["前收盘价"]
        prev_close_shift = close.shift(1)

        ratio = pd.Series(1.0, index=df.index)
        valid_mask = prev_close.notna() & prev_close_shift.notna() & (prev_close != 0)
        # 后复权：使用昨日收盘/当日前收的倒数，未来价格被向过去基准上推。
        ratio.loc[valid_mask] = (prev_close_shift[valid_mask] / prev_close[valid_mask]).astype(float)
        ratio.iloc[0] = 1.0
        return ratio.cumprod()

    def _collect_membership_categories(self, code: str, df: pd.DataFrame) -> None:
        """遍历成分股列，按1的连续区间生成记录。"""
        for column, filename in self.MEMBERSHIP_FILE_MAP.items():
            if column not in df.columns:
                continue
            series = df[column].fillna(0).astype(int)
            if series.eq(0).all():
                continue
            self._collect_category(
                code,
                series,
                lambda value, fname=filename: f"{self.membership_dir}/{fname}.txt" if value == 1 else None,
            )

    def _collect_category(
        self,
        code: str,
        series: pd.Series,
        key_builder,
    ) -> None:
        """通用的区间切片逻辑，根据key_builder决定输出文件。"""
        if series.empty:
            return
        current_key: Optional[str] = None
        start_date: Optional[pd.Timestamp] = None
        prev_date: Optional[pd.Timestamp] = None
        for date, value in series.items():
            key = key_builder(value)
            if key != current_key:
                if current_key is not None and start_date is not None and prev_date is not None:
                    self.category_records.setdefault(current_key, []).append(
                        MembershipRecord(code=code, start=start_date, end=prev_date)
                    )
                current_key = key
                start_date = date if key is not None else None
            prev_date = date
        if current_key is not None and start_date is not None and prev_date is not None:
            self.category_records.setdefault(current_key, []).append(
                MembershipRecord(code=code, start=start_date, end=prev_date)
            )

    def _write_category_files(self) -> None:
        """将缓冲区写入磁盘，并清理无数据的股票池文件。"""
        written_files = set()
        for relative_path, records in self.category_records.items():
            if not records:
                continue
            output_path = self.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sorted_records = sorted(records, key=lambda item: (item.code, item.start))
            delimiter = (
                "\t"
                if relative_path.startswith(self.membership_dir)
                else ","
            )
            lines = [record.format(delimiter=delimiter) for record in sorted_records]
            output_path.write_text("\n".join(lines), encoding="utf-8")
            written_files.add(relative_path)

        for relative_path in self.expected_category_files:
            if relative_path in written_files:
                continue
            target = self.output_dir / relative_path
            if target.exists():
                target.unlink()

    @staticmethod
    def _normalize_membership_value(value) -> int:
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

    def _map_industry_id(self, value) -> int:
        """将行业名称映射为稳定的整数ID，空值返回0。"""
        if pd.isna(value) or value == "":
            return 0
        name = str(value).strip()
        if name not in self.industry_mapping:
            self.industry_mapping[name] = self.next_industry_id
            self.next_industry_id += 1
        return self.industry_mapping[name]

    def _save_industry_mapping(self) -> None:
        """将行业名称与ID映射写盘，便于复现与下游使用。"""
        self.industry_mapping_file.parent.mkdir(parents=True, exist_ok=True)
        sorted_items = sorted(self.industry_mapping.items(), key=lambda kv: kv[1])
        mapping_dict = {name: idx for name, idx in sorted_items}
        self.industry_mapping_file.write_text(json.dumps(mapping_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_exchange_code(code: str) -> str:
        """将交易所前缀转换为大写，不改变原有顺序，例如 sz000900 -> SZ000900。"""
        if pd.isna(code):
            return code
        code_str = str(code).strip()
        lower_code = code_str.lower()
        for prefix in ("sz", "sh", "bj"):
            if lower_code.startswith(prefix):
                return f"{prefix.upper()}{code_str[len(prefix):]}"
        return code_str

    # Public API -----------------------------------------------------------------
    def save_data(self) -> None:
        """执行读取、清洗并输出csv及分类txt。"""

        self.category_records = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stock_results = self._process_stock_files()
        self._write_processed_results(stock_results)
        index_results = self._process_index_files()
        self._write_processed_results(index_results)
        self._save_industry_mapping()
        self._write_category_files()


def parse_args() -> argparse.Namespace:
    """构建CLI参数解析器。"""
    parser = argparse.ArgumentParser(description="CSV数据加载器示例")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="开始日期，格式YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="结束日期，格式YYYY-MM-DD",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="index",
        help="指数/日历CSV目录路径",
    )
    parser.add_argument(
        "--stock-dir",
        type=str,
        default="stock",
        help="股票原始CSV目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口，串联参数与加载流程。"""
    args = parse_args()
    loader = CSVDataLoader(
        start_date=args.start_date,
        end_date=args.end_date,
        index_dir=args.index_dir,
        stock_dir=args.stock_dir,
        output_dir=args.output_dir,
    )
    loader.save_data()
    print(f"数据清洗完成，结果已写入: {loader.output_dir.resolve()}")


if __name__ == "__main__":
    main()
