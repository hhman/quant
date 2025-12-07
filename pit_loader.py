from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


class PITDataLoader:
    """将宽表财务数据透视为 dump_pit.py 所需的长表格式。"""

    REQUIRED_COLUMNS: Sequence[str] = ("stock_code", "report_date", "publish_date")
    META_COLUMNS: Sequence[str] = ("stock_code", "statement_format", "report_date", "publish_date", "抓取时间")
    OUTPUT_COLUMNS: Sequence[str] = ("stock_code", "date", "period", "field", "value")
    INPUT_ENCODING = "gbk"
    OUTPUT_ENCODING = "utf-8-sig"

    def __init__(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        start_date: str,
        end_date: str,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) / "financial"
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.encoding = self.INPUT_ENCODING
        self.output_encoding = self.OUTPUT_ENCODING

    # Internal helpers -----------------------------------------------------------
    @staticmethod
    def _normalize_exchange_code(code: str) -> str:
        """规范股票代码字符串，统一交易所前缀大小写。"""
        if pd.isna(code):
            return code
        code_str = str(code).strip()
        lower = code_str.lower()
        for prefix in ("sz", "sh", "bj"):
            if lower.startswith(prefix):
                return f"{prefix.upper()}{code_str[len(prefix):]}"
        return code_str

    @staticmethod
    def _compute_period(report_dates: pd.Series) -> pd.Series:
        """将报告期日期转换为季度编码 YYYYQ。"""
        ts = pd.to_datetime(report_dates.astype(str), format="%Y%m%d", errors="coerce")
        quarter = ((ts.dt.month - 1) // 3 + 1).astype("Int64")
        return ts.dt.year.astype("Int64") * 100 + quarter

    def _pivot_to_long(self, df: pd.DataFrame, value_cols: Iterable[str]) -> pd.DataFrame:
        """宽表转长表：展开所有财务字段为 field/value。"""
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
        melted["stock_code"] = melted["stock_code"].map(self._normalize_exchange_code)
        return melted[list(self.OUTPUT_COLUMNS)]

    def _iter_csv_files(self) -> List[Path]:
        """遍历输入路径，返回待处理的 CSV 列表。"""
        if self.input_dir.is_file():
            return [self.input_dir]
        if self.input_dir.is_dir():
            return sorted(self.input_dir.rglob("*.csv"))
        return []

    # Public API -----------------------------------------------------------------
    def save_data(self) -> None:
        """读取财务宽表，透视并输出按股票划分的 PIT 长表 CSV。"""
        frames: List[pd.DataFrame] = []
        for csv_file in self._iter_csv_files():
            df = pd.read_csv(csv_file, encoding=self.encoding, skiprows=1, low_memory=False)
            if df.empty:
                continue
            missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                raise KeyError(f"{csv_file} 缺少必要列: {missing}")

            df = df.sort_values("publish_date")
            df["period"] = self._compute_period(df["report_date"])
            df["date"] = pd.to_datetime(df["publish_date"], errors="coerce")
            mask = (df["date"] >= self.start_date) & (df["date"] <= self.end_date)
            df = df.loc[mask]
            if df.empty:
                continue

            value_cols = [c for c in df.columns if c not in self.META_COLUMNS and c not in {"period", "date"}]
            if not value_cols:
                continue
            frames.append(self._pivot_to_long(df, value_cols))

        if not frames:
            return

        long_df = pd.concat(frames, ignore_index=True)
        # 按公告日主序排列，便于还原披露时间线
        long_df = long_df.sort_values(["stock_code", "date", "field", "period"])
        # 仅去除完全重复的行，保留所有公告日的修订
        long_df = long_df.drop_duplicates(subset=["stock_code", "date", "period", "field", "value"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for code, sub in long_df.groupby("stock_code"):
            out_path = self.output_dir / f"{code}.csv"
            sub[list(self.OUTPUT_COLUMNS)].to_csv(out_path, index=False, encoding=self.output_encoding)


def parse_args() -> argparse.Namespace:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="将财务宽表转为 PIT 长表（适配 dump_pit.py）")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="开始日期，格式YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="结束日期，格式YYYY-MM-DD")
    parser.add_argument("--input-dir", type=Path, default=Path("finance"), help="原始财务 CSV 目录")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="输出目录，按股票代码分文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = PITDataLoader(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    loader.save_data()
    print(f"数据清洗完成，结果已写入: {loader.output_dir.resolve()}")


if __name__ == "__main__":
    main()
