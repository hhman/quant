#!/usr/bin/env python3
"""Parquet缓存管理器，支持数据缓存、合并、读取和元数据管理。"""

from pathlib import Path
from typing import Optional
import pandas as pd


class CacheManager:
    """Parquet缓存管理器。

    功能：
    - 根据market、start_date、end_date、type管理缓存文件
    - 支持智能合并（新增列、替换列）
    - 元数据管理和查询
    - 缓存文件清理
    """

    CACHE_DIR = Path(".cache")

    def __init__(
        self,
        market: str,
        start_date: str,  # YYYY-MM-DD
        end_date: str,  # YYYY-MM-DD
    ) -> None:
        """初始化缓存管理器。

        Args:
            market: 市场名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        self.market = market
        self.start_date = start_date
        self.end_date = end_date

        self.start_date_compact = start_date.replace("-", "")
        self.end_date_compact = end_date.replace("-", "")

        self.CACHE_DIR.mkdir(exist_ok=True)

    def get_parquet_path(self, data_type: str) -> Path:
        """获取Parquet文件路径。

        Args:
            data_type: 数据类型

        Returns:
            Parquet文件路径
        """
        filename = f"{self.market}_{self.start_date_compact}_{self.end_date_compact}__{data_type}.parquet"
        return self.CACHE_DIR / filename

    @staticmethod
    def parse_filename(filename: str) -> dict:
        """解析缓存文件名，提取元信息。

        Args:
            filename: 文件名

        Returns:
            包含market、start_date、end_date、type的字典
        """
        name = filename.replace(".parquet", "")
        parts = name.split("__")

        if len(parts) != 2:
            raise ValueError(f"文件名格式错误: {filename}")

        info_part = parts[0].split("_")
        if len(info_part) < 3:
            raise ValueError(f"文件名格式错误: {filename}")

        market = info_part[0]
        start_compact = info_part[1]
        end_compact = info_part[2]

        start_date = f"{start_compact[:4]}-{start_compact[4:6]}-{start_compact[6:8]}"
        end_date = f"{end_compact[:4]}-{end_compact[4:6]}-{end_compact[6:8]}"

        return {
            "market": market,
            "start_date": start_date,
            "end_date": end_date,
            "type": parts[1],
        }

    def write_dataframe(
        self,
        df: pd.DataFrame,
        data_type: str,
        compression: str = "snappy",
        verbose: bool = True,
    ) -> None:
        """将DataFrame写入Parquet缓存文件。

        处理逻辑：
        - returns和styles类型统一写入all市场
        - 如果文件已存在，智能合并（新增列、替换列、保留列）
        - factor_raw、factor_std、neutralized类型会检查是否需要合并

        Args:
            df: 要写入的DataFrame
            data_type: 数据类型
            compression: 压缩算法（snappy/gzip/brotli/lz4）
            verbose: 是否输出详细信息
        """
        if data_type in ["returns", "styles"]:
            all_cache_mgr = CacheManager("all", self.start_date, self.end_date)
            path = all_cache_mgr.get_parquet_path(data_type)
            if verbose:
                data_name = "收益率" if data_type == "returns" else "风格因子"
                print(f"[合并] 将{data_name}写入all市场: {path.name}")
                print("[合并] all市场逻辑：统一合并，不替换")
        else:
            path = self.get_parquet_path(data_type)

        if path.exists():
            if data_type in ["returns", "styles"]:
                if verbose:
                    data_name = "收益率" if data_type == "returns" else "风格因子"
                    print(f"[跳过] {data_name}已存在，跳过写入")
                return

            if verbose:
                print("[合并] 缓存文件已存在，开始智能合并...")

            existing_df = pd.read_parquet(path)
            result_df, merge_info = self._smart_merge(existing_df, df, verbose)
            result_df.to_parquet(path, compression=compression, index=True)

            if verbose:
                self._log_merge_result(merge_info)
        else:
            if verbose:
                print("[写入] 缓存文件不存在，创建新文件")

            df.to_parquet(path, compression=compression, index=True)

    def _smart_merge(
        self,
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
        """智能合并两个DataFrame。

        合并规则：
        - 相同列名 = 替换为新列
        - 新列 = 追加到结果
        - 只在旧数据的列 = 保留

        Args:
            existing_df: 已存在的DataFrame
            new_df: 新的DataFrame
            verbose: 是否输出详细信息

        Returns:
            (合并后的DataFrame, 合并信息字典)
        """
        existing_cols = set(existing_df.columns)
        new_cols = set(new_df.columns)

        to_replace = existing_cols & new_cols
        to_append = new_cols - existing_cols
        to_keep = existing_cols - new_cols

        result_df = existing_df.drop(columns=list(to_replace))
        result_df = pd.concat([result_df, new_df], axis=1)

        return result_df, {
            "replaced": list(to_replace),
            "appended": list(to_append),
            "kept": list(to_keep),
        }

    def _log_merge_result(self, merge_info: dict) -> None:
        """输出合并结果日志。

        Args:
            merge_info: 合并信息字典
        """
        replaced = merge_info["replaced"]
        appended = merge_info["appended"]
        kept = merge_info["kept"]

        if replaced:
            print(f"[合并] 替换列 ({len(replaced)}):")
            for col in replaced:
                print(f"       - {col}")

        if appended:
            print(f"[合并] 新增列 ({len(appended)}):")
            for col in appended:
                print(f"       - {col}")

        if kept:
            print(f"[合并] 保留列 ({len(kept)}):")
            for col in kept:
                print(f"       - {col}")

        if not replaced and not appended:
            print("[合并] 无需合并，数据未变更")

    def read_dataframe(
        self,
        data_type: str,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """从Parquet缓存读取DataFrame。

        读取逻辑：
        - returns和styles优先读取all市场
        - 支持指定列读取
        - 缺失列会提供明确的错误提示

        Args:
            data_type: 数据类型
            columns: 要读取的列，None表示读取全部

        Returns:
            DataFrame

        Raises:
            FileNotFoundError: 缓存文件不存在
            ValueError: 请求的列不存在
        """
        if data_type in ["returns", "styles"]:
            all_path = (
                self.CACHE_DIR
                / f"all_{self.start_date_compact}_{self.end_date_compact}__{data_type}.parquet"
            )
            if all_path.exists():
                if self.market != "all":
                    data_name = "收益率" if data_type == "returns" else "风格因子"
                    print(f"[提示] 从all市场读取{data_name}（统一市场）")
                path = all_path
            else:
                path = self.get_parquet_path(data_type)
        else:
            path = self.get_parquet_path(data_type)

        if not path.exists():
            raise FileNotFoundError(f"缓存文件不存在: {path}\n请先运行 step1 生成缓存")

        if columns is not None:
            try:
                import pyarrow.parquet as pq

                schema = pq.read_schema(path)
                existing_cols = set(schema.names)
                missing_cols = set(columns) - existing_cols

                if missing_cols:
                    available_cols = sorted(existing_cols)
                    raise ValueError(
                        f"请求的列不存在:\n"
                        f"   缺失列: {sorted(missing_cols)}\n"
                        f"   可用列: {available_cols}\n"
                        f"   文件: {path.name}\n"
                        f"\n解决方法:\n"
                        f"   - 运行 step1 生成缺失列:\n"
                        f'     python step1/cli.py --factor-formulas "{" ".join(missing_cols)}" ...'
                    )
            except Exception as e:
                if "requested columns not present" in str(e):
                    raise e

        if columns is None:
            return pd.read_parquet(path)
        else:
            return pd.read_parquet(path, columns=columns)

    def read_columns(
        self,
        columns: list[str],
        data_type: str,
    ) -> pd.DataFrame:
        """读取指定的列。

        Args:
            columns: 列名列表
            data_type: 数据类型

        Returns:
            DataFrame
        """
        return self.read_dataframe(data_type, columns=columns)

    def check_columns(
        self,
        data_type: str,
        required_columns: list[str],
        verbose: bool = False,
    ) -> dict:
        """检查所需列是否存在于缓存中。

        Args:
            data_type: 数据类型
            required_columns: 需要的列列表
            verbose: 是否输出详细信息

        Returns:
            检查结果字典:
            {
                'exists': bool,  # 是否全部存在
                'missing': list,  # 缺失的列
                'available': list,  # 可用的列
                'path': Path  # 文件路径
            }
        """
        if data_type in ["returns", "styles"]:
            all_path = (
                self.CACHE_DIR
                / f"all_{self.start_date_compact}_{self.end_date_compact}__{data_type}.parquet"
            )
            path = all_path if all_path.exists() else self.get_parquet_path(data_type)
        else:
            path = self.get_parquet_path(data_type)

        if not path.exists():
            return {
                "exists": False,
                "missing": required_columns,
                "available": [],
                "path": path,
            }

        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(path)
            existing_cols = set(schema.names)
        except Exception:
            df = pd.read_parquet(path)
            existing_cols = set(df.columns)

        missing_cols = set(required_columns) - existing_cols

        result = {
            "exists": len(missing_cols) == 0,
            "missing": sorted(missing_cols),
            "available": sorted(existing_cols),
            "path": path,
        }

        if verbose and missing_cols:
            print(
                f"列缺失:\n"
                f"   缺失: {sorted(missing_cols)}\n"
                f"   可用: {sorted(existing_cols)}"
            )

        return result

    def list_columns(self, data_type: str) -> list[str]:
        """列出缓存文件中的所有列。

        Args:
            data_type: 数据类型

        Returns:
            列名列表
        """
        path = self.get_parquet_path(data_type)

        if not path.exists():
            return []

        df = pd.read_parquet(path)
        return df.columns.tolist()

    def file_exists(self, data_type: str) -> bool:
        """检查Parquet缓存文件是否存在。

        Args:
            data_type: 数据类型

        Returns:
            是否存在
        """
        return self.get_parquet_path(data_type).exists()

    def get_file_info(self, data_type: str) -> dict:
        """获取缓存文件信息。

        Args:
            data_type: 数据类型

        Returns:
            文件信息字典
        """
        path = self.get_parquet_path(data_type)

        if not path.exists():
            return {"path": path, "exists": False}

        df = pd.read_parquet(path)
        stat = path.stat()

        return {
            "path": path,
            "exists": True,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "modified_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
        }

    def delete_file(self, data_type: str) -> bool:
        """删除Parquet缓存文件。

        Args:
            data_type: 数据类型

        Returns:
            是否成功删除
        """
        path = self.get_parquet_path(data_type)

        if path.exists():
            path.unlink()
            return True
        return False

    def clean_all(self) -> int:
        """清理当前market+日期范围的所有缓存文件。

        Returns:
            删除的文件数量
        """
        pattern = f"{self.market}_{self.start_date_compact}_{self.end_date_compact}__*.parquet"
        files = list(self.CACHE_DIR.glob(pattern))

        for f in files:
            f.unlink()

        return len(files)

    def list_cache_files(self) -> list[Path]:
        """列出缓存目录中的所有Parquet文件。

        Returns:
            文件路径列表
        """
        return list(self.CACHE_DIR.glob("*.parquet"))
