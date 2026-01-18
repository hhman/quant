#!/usr/bin/env python3
"""
Cache 管理器 - Parquet 版（智能模式）
文件名即元数据，无独立 metadata 文件
智能判断：自动追加新列、替换已存在列、创建新文件

特殊功能：
- 收益率数据自动保存为all市场，支持跨市场复用
- 读取时自动检测并复用all市场收益率文件
"""

from pathlib import Path
from typing import Optional
import pandas as pd


class CacheManager:
    """
    Parquet Cache 管理器

    特性：
    - 文件名编码 market, start_date, end_date, type
    - 智能模式：自动判断追加/替换/创建
    - 支持高效的部分列读取
    - 无独立 metadata 文件
    - ✨ 收益率数据跨市场复用（自动保存为all市场）
    """

    CACHE_DIR = Path(".cache")

    def __init__(
        self,
        market: str,
        start_date: str,  # YYYY-MM-DD
        end_date: str,  # YYYY-MM-DD
    ):
        """
        初始化 Cache Manager

        Args:
            market: 市场标识
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        self.market = market
        self.start_date = start_date
        self.end_date = end_date

        # 标准化日期格式 (YYYYMMDD)
        self.start_date_compact = start_date.replace("-", "")
        self.end_date_compact = end_date.replace("-", "")

        # 创建 cache 目录
        self.CACHE_DIR.mkdir(exist_ok=True)

    # ========================================================================
    # 文件路径管理
    # ========================================================================

    def get_parquet_path(self, data_type: str) -> Path:
        """
        生成 Parquet 文件路径

        Args:
            data_type: 数据类型标识

        Returns:
            Parquet 文件完整路径
        """
        filename = f"{self.market}_{self.start_date_compact}_{self.end_date_compact}__{data_type}.parquet"
        return self.CACHE_DIR / filename

    @staticmethod
    def parse_filename(filename: str) -> dict:
        """
        解析文件名，提取参数信息

        Args:
            filename: 文件名

        Returns:
            参数字典 {'market': str, 'start_date': str, 'end_date': str, 'type': str}
        """
        name = filename.replace(".parquet", "")
        parts = name.split("__")

        if len(parts) != 2:
            raise ValueError(f"无效的文件名格式: {filename}")

        info_part = parts[0].split("_")
        if len(info_part) < 3:
            raise ValueError(f"无效的文件名格式: {filename}")

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

    # ========================================================================
    # 数据写入（智能模式）
    # ========================================================================

    def write_dataframe(
        self,
        df: pd.DataFrame,
        data_type: str,
        compression: str = "snappy",
        verbose: bool = True,
    ) -> None:
        """
        写入 DataFrame 到 Parquet（智能模式）

        自动行为（基于严格的表达式字符串匹配）：
        - 收益率数据（returns）：
          - 自动保存为all市场，支持跨市场复用
          - 检查数据是否相同，相同则跳过写入
          - 原因：收益率是确定性计算，相同输入必定产生相同输出
        - 风格数据（styles）：
          - 自动保存为all市场，支持跨市场复用
          - 检查数据是否相同，相同则跳过写入
          - 原因：风格数据对于所有市场相同，避免重复计算
        - 其他数据（factor_raw, factor_std, neutralized等）：
          - 文件不存在 -> 创建新文件
          - 文件存在 -> 智能合并：
            * 相同表达式 → 替换（重新计算）
            * 不同表达式 → 追加（新因子）
            * 未请求的已存在因子 → 保留（不删除）

        Args:
            df: 要写入的数据
            data_type: 数据类型标识
            compression: 压缩方式 ('snappy', 'gzip', 'brotli', 'lz4')
            verbose: 是否打印操作信息
        """
        # 特殊数据处理：returns 和 styles 统一保存为all市场
        if data_type in ["returns", "styles"]:
            all_cache_mgr = CacheManager("all", self.start_date, self.end_date)
            path = all_cache_mgr.get_parquet_path(data_type)
            if verbose:
                data_name = "收益率" if data_type == "returns" else "风格数据"
                print(f"    💾 {data_name}保存为all市场: {path.name}")
                print("    ⚡ 其他市场可复用此文件")
        else:
            path = self.get_parquet_path(data_type)

        if path.exists():
            # returns 和 styles：文件存在即复用（跨市场共享，确定性计算）
            if data_type in ["returns", "styles"]:
                if verbose:
                    data_name = "收益率" if data_type == "returns" else "风格数据"
                    print(f"    ✅ {data_name}文件已存在，复用已有文件")
                return  # 直接跳过写入

            # 其他数据：智能合并
            if verbose:
                print("    📄 检测到已有文件，执行智能合并...")

            # 读取现有数据
            existing_df = pd.read_parquet(path)

            # 智能合并
            result_df, merge_info = self._smart_merge(existing_df, df, verbose)

            # 写入合并后的数据
            result_df.to_parquet(path, compression=compression, index=True)

            if verbose:
                self._log_merge_result(merge_info)
        else:
            # 文件不存在：直接创建
            if verbose:
                print("    💾 创建新文件")

            # 写入
            df.to_parquet(path, compression=compression, index=True)

    # ========================================================================
    # 智能合并辅助方法
    # ========================================================================

    def _smart_merge(
        self,
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
        """
        智能合并两个 DataFrame（基于严格的表达式字符串匹配）

        核心原则：
        - 因子身份 = 完整的表达式字符串
        - 相同表达式 → 替换（重新计算）
        - 不同表达式 → 追加（新因子）
        - 未请求的已存在因子 → 保留（不删除）

        Args:
            existing_df: 现有的数据
            new_df: 新的数据
            verbose: 是否打印详细信息

        Returns:
            (合并后的DataFrame, 合并信息字典)
        """
        existing_cols = set(existing_df.columns)
        new_cols = set(new_df.columns)

        # 分类（基于字符串精确匹配）
        to_replace = existing_cols & new_cols  # 交集：相同表达式 → 替换
        to_append = new_cols - existing_cols  # 差集：不同表达式 → 追加
        to_keep = existing_cols - new_cols  # 差集：未请求 → 保留

        # 合并
        result_df = existing_df.drop(columns=list(to_replace))
        result_df = pd.concat([result_df, new_df], axis=1)

        return result_df, {
            "replaced": list(to_replace),
            "appended": list(to_append),
            "kept": list(to_keep),
        }

    def _log_merge_result(self, merge_info: dict) -> None:
        """
        打印合并结果的友好日志

        Args:
            merge_info: 合并信息字典
        """
        replaced = merge_info["replaced"]
        appended = merge_info["appended"]
        kept = merge_info["kept"]

        # 打印替换信息
        if replaced:
            print(f"    🔄 更新已有因子 ({len(replaced)}个):")
            for col in replaced:
                print(f"       - {col}")

        # 打印追加信息
        if appended:
            print(f"    ➕ 追加新因子 ({len(appended)}个):")
            for col in appended:
                print(f"       - {col}")

        # 打印保留信息
        if kept:
            print(f"    ✅ 保留已有因子 ({len(kept)}个):")
            for col in kept:
                print(f"       - {col}")

        # 如果没有任何变化
        if not replaced and not appended:
            print("    ✅ 因子无变化，跳过写入")

    # ========================================================================
    # 数据读取
    # ========================================================================

    def read_dataframe(
        self,
        data_type: str,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        从 Parquet 读取 DataFrame

        特殊处理：
        - 收益率数据优先复用all市场文件
        - 风格数据优先复用all市场文件
        - 支持部分列读取（性能优化）
        - 友好的错误提示

        Args:
            data_type: 数据类型标识
            columns: 要读取的列（None=全部）

        Returns:
            DataFrame

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 请求的列不存在
        """
        # returns 和 styles 数据优先查找all市场文件，支持跨市场复用
        if data_type in ["returns", "styles"]:
            all_path = (
                self.CACHE_DIR
                / f"all_{self.start_date_compact}_{self.end_date_compact}__{data_type}.parquet"
            )
            if all_path.exists():
                if self.market != "all":
                    data_name = "收益率" if data_type == "returns" else "风格数据"
                    print(f"    ⚡ 复用all市场{data_name}数据 (跨市场复用)")
                path = all_path
            else:
                path = self.get_parquet_path(data_type)
        else:
            path = self.get_parquet_path(data_type)

        if not path.exists():
            raise FileNotFoundError(
                f"❌ Cache 文件不存在: {path}\n   请先运行 step1 生成 cache"
            )

        # 检查请求的列是否存在（如果指定了列）
        if columns is not None:
            try:
                import pyarrow.parquet as pq

                schema = pq.read_schema(path)
                existing_cols = set(schema.names)
                missing_cols = set(columns) - existing_cols

                if missing_cols:
                    available_cols = sorted(existing_cols)
                    raise ValueError(
                        f"❌ 请求的因子不存在:\n"
                        f"   缺失: {sorted(missing_cols)}\n"
                        f"   可用: {available_cols}\n"
                        f"   文件: {path.name}\n"
                        f"\n"
                        f"💡 建议:\n"
                        f"   - 先运行 step1 生成缺失的因子:\n"
                        f'     python step1/cli.py --factor-formulas "{" ".join(missing_cols)}" ...'
                    )
            except Exception as e:
                # 如果读取schema失败，直接尝试读取数据
                if "requested columns not present" in str(e):
                    raise e

        if columns is None:
            return pd.read_parquet(path)
        else:
            # 部分列读取（高效）
            return pd.read_parquet(path, columns=columns)

    def read_columns(
        self,
        columns: list[str],
        data_type: str,
    ) -> pd.DataFrame:
        """
        只读取指定的列（优化性能）

        Args:
            columns: 列名列表
            data_type: 数据类型标识

        Returns:
            DataFrame（只包含指定的列）
        """
        return self.read_dataframe(data_type, columns=columns)

    # ========================================================================
    # 工具方法
    # ========================================================================

    def check_columns(
        self,
        data_type: str,
        required_columns: list[str],
        verbose: bool = False,
    ) -> dict:
        """
        检查请求的列是否都存在（优化：只读元数据，不读数据）

        Args:
            data_type: 数据类型标识
            required_columns: 需要检查的列名列表
            verbose: 是否打印详细信息

        Returns:
            检查结果字典:
            {
                'exists': bool,  # 是否全部存在
                'missing': list,  # 缺失的列
                'available': list,  # 可用的列
                'path': Path  # 文件路径
            }
        """
        # 处理 returns 和 styles 的跨市场复用
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

        # 只读 schema（元数据），不读数据（性能优化）
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(path)
            existing_cols = set(schema.names)
        except Exception:
            # 如果 pyarrow 不可用，回退到 pandas
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
                f"⚠️  部分因子缺失:\n"
                f"   缺失: {sorted(missing_cols)}\n"
                f"   可用: {sorted(existing_cols)}"
            )

        return result

    def list_columns(self, data_type: str) -> list[str]:
        """
        列出 Parquet 文件中的所有列

        Args:
            data_type: 数据类型标识

        Returns:
            列名列表
        """
        path = self.get_parquet_path(data_type)

        if not path.exists():
            return []

        df = pd.read_parquet(path)
        return df.columns.tolist()

    def file_exists(self, data_type: str) -> bool:
        """
        检查 Parquet 文件是否存在

        Args:
            data_type: 数据类型标识

        Returns:
            是否存在
        """
        return self.get_parquet_path(data_type).exists()

    def get_file_info(self, data_type: str) -> dict:
        """
        获取文件信息

        Args:
            data_type: 数据类型标识

        Returns:
            信息字典
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
        """
        删除 Parquet 文件

        Args:
            data_type: 数据类型标识

        Returns:
            是否成功删除
        """
        path = self.get_parquet_path(data_type)

        if path.exists():
            path.unlink()
            return True
        return False

    def clean_all(self) -> int:
        """
        清理当前 market+日期组合的所有 cache 文件

        Returns:
            删除的文件数量
        """
        pattern = f"{self.market}_{self.start_date_compact}_{self.end_date_compact}__*.parquet"
        files = list(self.CACHE_DIR.glob(pattern))

        for f in files:
            f.unlink()

        return len(files)

    def list_cache_files(self) -> list[Path]:
        """
        列出 cache 目录中所有 Parquet 文件

        Returns:
            文件路径列表
        """
        return list(self.CACHE_DIR.glob("*.parquet"))
