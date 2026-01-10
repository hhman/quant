#!/usr/bin/env python3
"""
Cache管理器 - 智能的数据加载和子集选取
"""
from pathlib import Path
from typing import Tuple
import pandas as pd
from datetime import datetime

from metadata import CacheMetadata


class CacheManager:
    """
    Cache文件管理器
    负责元数据管理、数据校验和智能子集选取
    """

    def __init__(self, params: dict):
        """
        初始化Cache管理器

        Args:
            params: 标准化的参数字典
        """
        self.params = params
        self.cache_dir = Path(params['cache_dir'])
        self.cache_dir.mkdir(exist_ok=True)

    def get_metadata_path(self, stage: str) -> Path:
        """
        获取元数据文件路径

        Args:
            stage: step名称，如 "step1", "step2"

        Returns:
            元数据文件路径
        """
        return self.cache_dir / f"{stage}_metadata.json"

    def get_data_path(self, filename: str) -> Path:
        """
        获取数据文件路径

        Args:
            filename: 数据文件名

        Returns:
            数据文件完整路径
        """
        return self.cache_dir / filename

    def save_metadata(self, stage: str):
        """
        保存当前step的元数据

        Args:
            stage: step名称
        """
        metadata = CacheMetadata(
            market=self.params['market'],
            start_date=self.params['start_date'],
            end_date=self.params['end_date'],
            factor_formulas=self.params['factor_formulas'],
            periods=self.params['periods'],
            generated_at=datetime.now().isoformat(),
            stage=stage
        )
        metadata.save(self.get_metadata_path(stage))
        return metadata

    def load_and_validate_metadata(self, stage: str) -> Tuple[CacheMetadata, dict]:
        """
        加载并校验元数据
        支持子集匹配：只要请求的数据在cache中，就能通过校验

        Args:
            stage: 要校验的step名称

        Returns:
            (cache元数据, 匹配信息)

        Raises:
            FileNotFoundError: 找不到元数据文件
            ValueError: 配置不满足要求
        """
        meta_path = self.get_metadata_path(stage)

        if not meta_path.exists():
            raise FileNotFoundError(
                f"找不到{stage}的cache元数据: {meta_path}\n"
                f"请先运行: python {stage}.py "
                f"--market {self.params['market']} "
                f"--factor-formulas ... --periods ..."
            )

        cache_meta = CacheMetadata.load(meta_path)

        # 检查是否包含请求的数据（支持子集）
        contains, error_msg, match_info = cache_meta.contains_request(self.params)

        if not contains:
            raise ValueError(
                f"{stage} cache配置不满足请求!\n"
                f"{error_msg}\n\n"
                f"Cache信息:\n"
                f"  市场: {cache_meta.market}\n"
                f"  日期: [{cache_meta.start_date}, {cache_meta.end_date}]\n"
                f"  因子: {len(cache_meta.factor_names)}个\n"
                f"  周期: {list(cache_meta.periods.keys())}\n"
                f"\n"
                f"请调整CLI参数或重新生成cache"
            )

        return cache_meta, match_info

    def load_data_subset(self, data_path: Path, match_info: dict) -> pd.DataFrame:
        """
        加载数据并根据match_info进行智能子集选取
        支持：日期截取、因子过滤、周期过滤

        Args:
            data_path: 数据文件路径
            match_info: 匹配信息字典

        Returns:
            处理后的DataFrame
        """
        df = pd.read_parquet(data_path)

        # 1. 日期范围截取
        if match_info['date_range']['needs_slice']:
            start, end = match_info['date_range']['request']
            df = df.loc[(slice(None), slice(start, end)), :]
            if self.params['verbose']:
                print(f"  ⚠️ 截取日期范围: [{start}, {end}]")

        # 2. 因子列过滤
        if match_info['factors']['needs_filter']:
            selected_factors = match_info['factors']['request_subset']

            # 分离因子列和非因子列（如ret_1d等）
            factor_cols = [col for col in df.columns if col in selected_factors]
            other_cols = [
                col for col in df.columns
                if col not in selected_factors and not col.startswith('ret_')
            ]

            # 检查是否有ret列需要保留
            ret_cols = [col for col in df.columns if col.startswith('ret_')]

            df = df[ret_cols + factor_cols + other_cols]
            if self.params['verbose']:
                print(f"  ⚠️ 选取因子: {selected_factors}")

        # 3. 收益率周期过滤（只对有ret_前缀的列生效）
        if match_info['periods']['needs_filter']:
            selected_periods = match_info['periods']['request_subset']
            ret_cols = [f"ret_{p}" for p in selected_periods]

            # 只保留请求的收益率列
            existing_ret_cols = [col for col in df.columns if col in ret_cols]
            if existing_ret_cols:
                non_ret_cols = [col for col in df.columns if not col.startswith('ret_')]
                df = df[non_ret_cols + existing_ret_cols]
                if self.params['verbose']:
                    print(f"  ⚠️ 选取周期: {selected_periods}")

        return df

    def print_match_summary(self, match_info: dict):
        """
        打印cache匹配摘要信息

        Args:
            match_info: 匹配信息字典
        """
        if not self.params['verbose']:
            return

        print(f"\n📦 Cache匹配信息:")
        print(f"  市场: {match_info['cache_market']}")

        date_info = match_info['date_range']
        print(f"  日期范围:")
        print(f"    Cache: [{date_info['cache'][0]}, {date_info['cache'][1]}]")
        print(f"    请求: [{date_info['request'][0]}, {date_info['request'][1]}]")
        if date_info['needs_slice']:
            print(f"    ⚠️  将进行日期截取")

        factor_info = match_info['factors']
        print(f"  因子:")
        print(f"    Cache总数: {len(factor_info['cache_all'])}个")
        print(f"    请求数量: {len(factor_info['request_subset'])}个")
        if factor_info['needs_filter']:
            print(f"    ⚠️  将进行因子过滤")
            # 显示请求的因子
            requested = factor_info['request_subset']
            if len(requested) <= 5:
                print(f"    请求的因子: {requested}")
            else:
                print(f"    请求的因子: {requested[:5]}... (共{len(requested)}个)")

        period_info = match_info['periods']
        print(f"  周期:")
        print(f"    Cache总数: {len(period_info['cache_all'])}个")
        print(f"    请求数量: {len(period_info['request_subset'])}个")
        if period_info['needs_filter']:
            print(f"    ⚠️  将进行周期过滤")
            print(f"    请求的周期: {period_info['request_subset']}")
        print()
