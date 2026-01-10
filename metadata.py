#!/usr/bin/env python3
"""
Cache元数据管理
支持智能的子集匹配和数据选取
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import json
from pathlib import Path
from datetime import datetime


@dataclass
class CacheMetadata:
    """Cache元数据 - 记录cache生成时的完整配置"""
    market: str
    start_date: str
    end_date: str

    factor_formulas: List[str]  # 因子表达式列表
    periods: Dict[str, int]

    generated_at: str
    stage: str

    def save(self, path: Path):
        """
        保存元数据到JSON文件

        Args:
            path: 保存路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'CacheMetadata':
        """
        从JSON文件加载元数据

        Args:
            path: 元数据文件路径

        Returns:
            CacheMetadata实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def contains_request(self, request: dict) -> Tuple[bool, str, dict]:
        """
        检查cache是否包含请求的数据
        支持子集匹配：cache有101个因子，step2可以只分析其中1个

        返回: (是否包含, 错误信息, 匹配信息)

        规则：
        1. market必须完全一致
        2. 日期范围必须在cache内（或使用cache的默认范围）
        3. 请求的因子必须是cache因子的子集（或使用cache的所有因子）
        4. 请求的周期必须是cache周期的子集（或使用cache的所有周期）
        """
        errors = []
        match_info = {}

        # 1. Market校验（必须完全一致）
        if self.market != request.get('market'):
            errors.append(
                f"市场不匹配: cache={self.market}, "
                f"请求={request.get('market')}"
            )
            return False, errors[0], {}

        match_info['cache_market'] = self.market

        # 2. 日期范围校验（支持子集）
        cache_start = datetime.fromisoformat(self.start_date)
        cache_end = datetime.fromisoformat(self.end_date)

        # 默认使用cache的时间范围
        req_start = cache_start
        req_end = cache_end

        # 如果请求指定了时间范围，检查是否在cache内
        if request.get('start_date'):
            req_start = datetime.fromisoformat(request['start_date'])

        if request.get('end_date'):
            req_end = datetime.fromisoformat(request['end_date'])

        # 检查是否在cache范围内
        if req_start < cache_start or req_end > cache_end:
            errors.append(
                f"日期范围超出cache: "
                f"cache[{self.start_date}, {self.end_date}], "
                f"请求[{req_start.date()}, {req_end.date()}]"
            )
            return False, errors[0], {}

        match_info['date_range'] = {
            'cache': (self.start_date, self.end_date),
            'request': (req_start.date().isoformat(), req_end.date().isoformat()),
            'needs_slice': (req_start != cache_start or req_end != cache_end)
        }

        # 3. 因子校验（支持子集）
        cache_factors = set(self.factor_formulas)
        req_factors = cache_factors  # 默认使用所有因子

        if request.get('factor_formulas'):
            req_factors = set(request['factor_formulas'])

        # 检查是否是子集
        if not req_factors.issubset(cache_factors):
            missing = req_factors - cache_factors
            errors.append(
                f"cache缺少因子: {missing}\n"
                f"cache中的因子: {list(cache_factors)[:10]}..."
            )
            return False, errors[0], {}

        match_info['factors'] = {
            'cache_all': list(cache_factors),
            'request_subset': list(req_factors),
            'needs_filter': len(req_factors) < len(cache_factors)
        }

        # 4. 周期校验（支持子集）
        cache_periods = set(self.periods.keys())
        req_periods = cache_periods  # 默认使用所有周期

        if request.get('periods'):
            req_periods = set(request['periods'].keys())

        # 检查是否是子集
        if not req_periods.issubset(cache_periods):
            missing = req_periods - cache_periods
            errors.append(
                f"cache缺少周期: {missing}\n"
                f"cache中的周期: {list(cache_periods)}"
            )
            return False, errors[0], {}

        match_info['periods'] = {
            'cache_all': list(cache_periods),
            'request_subset': list(req_periods),
            'needs_filter': len(req_periods) < len(cache_periods)
        }

        return True, "", match_info


@dataclass
class Step0Metadata:
    """Step0元数据 - 记录数据预处理配置"""
    start_date: str
    end_date: str

    raw_data_dir: str
    stock_dir: str
    index_dir: str
    finance_dir: str
    qlib_dir: str
    qlib_src_dir: str

    generated_at: str
    stage: str  # "step0"

    def save(self, path: Path):
        """
        保存元数据到JSON文件

        Args:
            path: 保存路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'Step0Metadata':
        """
        从JSON文件加载元数据

        Args:
            path: 元数据文件路径

        Returns:
            Step0Metadata实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def get_qlib_dir(self) -> Path:
        """获取qlib数据目录路径"""
        return Path(self.qlib_dir)
