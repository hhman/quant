#!/usr/bin/env python3
"""
Step1: 数据提取与预处理
功能：从qlib提取因子数据，进行去极值和标准化处理
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from utils.cache_manager import CacheManager
from core.factor_analysis import ext_out_3std, z_score


def calculate_factors(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """
    计算因子的核心逻辑函数

    Parameters:
    -----------
    market : str
        市场标识 (用于因子和风格数据)
    start_date : str
        起始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    factor_formulas : list[str]
        因子表达式列表
    provider_uri : str
        Qlib数据路径

    Returns:
    --------
    None
    """
    # 固定计算 1d, 1w, 1m 三个周期的收益率
    periods = {"1d": 1, "1w": 5, "1m": 20}
    # 计算最长收益率周期,用于延长时间范围
    max_lag = max(periods.values())  # 20天

    # 收益率数据使用all市场,避免幸存者偏差
    returns_market = "all"

    # 延长收益率数据的结束时间,确保尾部收益率计算完整
    # 缓冲期 = 最长周期 + 10天额外缓冲
    buffer_days = max_lag + 10
    end_date_extended = (
        pd.Timestamp(end_date) + pd.Timedelta(days=buffer_days)
    ).strftime("%Y-%m-%d")

    # 创建 cache manager (使用原始end_date)
    cache_mgr = CacheManager(market, start_date, end_date)

    # 初始化qlib
    print(f"📊 初始化Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    print("\n📈 Step1: 数据提取与预处理")
    print(f"  市场: {market}")
    print(f"  因子日期: [{start_date}, {end_date}]")
    print(
        f"  收益率日期: [{start_date}, {end_date_extended}] (延长{buffer_days}天缓冲)"
    )
    print(f"  因子: {len(factor_formulas)}个")
    for i, formula in enumerate(factor_formulas[:5], 1):
        print(f"    {i}. {formula}")
    if len(factor_formulas) > 5:
        print(f"    ... (共{len(factor_formulas)}个)")
    print(f"  周期: {list(periods.keys())}")

    instruments = D.instruments(market=market)

    # 提取因子数据（直接使用CLI传入的表达式）
    factor_df = D.features(
        instruments=instruments,
        fields=factor_formulas,
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    # 使用表达式作为列名（或者你可以用简洁的别名）
    factor_df.columns = factor_formulas

    # 丢弃全空列
    dropped_cols = [col for col in factor_df.columns if factor_df[col].isna().all()]
    if dropped_cols:
        print(f"  ⚠️  丢弃全空因子: {dropped_cols}")
        factor_df = factor_df.drop(columns=dropped_cols)

    valid_factor_cols = factor_df.columns.to_list()

    # ========== 保存4个独立cache文件 ==========
    print("  保存cache文件...")

    # 1. 保存原始因子
    factor_raw = factor_df.copy()
    cache_mgr.write_dataframe(factor_raw, "factor_raw")
    print(f"  ✓ 保存: factor_raw ({factor_raw.shape})")

    # 去极值
    print("  执行去极值处理...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: ext_out_3std(x, valid_factor_cols)
    )

    # 标准化
    print("  执行标准化处理...")
    factor_df = factor_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: z_score(x, valid_factor_cols)
    )

    # 2. 保存标准化因子
    cache_mgr.write_dataframe(factor_df, "factor_std")
    print(f"  ✓ 保存: factor_std ({factor_df.shape})")

    # 提取收益率数据
    print("  提取收益率数据...")
    print(f"    市场: {returns_market}")
    print(f"    时间范围: {start_date} ~ {end_date_extended}")
    print(f"    原因: 延长{buffer_days}天确保尾部收益率计算完整 (最长周期{max_lag}天)")

    ret_map = {
        f"ret_{label}": f"Ref($close, -{lag})/$close - 1"
        for label, lag in periods.items()
    }

    # 收益率使用all市场,避免幸存者偏差
    ret_instruments = D.instruments(market=returns_market)
    print(f"    使用 {returns_market} 计算收益率 (避免幸存者偏差)")

    ret_df = D.features(
        instruments=ret_instruments,
        fields=ret_map.values(),
        start_time=start_date,
        end_time=end_date_extended,  # 使用延长的结束日期
        freq="day",
    )
    ret_df.columns = ret_map.keys()

    print(
        f"    原始数据范围: {ret_df.index.get_level_values('datetime').min()} ~ {ret_df.index.get_level_values('datetime').max()}"
    )

    # 截取到用户指定的结束日期
    ret_df = ret_df[ret_df.index.get_level_values("datetime") <= pd.Timestamp(end_date)]
    print(
        f"    截取后范围: {ret_df.index.get_level_values('datetime').min()} ~ {ret_df.index.get_level_values('datetime').max()}"
    )

    # 3. 保存收益率数据
    cache_mgr.write_dataframe(ret_df, "returns")
    print(f"  ✓ 保存: returns ({ret_df.shape})")
    print(f"    - 股票数: {len(ret_df.index.get_level_values('instrument').unique())}")

    # ========== 提取风格数据 ==========
    print("  提取风格数据（市值、行业等）...")
    print("    💡 使用all市场提取，支持跨市场复用")

    # 使用 all 市场提取风格数据（覆盖全市场股票）
    all_instruments = D.instruments(market="all")

    # 总市值
    total_mv = D.features(
        instruments=all_instruments,
        fields=["$total_mv"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    total_mv["$total_mv"] = (
        total_mv.groupby(level="instrument")["$total_mv"].ffill().bfill()
    )

    # 行业分类
    industry = D.features(
        instruments=all_instruments,
        fields=["$industry"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    industry["$industry"] = (
        industry.groupby(level="instrument")["$industry"].ffill().bfill()
    )

    # 流通市值
    float_mv = D.features(
        instruments=all_instruments,
        fields=["$float_mv"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    float_mv["$float_mv"] = (
        float_mv.groupby(level="instrument")["$float_mv"].ffill().bfill()
    )

    # 4. 保存风格数据（合并成一个文件）
    styles_df = pd.concat([total_mv, industry, float_mv], axis=1)
    cache_mgr.write_dataframe(styles_df, "styles")
    print(f"  ✓ 保存: styles ({styles_df.shape})")
    print(f"    - 风格列: {len(styles_df.columns)}个 ($total_mv, $industry, $float_mv)")

    # 生成数据质量报告
    print("\n📊 数据质量摘要:")
    factor_missing_rate = factor_df.isna().mean().mean()
    ret_missing_rate = ret_df.isna().mean().mean()
    total_mv_missing_rate = total_mv["$total_mv"].isna().mean()
    industry_missing_rate = industry["$industry"].isna().mean()
    float_mv_missing_rate = float_mv["$float_mv"].isna().mean()

    print(f"  因子缺失率: {factor_missing_rate:.2%}")
    print(f"  收益率缺失率: {ret_missing_rate:.2%}")
    print(f"  总市值缺失率: {total_mv_missing_rate:.2%}")
    print(f"  行业缺失率: {industry_missing_rate:.2%}")
    print(f"  流通市值缺失率: {float_mv_missing_rate:.2%}")

    # 缺失率阈值检测
    MISSING_RATE_THRESHOLD = 0.8  # 80%阈值
    if total_mv_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(
            f"总市值缺失率 {total_mv_missing_rate:.2%} 超过阈值 {MISSING_RATE_THRESHOLD:.2%}"
        )
    if industry_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(
            f"行业缺失率 {industry_missing_rate:.2%} 超过阈值 {MISSING_RATE_THRESHOLD:.2%}"
        )
    if float_mv_missing_rate > MISSING_RATE_THRESHOLD:
        raise ValueError(
            f"流通市值缺失率 {float_mv_missing_rate:.2%} 超过阈值 {MISSING_RATE_THRESHOLD:.2%}"
        )

    # 因子数据分布统计
    print("\n📈 因子数据分布统计（标准化后）:")
    factor_stats = pd.DataFrame(
        {
            "均值": factor_df.mean(),
            "标准差": factor_df.std(),
            "最小值": factor_df.min(),
            "最大值": factor_df.max(),
            "缺失率": factor_df.isna().mean(),
        }
    )
    print(factor_stats.head(10))  # 只显示前10个因子
    if len(factor_stats) > 10:
        print(f"  ... (共{len(factor_stats)}个因子)")

    print("\n✅ Step1完成!")
    print(f"   Cache目录: {cache_mgr.CACHE_DIR}")
    print("   输出文件: factor_raw, factor_std, returns, styles")
