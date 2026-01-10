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

from cli_config import parse_common_args, normalize_args
from cache_manager import CacheManager
from factor_analysis import ext_out_3std, z_score


def main():
    # 解析CLI参数
    args = parse_common_args()
    params = normalize_args(args, "step1")

    # 创建cache管理器
    cache_mgr = CacheManager(params)

    if params['dry_run']:
        print("🔍 模拟运行模式")
        print(f"  市场: {params['market']}")
        print(f"  日期: [{params['start_date']}, {params['end_date']}]")
        print(f"  因子: {params['factor_formulas']}")
        print(f"  周期: {list(params['periods'].keys())}")
        print(f"  Cache目录: {cache_mgr.cache_dir}")
        return

    # 初始化qlib
    print(f"📊 初始化Qlib: {params['provider_uri']}")
    qlib.init(provider_uri=params['provider_uri'], region=REG_CN)

    print(f"\n📈 Step1: 数据提取与预处理")
    print(f"  市场: {params['market']}")
    print(f"  日期: [{params['start_date']}, {params['end_date']}]")
    print(f"  因子: {len(params['factor_formulas'])}个")
    for i, formula in enumerate(params['factor_formulas'][:5], 1):
        print(f"    {i}. {formula}")
    if len(params['factor_formulas']) > 5:
        print(f"    ... (共{len(params['factor_formulas'])}个)")
    print(f"  周期: {list(params['periods'].keys())}")

    instruments = D.instruments(market=params['market'])

    # 提取因子数据（直接使用CLI传入的表达式）
    factor_df = D.features(
        instruments=instruments,
        fields=params['factor_formulas'],
        start_time=params['start_date'],
        end_time=params['end_date'],
        freq="day",
    )
    # 使用表达式作为列名（或者你可以用简洁的别名）
    factor_df.columns = params['factor_formulas']

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
    factor_raw.to_parquet(
        cache_mgr.get_data_path("factor_raw.parquet"),
        compression="snappy",
        index=True
    )
    print(f"  ✓ 保存: factor_raw.parquet ({factor_raw.shape})")

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
    factor_df.to_parquet(
        cache_mgr.get_data_path("factor_standardized.parquet"),
        compression="snappy",
        index=True
    )
    print(f"  ✓ 保存: factor_standardized.parquet ({factor_df.shape})")

    # 提取收益率数据
    print("  提取收益率数据...")
    ret_map = {
        f"ret_{label}": f"Ref($close, -{lag})/$close - 1"
        for label, lag in params['periods'].items()
    }
    ret_df = D.features(
        instruments=instruments,
        fields=ret_map.values(),
        start_time=params['start_date'],
        end_time=params['end_date'],
        freq="day",
    )
    ret_df.columns = ret_map.keys()

    # 3. 保存收益率数据
    ret_df.to_parquet(
        cache_mgr.get_data_path("data_returns.parquet"),
        compression="snappy",
        index=True
    )
    print(f"  ✓ 保存: data_returns.parquet ({ret_df.shape})")

    # ========== 提取风格数据 ==========
    print("  提取风格数据（市值、行业等）...")

    # 总市值
    total_mv = D.features(
        instruments=instruments,
        fields=["$total_mv"],
        start_time=params['start_date'],
        end_time=params['end_date'],
        freq="day"
    )
    total_mv["$total_mv"] = (
        total_mv
        .groupby(level="instrument")["$total_mv"]
        .ffill()
        .bfill()
    )

    # 行业分类
    industry = D.features(
        instruments=instruments,
        fields=["$industry"],
        start_time=params['start_date'],
        end_time=params['end_date'],
        freq="day"
    )
    industry["$industry"] = (
        industry
        .groupby(level="instrument")["$industry"]
        .ffill()
        .bfill()
    )

    # 流通市值
    float_mv = D.features(
        instruments=instruments,
        fields=["$float_mv"],
        start_time=params['start_date'],
        end_time=params['end_date'],
        freq="day"
    )
    float_mv["$float_mv"] = (
        float_mv
        .groupby(level="instrument")["$float_mv"]
        .ffill()
        .bfill()
    )

    # 4. 保存风格数据（合并成一个文件）
    styles_df = pd.concat([total_mv, industry, float_mv], axis=1)
    styles_df.to_parquet(
        cache_mgr.get_data_path("data_styles.parquet"),
        compression="snappy",
        index=True
    )
    print(f"  ✓ 保存: data_styles.parquet ({styles_df.shape})")
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

    # 因子数据分布统计
    print(f"\n📈 因子数据分布统计（标准化后）:")
    factor_stats = pd.DataFrame({
        '均值': factor_df.mean(),
        '标准差': factor_df.std(),
        '最小值': factor_df.min(),
        '最大值': factor_df.max(),
        '缺失率': factor_df.isna().mean()
    })
    print(factor_stats.head(10))  # 只显示前10个因子
    if len(factor_stats) > 10:
        print(f"  ... (共{len(factor_stats)}个因子)")

    # 保存元数据
    cache_mgr.save_metadata("step1")
    print(f"  ✓ 保存: step1_metadata.json")

    print(f"\n✅ Step1完成!")
    print(f"   Cache目录: {cache_mgr.cache_dir}")
    print(f"   输出文件: factor_raw.parquet, factor_standardized.parquet, data_returns.parquet, data_styles.parquet")


if __name__ == "__main__":
    main()
