#!/usr/bin/env python3
"""
Step2: 行业市值中性化
功能：对因子数据进行行业和市值中性化处理
支持智能cache子集匹配
完全脱离qlib依赖
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cli_config import parse_common_args, normalize_args
from cache_manager import CacheManager
from factor_analysis import neutralize_industry_marketcap


def main() -> None:
    # 解析CLI参数
    args = parse_common_args()
    params = normalize_args(args, "step2")

    # 创建cache管理器
    cache_mgr = CacheManager(params)

    if params['dry_run']:
        print("🔍 模拟运行模式")
        print(f"  市场: {params['market']}")
        print(f"  Cache目录: {cache_mgr.cache_dir}")
        return

    print(f"\n🎯 Step2: 行业市值中性化")

    # 校验step1的cache（支持子集匹配）
    try:
        cache_meta, match_info = cache_mgr.load_and_validate_metadata("step1")
        cache_mgr.print_match_summary(match_info)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

    # 加载数据（从step1的新cache结构）
    print("📥 加载数据...")
    factor_std = cache_mgr.load_data_subset(
        cache_mgr.get_data_path("factor_standardized.parquet"),
        match_info
    )
    styles_df = cache_mgr.load_data_subset(
        cache_mgr.get_data_path("data_styles.parquet"),
        match_info
    )
    print(f"  ✓ 标准化因子: {factor_std.shape}")
    print(f"  ✓ 风格数据: {styles_df.shape}")

    # 提取因子列
    factor_cols = [col for col in factor_std.columns if col in params['factor_formulas']]

    # 检查必需的风格列
    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [col for col in factor_cols + required_style_cols if col not in styles_df.columns]
    if missing_cols:
        print(f"❌ 错误: 缺少列: {missing_cols}")
        sys.exit(1)

    # 合并因子和风格数据
    data_for_neutralize = pd.concat([factor_std[factor_cols], styles_df[required_style_cols]], axis=1)

    print(f"  ✓ 因子列: {len(factor_cols)}个")
    print(f"  ✓ 风格列: {required_style_cols}")

    # 执行中性化
    print("⚙️  执行行业市值中性化...")
    result_list = []
    for dt in data_for_neutralize.index.get_level_values("datetime").unique():
        daily_group = (
            data_for_neutralize.xs(dt, level="datetime")
            .assign(datetime=dt)
            .set_index("datetime", append=True)
            .reorder_levels(["instrument", "datetime"])
        )
        daily = neutralize_industry_marketcap(
            daily_group,
            factor_list=factor_cols,
            total_mv_col="$total_mv",
            industry_col="$industry",
            float_mv_col="$float_mv",
        )
        result_list.append(daily)

    if result_list:
        result = (
            pd.concat(result_list, axis=0)
            .sort_index(level=["instrument", "datetime"])
        )
        result.to_parquet(
            cache_mgr.get_data_path("factor_行业市值中性化.parquet"),
            compression="snappy",
            index=True
        )
        print(f"  ✓ 保存: factor_行业市值中性化.parquet ({result.shape})")
        print(result)

        # 保存元数据
        cache_mgr.save_metadata("step2")
        print(f"  ✓ 保存: step2_metadata.json")

        print(f"\n✅ Step2完成!")
    else:
        print("❌ 错误: 中性化失败，结果为空")
        sys.exit(1)


if __name__ == "__main__":
    main()
