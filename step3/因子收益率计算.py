#!/usr/bin/env python3
"""
Step3: 因子收益回归
功能：通过回归分析因子的预测能力
支持智能cache子集匹配
完全脱离qlib依赖
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from cli_config import parse_common_args, normalize_args
from cache_manager import CacheManager
from factor_analysis import factor_return_industry_marketcap


def main() -> None:
    # 解析CLI参数
    args = parse_common_args()
    params = normalize_args(args, "step3")

    # 创建cache管理器
    cache_mgr = CacheManager(params)

    if params['dry_run']:
        print("🔍 模拟运行模式")
        print(f"  市场: {params['market']}")
        print(f"  Cache目录: {cache_mgr.cache_dir}")
        return

    print(f"\n📊 Step3: 因子收益回归")

    # 校验step1的cache（step3依赖step1的输出）
    try:
        cache_meta, match_info = cache_mgr.load_and_validate_metadata("step1")
        if params['verbose']:
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
    ret_df = cache_mgr.load_data_subset(
        cache_mgr.get_data_path("data_returns.parquet"),
        match_info
    )

    # 合并数据用于回归
    data = pd.concat([factor_std, styles_df, ret_df], axis=1)
    print(f"  ✓ 合并数据: {data.shape}")

    # 提取列
    factor_cols = [col for col in factor_std.columns if col in params['factor_formulas']]
    ret_cols = [col for col in ret_df.columns if col.startswith('ret_')]

    # 检查必需的风格列（step1提供的是$total_mv，不是$log_mv）
    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [col for col in factor_cols + ret_cols + required_style_cols if col not in data.columns]
    if missing_cols:
        print(f"❌ 错误: 缺少列: {missing_cols}")
        sys.exit(1)

    # 选择需要的列
    needed_cols = factor_cols + ret_cols + required_style_cols
    data = data[needed_cols]

    print(f"  ✓ 因子列: {len(factor_cols)}个")
    print(f"  ✓ 收益率列: {len(ret_cols)}个")
    print(f"  ✓ 风格列: {required_style_cols}")

    # 执行回归分析
    print("⚙️  执行因子收益回归...")
    coef_list = []
    t_list = []
    for dt in data.index.get_level_values("datetime").unique():
        daily_group = (
            data.xs(dt, level="datetime")
            .assign(datetime=dt)
            .set_index("datetime", append=True)
            .reorder_levels(["instrument", "datetime"])
        )
        result = factor_return_industry_marketcap(
            daily_group,
            factor_list=factor_cols,
            ret_list=ret_cols,
            total_mv_col="$total_mv",
            industry_col="$industry",
            float_mv_col="$float_mv",
        )
        coef_df, t_df = result
        coef_list.append(coef_df)
        t_list.append(t_df)

    if coef_list and t_list:
        coef_all = pd.concat(coef_list, axis=0)
        t_all = pd.concat(t_list, axis=0)

        coef_all.to_parquet(
            cache_mgr.get_data_path("factor_回归收益率.parquet"),
            compression="snappy",
            index=True
        )
        print(f"  ✓ 保存: factor_回归收益率.parquet")

        t_all.to_parquet(
            cache_mgr.get_data_path("factor_回归t值.parquet"),
            compression="snappy",
            index=True
        )
        print(f"  ✓ 保存: factor_回归t值.parquet")

        print(coef_all)
        print(t_all)

        # 生成汇总统计
        # 注意：汇总统计需要对整个时间序列计算，而不是单日的结果
        def _coef_summary(series: pd.Series) -> pd.Series:
            s = series.dropna()
            if s.empty:
                return pd.Series(dtype=float)
            mean = s.mean()
            std = s.std()
            t_test = mean / std * np.sqrt(len(s)) if std != 0 else np.nan
            return pd.Series({
                "因子收益率均值": mean,
                "因子收益率序列t检验": t_test,
            })

        def _t_summary(series: pd.Series) -> pd.Series:
            s = series.dropna()
            if s.empty:
                return pd.Series(dtype=float)
            t_mean = s.mean()
            t_std = s.std()
            abs_mean = s.abs().mean()
            gt2_rate = (s.abs() > 2).sum() / len(s)
            t_mean_over_std = t_mean / t_std if t_std != 0 else np.nan
            return pd.Series({
                "|t|均值": abs_mean,
                "|t|>2占比": gt2_rate,
                "t均值": t_mean,
                "t均值/标准差": t_mean_over_std,
            })

        coef_summary = pd.DataFrame({col: _coef_summary(coef_all[col]) for col in coef_all.columns}).T
        t_summary = pd.DataFrame({col: _t_summary(t_all[col]) for col in t_all.columns}).T

        coef_summary.to_excel(cache_mgr.get_data_path("factor_回归收益率_summary.xlsx"))
        t_summary.to_excel(cache_mgr.get_data_path("factor_回归t值_summary.xlsx"))
        print(f"  ✓ 保存汇总报告")

        # 保存元数据
        cache_mgr.save_metadata("step3")
        print(f"  ✓ 保存: step3_metadata.json")

        print(f"\n✅ Step3完成!")
    else:
        print("❌ 错误: 未生成回归结果，因样本不足或全部回归失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
