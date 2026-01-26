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

from utils.cache_manager import CacheManager
from core.factor_analysis import factor_return_industry_marketcap


def calculate_returns(
    market: str,
    start_date: str,
    end_date: str,
    factor_formulas: list[str],
    provider_uri: str,
) -> None:
    """
    计算因子收益率的核心逻辑函数

    Parameters:
    -----------
    market : str
        市场标识
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
    # 创建cache管理器
    cache_mgr = CacheManager(market, start_date, end_date)

    print("\n📊 Step3: 因子收益回归")

    # 加载数据
    print("📥 加载数据...")
    factor_std = cache_mgr.read_dataframe("factor_std")
    styles_df = cache_mgr.read_dataframe("styles")
    ret_df = cache_mgr.read_dataframe("returns")

    # 合并数据用于回归
    # 策略: 使用join而非concat,保留收益率数据(基于all)的完整性
    # factor_std和styles_df基于csi300, ret_df可能基于all
    data = factor_std.join(styles_df, how="left")
    data = data.join(ret_df, how="left")  # left join保留all的股票

    print(f"  ✓ 合并数据: {data.shape}")
    print(
        f"    - 因子股票数: {len(factor_std.index.get_level_values('instrument').unique())}"
    )
    print(
        f"    - 收益率股票数: {len(ret_df.index.get_level_values('instrument').unique())}"
    )
    print(
        f"    - 合并后股票数: {len(data.index.get_level_values('instrument').unique())}"
    )

    # 提取列
    factor_cols = [col for col in factor_std.columns if col in factor_formulas]
    ret_cols = [col for col in ret_df.columns if col.startswith("ret_")]

    # 如果没有匹配的因子，说明参数错误
    if not factor_cols:
        print(f"❌ 错误: 请求的因子 {factor_formulas} 在cache中不存在")
        print(
            f"  cache中的因子列: {[col for col in factor_std.columns if col not in ['$total_mv', '$industry', '$float_mv']]}"
        )
        sys.exit(1)

    # 检查必需的风格列（step1提供的是$total_mv，不是$log_mv）
    required_style_cols = ["$total_mv", "$industry", "$float_mv"]
    missing_cols = [
        col for col in ret_cols + required_style_cols if col not in data.columns
    ]
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

        cache_mgr.write_dataframe(coef_all, "return_coef")
        print(f"  ✓ 保存: return_coef ({coef_all.shape})")

        cache_mgr.write_dataframe(t_all, "return_tval")
        print(f"  ✓ 保存: return_tval ({t_all.shape})")

        # 生成汇总统计
        # 注意：汇总统计需要对整个时间序列计算，而不是单日的结果
        def _coef_summary(series: pd.Series) -> pd.Series:
            s = series.dropna()
            if s.empty:
                return pd.Series(dtype=float)
            mean = s.mean()
            std = s.std()
            t_test = mean / std * np.sqrt(len(s)) if std != 0 else np.nan
            return pd.Series(
                {
                    "因子收益率均值": mean,
                    "因子收益率序列t检验": t_test,
                }
            )

        def _t_summary(series: pd.Series) -> pd.Series:
            s = series.dropna()
            if s.empty:
                return pd.Series(dtype=float)
            t_mean = s.mean()
            t_std = s.std()
            abs_mean = s.abs().mean()
            gt2_rate = (s.abs() > 2).sum() / len(s)
            t_mean_over_std = t_mean / t_std if t_std != 0 else np.nan
            return pd.Series(
                {
                    "|t|均值": abs_mean,
                    "|t|>2占比": gt2_rate,
                    "t均值": t_mean,
                    "t均值/标准差": t_mean_over_std,
                }
            )

        coef_summary = pd.DataFrame(
            {col: _coef_summary(coef_all[col]) for col in coef_all.columns}
        ).T
        t_summary = pd.DataFrame(
            {col: _t_summary(t_all[col]) for col in t_all.columns}
        ).T

        # 使用紧凑日期格式保存汇总文件
        start_compact = start_date.replace("-", "")
        end_compact = end_date.replace("-", "")
        coef_summary.to_excel(
            f".cache/{market}_{start_compact}_{end_compact}__return_coef_summary.xlsx"
        )
        t_summary.to_excel(
            f".cache/{market}_{start_compact}_{end_compact}__return_tval_summary.xlsx"
        )
        print("  ✓ 保存汇总报告")

        print("\n✅ Step3完成!")
    else:
        print("❌ 错误: 未生成回归结果，因样本不足或全部回归失败")
        sys.exit(1)
