#!/usr/bin/env python3
"""
Step5: 遗传算法因子挖掘
功能：使用 Gplearn 的 SymbolicTransformer 自动挖掘因子表达式
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import qlib
from qlib.constant import REG_CN
from qlib.data import D

from core.gplearn import FactorMiner
from core.gplearn.config import DEFAULT_FEATURES, DEFAULT_TARGET, GPConfig
from utils.cache_manager import CacheManager


def mine_factors_with_gp(
    market: str,
    start_date: str,
    end_date: str,
    provider_uri: str,
    random_state: int = None,
) -> None:
    """
    使用遗传算法挖掘因子

    Parameters:
    -----------
    market : str
        市场标识
    start_date : str
        起始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    provider_uri : str
        Qlib 数据路径
    random_state : int
        随机种子（不指定则自动生成）
    """
    # 处理随机种子：未指定时自动生成
    if random_state is None:
        import random

        random_state = random.randint(0, 2**32 - 1)
        print(f"  自动生成随机种子: {random_state}")

    # 初始化 Qlib
    print(f"🔧 初始化 Qlib: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 创建cache管理器
    cache_mgr = CacheManager(market, start_date, end_date)

    print("\n🧬 Step5: 遗传算法因子挖掘")
    print(f"  市场: {market}")
    print(f"  日期: {start_date} ~ {end_date}")
    print(f"  特征: {len(DEFAULT_FEATURES)} 个")
    print(f"  目标: {DEFAULT_TARGET}")
    print(f"  随机种子: {random_state}")

    # 获取股票列表
    instruments = D.instruments(market=market)

    # 加载特征数据
    print("📥 加载特征数据...")
    features_df = D.features(
        instruments=instruments,
        fields=DEFAULT_FEATURES,
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    features_df.columns = DEFAULT_FEATURES
    features_df = features_df.groupby(level="instrument", group_keys=False).apply(
        lambda x: x.ffill().bfill()
    )
    print(f"  ✓ 特征数据: {features_df.shape}")

    # 从cache加载收益率数据作为标签
    print("📥 加载标签数据...")
    ret_df = cache_mgr.read_dataframe("returns")
    ret_df = ret_df[["ret_1d"]]
    ret_df.columns = [DEFAULT_TARGET]
    print(f"  ✓ 标签数据: {ret_df.shape}")

    # GP 挖掘
    print("⚙️  训练 GP 模型...")
    miner = FactorMiner(
        features=DEFAULT_FEATURES,
        target=DEFAULT_TARGET,
        gp_config=GPConfig(),
        random_state=random_state,
    )

    expressions = miner.run(features_df, ret_df)

    # 保存表达式文件
    print("💾 保存表达式文件...")
    output_dir = Path(".cache")
    output_dir.mkdir(exist_ok=True)

    start_compact = start_date.replace("-", "")
    end_compact = end_date.replace("-", "")
    filename = (
        f"{market}_{start_compact}_{end_compact}__gp_seed{random_state}.expression.txt"
    )
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        for expr in expressions:
            f.write(f"{expr}\n")

    print(f"  ✓ 表达式已保存: {output_path}")

    # 输出结果
    print(f"\n✅ 挖掘完成！发现的 {len(expressions)} 个因子表达式:")
    for i, expr in enumerate(expressions, 1):
        print(f"\n  Factor {i}:")
        print(f"    {expr}")

    print("\n✅ Step5完成!")
