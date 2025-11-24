import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List

import qlib
from qlib.constant import REG_CN
from qlib.data import D


market = "csi300"
start_time = "2020-01-01"
end_time = "2024-12-31"
fields = ["Quantile($close, 5, 0.8)/$close", "Ref($close, -1)/$close - 1", "Log($total_mv)", "$industry_id"]


def ext_out_mad(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    # Median absolute deviation outlier removal
    :param group: Daily factor data DataFrame
    :param factor_list: List of factor names to process
    """
    for f in factor_list:
        factor = group[f]
        median = factor.median()
        mad = (factor - median).abs().median()
        edge_up = median + 3 * mad
        edge_low = median - 3 * mad
        factor = factor.astype("float64")
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)
        group[f] = factor.astype("float32")
    return group


def ext_out_3std(group: pd.DataFrame, factor_list: list, noise_std: float = 1e-10,
                      ) -> pd.DataFrame:
    """
    # 3-sigma 异常值移除并添加噪音确保唯一的分箱边界
    :param group: 日度因子数据 DataFrame
    :param factor_name: 需要处理的因子名称
    :param noise_std: 添加噪音的标准差，默认为 1e-10
    :param group_cnt: 分箱的数量，默认为 5
    """
    for f in factor_list:
        # 获取指定因子的列
        factor = group[f]

        # 添加噪音到因子列，噪音是均值为 0，标准差为 noise_std 的正态分布
        noise = np.random.normal(0, noise_std, size=len(factor))
        factor += noise  # 将噪音加到因子列

        # 确保因子列是浮动类型，避免 dtype 不兼容
        factor = factor.astype(float)

        # 计算 3-sigma 范围的上下边界
        edge_up = factor.mean() + 3 * factor.std()
        edge_low = factor.mean() - 3 * factor.std()

        # 使用 clip 限制因子的上下边界，去除异常值
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)

        # 将因子列更新回去
        group[f] = factor

    return group


def z_score(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    # Z-score standardization
    :param group: Daily factor data DataFrame
    :param factor_list: List of factor names to process
    """
    for f in factor_list:
        factor = group[f]
        if factor.std() != 0:
            group[f] = (factor - factor.mean()) / factor.std()
        else:
            group[f] = np.nan
    return group


def universal_neutralization(
    group: pd.DataFrame, 
    alpha_factors: List[str],        # 需要被中性化的 Alpha 因子列名列表 (Y)
    continuous_styles: List[str],    # 连续风格因子列名列表 (X)
    categorical_styles: List[str]    # 分类风格因子列名列表 (X)
) -> pd.DataFrame:
    """
    【单日截面】通用中性化函数：通过一次多元线性回归，同时剥离所有指定的连续和分类风格的影响。
    
    该函数假定输入 'group' 已经是按日期分组后的单个 DataFrame（即单日截面数据）。
    
    :param group: 单日截面数据 DataFrame
    :param alpha_factors: Alpha 因子列表 (因变量 Y)
    :param continuous_styles: 连续风格因子列表 (如: 'log_mv')
    :param categorical_styles: 分类风格因子列表 (如: 'industry')
    :return: 包含中性化后残差的 DataFrame
    """
    
    # ---------------------------------------------
    # 步骤 1: 构建风格回归量 X
    # ---------------------------------------------
    
    # 1.1 处理连续变量
    X_continuous = group[continuous_styles].astype("float64")
    
    # 1.2 处理分类变量 (生成哑变量)
    X_dummies = pd.DataFrame(index=group.index)
    for col in categorical_styles:
        # 核心：使用 drop_first=True，避免与截距项产生多重共线性
        dummies = pd.get_dummies(group[col], drop_first=True, prefix=col, dtype=float)
        X_dummies = pd.concat([X_dummies, dummies], axis=1)
        
    # 1.3 联合所有回归量 (X)
    X_styles = pd.concat([X_continuous, X_dummies], axis=1)
    
    # ---------------------------------------------
    # 步骤 2: 准备回归数据，清理 NaN
    # ---------------------------------------------
    
    reg_columns = alpha_factors + list(X_styles.columns)
    # 合并 Y 和 X，清理所有 NaN 值
    reg_data = pd.concat([group[alpha_factors].astype("float64"), X_styles], axis=1).dropna()
    
    # 检查样本量，确保可以执行回归
    if len(reg_data) < len(X_styles.columns) + 2: 
         return group # 样本量不足，返回原数据

    X_reg = reg_data[X_styles.columns]
    # 添加常数项
    X_reg = sm.add_constant(X_reg, has_constant='add')
    
    # ---------------------------------------------
    # 步骤 3: OLS 回归并提取残差
    # ---------------------------------------------
    
    for factor in alpha_factors:
        Y_reg = reg_data[factor]
        
        try:
            model = sm.OLS(Y_reg, X_reg).fit()
            residual = model.resid.astype(group[factor].dtype, copy=False)

            # 将残差映射回原 DataFrame 的因子列
            group.loc[residual.index, factor] = residual
        except Exception:
            # 忽略 OLS 回归失败的情况（如数据共线性、奇异矩阵等），保留原值
            print("忽略 OLS 回归失败的情况（如数据共线性、奇异矩阵等），保留原值")
            continue
            
    return group


if __name__ == "__main__":
    provider_uri = "/Users/hm/Desktop/workspace/output/qlib_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    instruments = D.instruments(market=market)
    df = D.features(instruments=instruments, fields=fields, start_time=start_time, end_time=end_time, freq='day')
    df.columns = ["qtul5", "ret_1d", "log_mv", "industry"]

    # 去极值
    if False:
        df = df.groupby(level="datetime", group_keys=False).apply(lambda x: ext_out_mad(x, ["qtul5"]))
    else:
        df = df.groupby(level="datetime", group_keys=False).apply(lambda x: ext_out_3std(x, ["qtul5"]))

    # 标准化
    df = df.groupby(level="datetime", group_keys=False).apply(lambda x: z_score(x, ["qtul5"]))

    # 中性化
    df = df.groupby(level="datetime", group_keys=False).apply(lambda x: universal_neutralization(x, ["qtul5", "ret_1d"], ["log_mv"], ["industry"]))

    print(df)
