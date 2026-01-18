"""
Gplearn 因子挖掘自定义异常类

本模块定义了遗传算法挖掘过程中可能出现的各种异常情况。
"""


class GplearnError(Exception):
    """Gplearn 因子挖掘基础异常类"""

    pass


class GplearnDataError(GplearnError):
    """
    数据异常

    当数据加载、处理或验证过程中出现问题时抛出。

    示例：
        - 数据缺失率过高
        - 数据格式不正确
        - 边界索引记录不完整
    """

    pass


class BoundaryPollutionError(GplearnError):
    """
    边界污染异常

    当时间序列算子在股票交界处产生跨股票计算时抛出。

    示例：
        - 边界删除不完整
        - 窗口大小设置不当
        - MultiIndex 还原失败
    """

    pass


class ExpressionConversionError(GplearnError):
    """
    表达式转换异常

    当 Gplearn 表达式转换为 Qlib 语法失败时抛出。

    示例：
        - 特征名称无效
        - 算子不支持
        - 语法解析失败
        - AST 构建失败
    """

    pass


class FitnessCalculationError(GplearnError):
    """
    适应度计算异常

    当适应度函数计算失败时抛出。

    示例：
        - 样本量不足
        - 边界删除后无数据
        - 相关系数计算失败（全为 NaN）
    """

    pass


class MiningConfigurationError(GplearnError):
    """
    挖掘配置异常

    当挖掘参数配置不正确时抛出。

    示例：
        - 参数范围不合理
        - 相互矛盾的参数
        - 不支持的特征列表
    """

    pass
