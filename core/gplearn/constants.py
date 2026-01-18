"""
Gplearn 因子挖掘常量定义

本模块定义了遗传算法挖掘过程中使用的常量，包括
数据维度、适应度类型、算子类型等。
"""

# ========== 数据维度常量 ==========

# 维度名称
DIM_INSTRUMENT = "instrument"  # 标的维度
DIM_DATETIME = "datetime"  # 时间维度
DIM_FEATURE = "feature"  # 特征维度

# MultiIndex 级别顺序
MULTIINDEX_LEVELS = [DIM_INSTRUMENT, DIM_DATETIME]

# ========== 适应度类型常量 ==========


class FitnessType:
    """适应度类型"""

    RANK_IC = "rank_ic"  # Rank IC（默认）
    WEIGHTED_IC = "weighted_ic"  # 加权 Pearson IC
    COMPOSITE_IC = "composite_ic"  # 复合适应度


# ========== 算子类型常量 ==========


class OperatorType:
    """算子类型"""

    # 时间序列算子
    SMA = "sma"  # 简单移动平均
    EMA = "ema"  # 指数移动平均
    STD = "std"  # 滚动标准差
    MOMENTUM = "momentum"  # 动量
    DELTA = "delta"  # 一阶差分
    CORR = "corr"  # 滚动相关系数

    # 截面算子
    RANK = "rank"  # 横截面排名
    ZSCORE = "zscore"  # 横截面标准化

    # 基础算子
    ADD = "add"  # 加法
    SUB = "sub"  # 减法
    MUL = "mul"  # 乘法
    DIV = "div"  # 除法
    ABS = "abs"  # 绝对值
    SQRT = "sqrt"  # 平方根
    LOG = "log"  # 对数


# ========== 时间窗口常量 ==========

# 预定义窗口大小
WINDOWS_SHORT = [5, 10, 20]  # 短期窗口
WINDOWS_MEDIUM = [20, 40, 60]  # 中期窗口
WINDOWS_LONG = [60, 120, 250]  # 长期窗口
ALL_WINDOWS = WINDOWS_SHORT + WINDOWS_MEDIUM + WINDOWS_LONG

# ========== 质量控制常量 ==========

# 因子质量阈值
MIN_RANK_IC = 0.03  # 最小 Rank IC 均值
MIN_ICIR = 0.5  # 最小 ICIR
MIN_SHARPE = 1.0  # 最小夏普比率
MAX_DRAWDOWN = 0.2  # 最大回撤

# 表达式复杂度限制
MAX_TREE_DEPTH = 10  # 最大树深度
MAX_TREE_LENGTH = 50  # 最大节点数

# 数据质量阈值
MAX_MISSING_RATIO = 0.05  # 最大缺失值比例
MAX_OUTLIER_RATIO = 0.1  # 最大异常值比例

# ========== 表达式转换常量 ==========

# 算子映射表（Gplearn → Qlib）
OPERATOR_MAPPING = {
    "add": "+",
    "sub": "-",
    "mul": "Mul",
    "div": "Div",
    "abs": "Abs",
    "sqrt": "Sqrt",
    "log": "Log",
    "max": "Max",
    "min": "Min",
    "rank": "Rank",
    "zscore": "ZScore",
}

# ========== 日志常量 ==========

# 日志级别
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
