"""
Gplearn 因子挖掘配置常量

本模块定义了遗传算法挖掘的默认参数配置，包括种群大小、
进化代数、窗口大小等关键参数。
"""

# ========== 遗传算法默认参数 ==========

# 种群配置
DEFAULT_POPULATION_SIZE = 1000  # 初始种群大小
DEFAULT_GENERATIONS = 20  # 进化代数
DEFAULT_TOURNAMENT_SIZE = 50  # 锦标赛选择大小

# 输出配置
DEFAULT_N_COMPONENTS = 20  # 最终输出的因子数量
DEFAULT_HALL_OF_FAME = 100  # 候选池大小

# 程序复杂度控制
DEFAULT_INIT_DEPTH = (2, 6)  # 初始树深度范围
DEFAULT_MAX_DEPTH = 10  # 最大树深度

# 停止条件
DEFAULT_STOPPING_TOLERANCE = 0.01  # 适应度改进小于该值时停止

# ========== 数据处理配置 ==========

# 时间序列窗口
DEFAULT_WINDOW_SIZE = 10  # 默认滚动窗口大小
MAX_WINDOW_SIZE = 60  # 最大窗口大小
MIN_WINDOW_SIZE = 5  # 最小窗口大小

# 并行配置
DEFAULT_N_JOBS = 4  # 默认并行度
DEFAULT_VERBOSE = 1  # 详细程度（0=静默，1=进度，2=详细）

# 随机种子
DEFAULT_RANDOM_STATE = 42  # 随机种子（保证可复现）

# ========== 适应度配置 ==========

# 适应度阈值
FITNESS_THRESHOLD = 0.03  # IC 低于此值视为无效因子
FITNESS_MIN_SAMPLES = 100  # 计算适应度至少需要的样本数

# 边界删除配置
BOUNDARY_DELETE_WINDOW = True  # 是否删除边界污染数据

# ========== 算子配置 ==========

# Gplearn 内置算子
GPLEARN_FUNCTIONS = [
    "add",
    "sub",
    "mul",
    "div",
    "abs",
    "neg",
    "sqrt",
    "log",
    "max",
    "min",
]

# ========== 表达式转换配置 ==========

# 特征占位符前缀
FEATURE_PREFIX = "X"  # Gplearn 特征占位符前缀（X0, X1, ...）

# Qlib 特征前缀
QLIB_FEATURE_PREFIX = "$"  # Qlib 特征前缀（$close, $volume, ...）

# ========== 输出配置 ==========

# 输出文件后缀
OUTPUT_SUFFIX_PARQUET = "parquet"
OUTPUT_SUFFIX_PICKLE = "pkl"
OUTPUT_SUFFIX_JSON = "json"
OUTPUT_SUFFIX_CSV = "csv"

# 默认输出目录
DEFAULT_OUTPUT_DIR = ".cache"
