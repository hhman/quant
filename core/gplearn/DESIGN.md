# GPLearn 遗传算法因子挖掘系统 - 架构设计与核心机制

> 本文档记录 GPLearn 模块的核心设计理念与技术实现细节，重点阐述数据流处理、算子装饰器机制、边界检测等关键特性的工作原理。

---

## 目录

1. [系统设计原则](#系统设计原则)
2. [核心架构决策](#核心架构决策)
3. [数据展平处理机制](#数据展平处理机制)
4. [时序算子与边界检测](#时序算子与边界检测)
5. [截面算子与面板转换](#截面算子与面板转换)
6. [适应度函数与双面板机制](#适应度函数与双面板机制)
7. [TLS 状态管理](#tls-状态管理)
8. [装饰器注册机制](#装饰器注册机制)
9. [配置管理策略](#配置管理策略)

---

## 系统设计原则

### 1. 函数式优先 (Function-First Design)

**原则**：系统采用函数式编程范式，避免复杂的面向对象层次结构。

**实践**：
```python
# 好的设计：纯函数
def rolling_sma(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window).mean().values

# 避免：不必要的类封装
class RollingOperator:
    def compute(self, arr, window):
        # ...
```

**收益**：
- 代码简洁直观
- 易于测试（输入 → 输出）
- 可组合性强

---

### 2. 明确的使用边界 (Explicit Scope Boundaries)

**系统定位**：
- **用户**：个人研究使用（单用户）
- **数据源**：仅支持 Qlib
- **执行模式**：支持多线程训练（`n_jobs=4` 默认）
- **环境**：本地脚本运行

**为何支持多线程？**

使用全局变量管理状态，所有线程共享读访问，无需锁机制：
- 边界索引正常访问 → 时序算子正常工作
- 面板数据转换正常 → 截面算子可用
- 只读场景天然安全，无竞态条件

---

### 3. 实用主义工程 (Pragmatic Engineering)

**拒绝过度设计**：
- 日志系统 → 使用 `print` + shell 重定向
- 单元测试 → 手动验证 + Jupyter 交互
- 数据库 → 文件系统（`.cache/` 目录）
- Web API → CLI + Python 函数调用

**核心原则**：在满足功能需求的前提下，保持系统最简化。

---

## 核心架构决策

### 决策 1：使用全局变量传递状态

**问题背景**：

Gplearn 的算子函数签名固定为 `func(arr, window)`，无法传递额外参数（如 MultiIndex、边界索引）。

**解决方案**：

使用全局变量在全局作用域传递状态，配合上下文管理器自动管理生命周期：

```python
# 训练时自动管理状态
with global_state(index, boundaries):
    transformer.fit(X, y)
    # 算子内部隐式访问全局变量
```

**为何是最佳方案**：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **全局变量（当前）** | • 算子签名简洁<br>• 代码侵入小<br>• 支持多线程<br>• 无性能开销 | • 需手动管理生命周期 | 多线程训练 |
| TLS | 线程隔离 | • 不支持多线程<br>• 测试困难 | |
| 改写 Gplearn | 彻底解决 | • 升级困难<br>• 维护成本高 | |

---

## 数据展平处理机制

### 问题描述

**Gplearn 的输入要求**：2D 数组 `(n_samples, n_features)`

**Qlib 的输出格式**：面板数据 `MultiIndex DataFrame`，形状为 `(n_instruments, n_dates)`

```
         datetime
         2020-01-01  2020-01-02  2020-01-03  ...
instrument
000001       10.5        10.6        10.7
000002       20.3        20.1        20.4
000003       30.1        30.2        30.0
...
```

### 展平算法

**数据转换流程**：

1. 面板数据按日期优先展平（Fortran order）
2. 重塑为 2D 数组供 Gplearn 使用
3. 保留原始 MultiIndex 用于可逆转换
4. 计算边界索引记录每只股票的起始位置

**代码示例**：
```python
# 面板数据 → 1D 数组（按列展平）
X = features_df.values.flatten(order="F")

# 展平结果
X = [
    10.5,  # 000001, 2020-01-01
    20.3,  # 000002, 2020-01-01
    30.1,  # 000003, 2020-01-01
    ...,
    10.6,  # 000001, 2020-01-02
    20.1,  # 000002, 2020-01-02
    30.2,  # 000003, 2020-01-02
    ...
]

# 重塑为 2D 数组
X = X.reshape(n_samples, n_features)

# 保留原始索引（可逆转换）
index = features_df.index  # MultiIndex (instrument, datetime)

# 计算边界索引（每只股票的起始位置）
boundaries = [0, 750, 1500, ...]  # 假设每只股票 750 个交易日
```

### 边界索引的作用

**问题**：展平后，不同股票的数据被拼接在一起，如何区分？

**解决**：边界索引记录每只股票的起始位置

```python
# 假设有 3 只股票，每只 3 天
index = [
    ('000001', '2020-01-01'),  # position 0
    ('000001', '2020-01-02'),  # position 1
    ('000001', '2020-01-03'),  # position 2
    ('000002', '2020-01-01'),  # position 3  ← 边界
    ('000002', '2020-01-02'),  # position 4
    ('000002', '2020-01-03'),  # position 5
    ('000003', '2020-01-01'),  # position 6  ← 边界
    ('000003', '2020-01-02'),  # position 7
    ('000003', '2020-01-03'),  # position 8
]

boundaries = [0, 3, 6]  # 每只股票的起始位置
```

**使用场景**：
- 时序算子边界检测：防止计算跨股票
- 面板数据转换：unstack 时恢复维度

---

## 时序算子与边界检测

### 问题：跨股票污染

**场景**：计算 5 日移动平均

```python
# 展平数据
arr = [100.0, 101.0, 102.0,  # 000001 的 3 天
       200.0, 201.0, 202.0,  # 000002 的 3 天
       ...]

# 如果没有边界检测，window=5 会这样计算：
# position 2 (000001, day3) → 使用 position 0-4
# → 错误！包含了 000002 的数据
```

**后果**：
- 价格数据污染（不同股票的价格被混在一起）
- 技术指标失效（MA、STD 等）

### 边界检测装饰器

**实现**：`core/gplearn/common/decorators.py:17-94`

装饰器自动检测函数签名，根据 arity 和 window_size 选择边界检测策略：
- `arity=1`：对时序算子标记边界后的 `window_size` 个位置为 NaN
- `arity=2`：
  - 当 `window_size=None` 时，不进行边界检测（用于 add/sub/mul/div 等算术运算）
  - 当 `window_size > 1` 时，标记边界（用于 corr 等双数组时序算子）

关键逻辑：
```python
# 跳过第一个边界（起始位置），从第二个边界开始标记
for b in boundary_indices[1:]:
    end_idx = min(b + window_size, arr_length)
    result[b:end_idx] = np.nan
```

### 工作流程示例

**场景**：2 只股票各 5 天数据展平后
- `arr = [10.1, 10.2, ..., 10.5, 20.1, 20.2, ..., 20.5]`
- `boundaries = [0, 5]`（第二只股票从 position 5 开始）

**执行流程**：
1. 计算 20 日移动平均
2. 检测到边界 position 5
3. 将 position 5-25 标记为 NaN（防止跨股票污染）
4. 前 20 个和边界后 20 个都是 NaN

### 支持的时序算子

所有带 `@with_boundary_check` 装饰器的算子：

**时序算子（arity=1）**

| 算子 | 功能 | 边界检测逻辑 |
|------|------|-------------|
| `sma` | 简单移动平均 | 删除边界后 `window` 个数据点 |
| `ema` | 指数移动平均 | 删除边界后 `window` 个数据点 |
| `std` | 滚动标准差 | 删除边界后 `window` 个数据点 |
| `delta` | 一阶差分 | 删除边界后 `window` 个数据点 |
| `max` / `min` | 滚动最大/最小值 | 删除边界后 `window` 个数据点 |
| `ts_rank` | 时间序列排名 | 删除边界后 `window` 个数据点 |

**双数组时序算子（arity=2）**

| 算子 | 功能 | 边界检测逻辑 |
|------|------|-------------|
| `corr` | 滚动相关系数 | 删除边界后 `window` 个数据点 |
| `add` / `sub` / `mul` / `div` | 算术运算 | 不进行边界检测（`window_size=None`）|

**关键点**：
- 装饰器自动处理边界，算子函数本身无需关心
- 通过全局变量隐式获取边界，算子签名保持简洁
- 算术运算算子不进行边界检测，因为它们是逐元素操作，不会跨股票污染

---

## 截面算子与面板转换

### 问题：横截面操作需要面板数据

**场景**：计算某一天所有股票的排名

```python
# 展平数据（按日期优先）
arr = [10.5, 20.3, 30.1,  # 2020-01-01 的 3 只股票
       10.6, 20.1, 30.2,  # 2020-01-02 的 3 只股票
       ...]

# 目标：计算每天的横截面排名
# 2020-01-01: rank(10.5, 20.3, 30.1) → [0.0, 0.5, 1.0]
# 2020-01-02: rank(10.6, 20.1, 30.2) → [0.0, 0.5, 1.0]
```

**问题**：Gplearn 传入的是 1D 数组，如何恢复面板结构？

### 解决方案：面板转换装饰器

**实现位置**：`core/gplearn/common/decorators.py:104-132`

装饰器自动完成以下转换流程：
1. 从全局状态获取 MultiIndex
2. 将 1D 数组转换为 DataFrame（保留 MultiIndex）
3. 使用 `unstack(level=0)` 将 DataFrame 转换为面板数据（日期为行，股票为列）
4. 调用截面算子函数处理面板数据
5. 使用 `stack()` 将面板数据展平为 1D 数组返回

### 使用示例

```python
@register_operator(name="rank", category="cross_sectional")
@with_panel_builder  # ← 自动转换面板数据
def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """横截面排名"""
    return panel.rank(axis=1, pct=True).fillna(0.5)
```

**执行流程**：
- Gplearn 调用 `rank([10.5, 20.3, 30.1, 10.6, 20.1, 30.2])`
- `with_panel_builder` 转换为面板数据（日期为行，股票为列）
- `cross_sectional_rank` 计算每天的横截面排名
- `stack()` 将结果展平为 1D 数组返回

### unstack 机制详解

**MultiIndex 结构**：
```python
index = pd.MultiIndex.from_tuples([
    ('000001', '2020-01-01'),
    ('000001', '2020-01-02'),
    ('000002', '2020-01-01'),
    ('000002', '2020-01-02'),
], names=['instrument', 'datetime'])
```

**unstack(level=0)**：将 `instrument` 层级展开为列
```python
# 展平数据
df = pd.DataFrame({'value': [10.5, 10.6, 20.3, 20.1]}, index=index)

# unstack
panel = df['value'].unstack(level=0)
# output:
# instrument  000001 000002
# datetime
# 2020-01-01   10.5   20.3
# 2020-01-02   10.6   20.1
```

**关键点**：
- `unstack(level=0)` → 第 0 层（instrument）变为列
- 行索引变为剩余层级（datetime）
- 结果形状：`(n_dates, n_instruments)`

---

## 适应度函数与双面板机制

### 问题：适应度函数需要同时访问 y_true 和 y_pred

**Gplearn 的限制**：适应度函数签名固定为 `func(y_true, y_pred)`

**需求**：计算 Rank IC（需要横截面相关性）

```python
def rank_ic(y_true, y_pred):
    # 需要将 y_true 和 y_pred 都转换为面板
    # 然后计算每天的横截面相关性
    pass
```

### 解决方案：双面板转换装饰器

**实现位置**：`core/gplearn/common/decorators.py:58-101`

装饰器处理流程：
1. 获取 MultiIndex（从参数或全局状态）
2. 调用 `build_dual_panel()` 同时构建 y_true 和 y_pred 的面板数据
3. 清洗面板数据（删除全 NaN 列）
4. 调用适应度函数处理双面板数据

### build_dual_panel 实现

**实现位置**：`core/gplearn/common/panel.py:76-95`

函数功能：
- 将 y_true 和 y_pred 合并为单个 DataFrame（共享 MultiIndex）
- 同时对两个序列执行 `unstack(level=0)` 操作
- 返回两个面板数据（形状为 `(n_dates, n_instruments)`）

### 使用示例：Rank IC 适应度函数

**函数签名**：
```python
@register_fitness(name="rank_ic")
@with_panel_convert(min_samples=100)
def rank_ic_fitness(y_true_panel: pd.DataFrame, y_pred_panel: pd.DataFrame):
    ...
```

**功能说明**：
- 计算每天的横截面相关性（Spearman 相关系数）
- 返回平均 Rank IC 作为适应度值

**数据流**：
```python
# 输入（展平）
y_true = [0.01, -0.02, 0.03, ...]  # (n_samples,)
y_pred = [0.5, -0.3, 0.8, ...]     # (n_samples,)

# with_panel_convert 转换
y_true_panel =  # (n_dates, n_instruments)
y_pred_panel =  # (n_dates, n_instruments)

# rank_ic_fitness 计算
# → 每天：corr(y_true_panel[date], y_pred_panel[date])
# → 均值：mean(IC)
```

---

## 全局状态管理

### 全局变量存储结构

**位置**：`core/gplearn/common/state.py`

```python
# 全局变量（所有线程共享）
_index_global: Optional[pd.MultiIndex] = None
_boundaries_global: Optional[List[int]] = None
```

### API 设计

提供以下接口函数：
- `set_index(multi_index)`: 保存 MultiIndex
- `set_boundary_indices([0, 750, ...])`: 保存边界索引
- `get_index()`: 获取 MultiIndex
- `get_boundary_indices()`: 获取边界索引
- `clear_globals()`: 清理所有全局数据

### 生命周期管理

使用上下文管理器自动管理全局状态的生命周期：

```python
# 训练流程
def _train(X, y, index, boundaries):
    # 使用上下文管理器自动管理全局状态
    with global_state(index, boundaries):
        transformer.fit(X, y)
        # 算子会通过全局变量获取状态
    # 退出上下文时自动清理全局状态
```

**优势**：
- 自动清理，无需手动管理
- 支持多线程读访问
- 异常安全，保证清理

---

## 注册机制

### 注册表实现

**实现位置**：`core/gplearn/common/registry.py`

核心函数：
- `create_registry(name)`: 创建空注册表
- `register(registry, name, **meta)`: 通用注册装饰器，自动推断函数 arity
- `get(registry, name, registry_name)`: 从注册表获取函数

### 算子注册表管理

**实现位置**：`core/gplearn/common/registry.py`

核心组件：
- 模块级单例 `_OPERATOR_REGISTRY`
- `_get_operator_registry()`: 延迟初始化注册表
- `register_operator(name, category, **meta)`: 算子注册装饰器
- `get_operator(name)`: 获取算子函数

### 算子定义与注册

**实现位置**：`core/gplearn/operators.py`

算子定义模式：
- arity=1 算子：使用预定义窗口（如 `sma_20`、`ema_10`）
- arity=2 算子：支持双数组运算（如 `add`、`sub`、`corr_10`）

### 时序算子的 arity 选择

**设计决策**：使用 `arity=1` + 预定义窗口，而非 `arity=2` + 运行时参数

**理由**：
1. Gplearn 在编译阶段构建算子集，预定义窗口避免运行时复杂性
2. 常用窗口（5, 10, 20, 60, 120, 250）覆盖大部分研究场景
3. arity=1 的纯函数更简洁，易于测试和组合

### Gplearn 适配层

**位置**：`core/gplearn/common/registry.py`

适配层将自定义算子转换为 gplearn 兼容的函数对象：
- `arity=1`：直接包装，无需额外处理
- `arity=2`：直接包装，支持双数组运算和时序相关性算子

### 获取所有算子

**实现位置**：`core/gplearn/common/registry.py`

函数功能：
- 遍历所有已注册算子
- 获取算子函数和元数据（arity、category 等）
- 将算子适配为 Gplearn 兼容的函数对象
- 返回算子列表供遗传算法使用

---

## 配置管理策略

### 使用 dataclass 管理配置

**位置**：`core/gplearn/config.py`

```python
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class DataConfig:
    """数据加载配置"""
    features: List[str] = field(default_factory=lambda: [
        "$close", "$open", "$high", "$low", "$volume", "$amount", "$vwap"
    ])
    target: str = "Ref($close, -1)/$close - 1"
    fillna_price: str = "ffill"    # 价格类特征填充策略
    fillna_volume: str = "zero"    # 成交量类特征填充策略

@dataclass(frozen=True)
class GPConfig:
    """遗传算法配置"""
    population_size: int = 500
    generations: int = 10
    hall_of_fame: int = 50
    n_components: int = 10
    # ...其他参数
```

**为何使用 dataclass**：
- 类型安全（IDE 自动补全）
- 不可变性（`frozen=True` 防止意外修改）
- 可文档化（docstring）
- 无需外部配置文件（YAML/TOML）

---

## 核心技术特性总结

| 特性 | 实现机制 | 核心文件 |
|------|---------|---------|
| **数据展平** | `flatten(order="F")` + 边界索引 | `panel.py` |
| **边界检测** | `@with_boundary_check` 装饰器 | `decorators.py` |
| **面板转换** | `@with_panel_builder` + unstack/stack | `decorators.py`, `panel.py` |
| **双面板构建** | `build_dual_panel()` 同时 unstack | `panel.py` |
| **算子注册** | `@register_operator` + 注册表 | `registry.py`, `operators.py` |
| **全局状态管理** | 全局变量 + 上下文管理器 | `state.py` |

---

## 扩展指南

### 新增时序算子（arity=1）

使用 `@register_operator` 和 `@with_boundary_check` 装饰器定义算子，函数接收单个数组参数。

### 批量创建时序算子

通过循环遍历常用窗口参数（5, 10, 20, 60），自动生成多个窗口版本的算子。

### 新增截面算子

使用 `@with_panel_builder` 装饰器，函数接收面板数据（DataFrame），返回面板数据。

### 新增算术运算算子（arity=2）

使用 `@register_operator` 装饰器指定 `arity=2`，函数接收两个数组参数。

### 新增双数组时序算子（arity=2）

结合 `@with_boundary_check` 和 `arity=2`，实现如滚动相关系数等算子。

---

## 开发工作流

### 修改配置

编辑 `core/gplearn/config.py`，修改 dataclass 默认值。

### 调整数据清洗策略

编辑 `core/gplearn/data.py` 中的 `clean_features()` 函数。

### 调试算子

使用 `get_operator()` 获取算子函数进行测试。

---

## 设计哲学总结

核心设计原则：
1. 函数式优先：简洁、可测试、可组合
2. 明确边界：个人研究 + Qlib + 多线程训练 + 本地
3. 实用主义：全局状态、预定义窗口、print 调试
4. 装饰器驱动：零学习成本的扩展机制
5. 拒绝过度工程：不添加不需要的抽象层

**核心原则**：在满足功能需求的前提下，保持系统最简化。
