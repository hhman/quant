# GPLearn 架构设计文档

## 核心设计理念

系统围绕**隐式状态传递**构建完整适配层，在不修改 Gplearn 源码的前提下，实现面板数据兼容。

### 核心问题

Gplearn 设计时仅支持：
- 2D 数组输入 `(n_samples, n_features)`
- 算子签名 `func(arr)` 或 `func(arr1, arr2)`
- metric 签名 `func(y, y_pred, w)`

量化因子挖掘需要：
- 面板数据计算（按日期横截面、按股票时序）
- 防止跨股票污染的边界检测
- 基于 DataFrame 的适应度评估

### 核心解决方案

**全局状态 + 装饰器适配层**

- 算子函数通过全局变量隐式获取 MultiIndex 和边界索引
- 装饰器自动完成 1D 数组 ↔ Panel DataFrame 的双向转换
- 注册表绑定元数据（arity、category、stopping_criteria）

---

## 架构分层

```
┌─────────────────────────────────────────────────────┐
│                  FactorMiner                        │
│  (协调数据流、创建 SymbolicTransformer)              │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                   适配层 (Decorators)                │
│  ┌─────────────────┬─────────────────────────────┐  │
│  │ @with_boundary_ │   @with_panel_builder       │  │
│  │ check           │   @with_panel_convert       │  │
│  │ (时序边界检测)   │   (截面/适应度面板转换)      │  │
│  └─────────────────┴─────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              状态与注册 (Registry + State)           │
│  • 全局变量: _index_global, _boundaries_global      │
│  • 注册表: _OPERATOR_REGISTRY, _FITNESS_REGISTRY    │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              面板转换 (Panel Utils)                  │
│  • flatten/unstack/stack                            │
│  • calc_boundaries                                  │
│  • build_dual_panel                                 │
└─────────────────────────────────────────────────────┘
```

---

## 数据流架构

### 输入数据流（Panel → Flatten）

```
MultiIndex DataFrame (Qlib 输出)
         ↓
dataframe_to_flatten()
         ↓
┌─────────────────────────────────────┐
│  arr: 1D 数组 (按日期优先展平)       │
│  index: MultiIndex (保留)            │
│  boundaries: [0, 750, 1500, ...]    │
└─────────────────────────────────────┘
         ↓
with global_state(index, boundaries)
         ↓
传递给 Gplearn
```

### 算子执行流（双向适配）

**时序算子 (arity=1)**
```
Gplearn 调用: sma_20(arr)
         ↓
@with_boundary_check wrapper
         ↓ (读取全局边界)
检测边界位置 → 标记 NaN
         ↓
sma_20 原函数计算
         ↓
返回结果
```

**截面算子**
```
Gplearn 调用: rank(arr)
         ↓
@with_panel_builder wrapper
         ↓ (读取全局 index)
arr → DataFrame → unstack → Panel
         ↓
cross_sectional_rank(panel)
         ↓
panel → stack → arr
         ↓
返回结果
```

**适应度函数**
```
Gplearn 调用: metric(y, y_pred, w)
         ↓
@with_panel_convert wrapper
         ↓ (读取全局 index)
build_dual_panel(y, y_pred, index)
         ↓
rank_ic_fitness(y_true_panel, y_pred_panel)
         ↓
返回 IC 均值
```

---

## 全局状态管理

### 设计模式：多读单写

**写阶段（主线程，一次性）**
```python
with global_state(index, boundaries):
    # 设置全局变量
    # 启动训练任务
```

**读阶段（工作线程，多次并行）**
```python
def operator_func(arr):
    boundaries = get_boundary_indices()  # 只读访问
    # 使用边界进行计算
```

### 并发安全保证

- 全局变量仅在上下文入口写入一次
- 工作线程仅读取，无写入操作
- 无竞态条件，无需锁机制
- 上下文退出自动清理

### 生命周期管理

上下文管理器确保：
- 进入时设置状态
- 异常时仍会清理
- 退出时自动清理

---

## 边界检测机制

### 问题

展平数据中不同股票相邻，时序算子（如 MA20）会跨股票读取数据：

```
[arr[0..749]  # 股票 A
 arr[750..1499]  # 股票 B
 ...]
```

position 749 计算时，会错误读取 position 750-769（股票 B 的数据）。

### 解决方案

装饰器根据 `arity` 和 `window_size` 决定边界检测策略：

- `arity=1, window_size=20`：标记边界后 20 个位置为 NaN
- `arity=2, window_size=None`：不检测（add/sub/mul/div 等逐元素运算）
- `arity=2, window_size=10`：标记边界后 10 个位置（corr 等双数组时序算子）

### 边界计算

```python
boundaries = [0, 750, 1500, ...]  # 每只股票起始位置

# 跳过第一个边界（从第二个边界开始标记）
for b in boundaries[1:]:
    result[b:b+window_size] = np.nan
```

---

## 面板数据转换

### unstack/stack 机制

**展平 → 面板**
```python
# 1D 数组 + MultiIndex
arr = [10.5, 20.3, 30.1, 10.6, 20.1, 30.2]
index = [('000001','2020-01-01'), ('000002','2020-01-01'),
         ('000003','2020-01-01'), ('000001','2020-01-02'), ...]

# 转换
df = pd.DataFrame({'value': arr}, index=index)
panel = df['value'].unstack(level=0)

# 结果：日期为行，股票为列
#           000001  000002  000003
# 2020-01-01   10.5   20.3   30.1
# 2020-01-02   10.6   20.1   30.2
```

**面板 → 展平**
```python
result = panel.stack().values  # 恢复为 1D 数组
```

### 双面板构建

适应度函数需要同时处理 y_true 和 y_pred：

```python
def build_dual_panel(y, y_pred, index):
    df = pd.DataFrame({'y_true': y, 'y_pred': y_pred}, index=index)
    y_true_panel = df['y_true'].unstack(level=0)
    y_pred_panel = df['y_pred'].unstack(level=0)
    return y_true_panel, y_pred_panel
```

---

## 注册表与元数据绑定

### 注册表结构

```python
{
    "sma_20": {
        "function": <function>,
        "name": "sma_20",
        "category": "time_series",
        "arity": 1
    },
    "rank_ic": {
        "function": <function>,
        "name": "rank_ic",
        "stopping_criteria": 0.03  # IC ≥ 3% 时早停
    }
}
```

### stopping_criteria 自动传递

```python
# 1. 注册时绑定元数据
@register_fitness(name="rank_ic", stopping_criteria=0.03)
@with_panel_convert(min_samples=100)
def rank_ic_fitness(y_true_panel, y_pred_panel):
    ...

# 2. FactorMiner 自动读取并传递
def _create_transformer(self, function_set):
    params = self.gp_config.to_dict()
    metric_name = params.pop("metric", "rank_ic")

    meta = _get_fitness_meta(metric_name)
    stopping_criteria = meta.get("stopping_criteria", 0.0)
    params["stopping_criteria"] = stopping_criteria  # 传递给 gplearn

    return SymbolicTransformer(..., **params)
```

### arity 自动推断

```python
def register(registry, name, **meta):
    if "arity" not in meta:
        sig = inspect.signature(func)
        # 自动计算参数个数
        arity = len([p for p in sig.parameters.values()
                     if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        meta["arity"] = arity
```

---

## Gplearn 适配层

### 算子适配

```python
def adapt_operator_to_gplearn(func, arity, name):
    from gplearn.functions import make_function
    return make_function(function=func, name=name, arity=arity, wrap=False)
```

### 适应度函数适配

```python
def _create_transformer(self, function_set):
    from gplearn.fitness import make_fitness

    fitness_func = _get_fitness_raw(metric_name)
    custom_metric = make_fitness(function=fitness_func, greater_is_better=True)
    params["metric"] = custom_metric

    return SymbolicTransformer(..., **params)
```

---

## 配置管理

### dataclass 设计

- `DataConfig`：数据源配置（特征列表、目标表达式）
- `GPConfig`：遗传算法配置（种群、代数、概率参数等）

### 配置工厂

- `get_fast_test_config()`：快速测试（20 种群、2 代）
- `get_production_config()`：生产环境（500 种群、20 代）

---

## 扩展指南

### 新增时序算子 (arity=1)

```python
@register_operator(name="my_indicator", category="time_series")
@with_boundary_check(window_size=20)
def my_indicator_20(arr):
    return pd.Series(arr).rolling(20).apply(my_logic).values
```

### 新增截面算子

```python
@register_operator(name="my_cs_op", category="cross_sectional")
@with_panel_builder
def my_cross_sectional(panel):
    return panel.apply(my_cs_logic, axis=1)
```

### 新增适应度函数

```python
@register_fitness(name="my_metric", stopping_criteria=0.05)
@with_panel_convert(min_samples=100)
def my_metric_fitness(y_true_panel, y_pred_panel):
    # 返回单个浮点数
    return ...
```

---

## 设计原则总结

1. **隐式状态传递**：全局变量 + 上下文管理器，保持算子签名简洁
2. **装饰器驱动**：适配逻辑封装在装饰器中，业务代码无侵入
3. **可逆转换**：flatten ↔ panel 双向转换，保留原始 MultiIndex
4. **元数据绑定**：注册表存储配置，自动推断 arity，自动传递 stopping_criteria
5. **多读单写**：全局变量一次写入、多次并行读取，零锁开销
