# GPlearn 因子挖掘框架 - 问题与优化评估报告

> **评估日期**: 2026-01-27
> **版本**: V2.1
> **评估视角**: 量化金融算法工程
> **项目定位**: 研究原型（非生产系统）

> **相关文档**:
> - 系统设计思想：[core/gplearn/DESIGN.md](../core/gplearn/DESIGN.md)
> - CLI 使用说明：[step5/README.md](../step5/README.md)

---

## 目录

1. [项目原则](#项目原则)
2. [当前问题清单](#当前问题清单)
3. [修改边界与实施方案](#修改边界与实施方案)
4. [优先级矩阵](#优先级矩阵)
5. [实施路线图](#实施路线图)
6. [风险提示](#风险提示)

---

## 项目原则

### 🚫 核心约束

在评估和改进框架时，必须严格遵守以下原则：

#### 原则1：不引入日志系统

**说明**:
- ❌ 不使用 `logging` 模块
- ❌ 不引入 `loguru` 等第三方日志库
- ✅ 仅使用 `print()` 输出关键信息

**原因**:
- 保持项目简洁，避免过度工程化
- 研究原型阶段，标准输出足够使用
- 减少依赖，降低维护成本

---

#### 原则2：不引入单元测试等测试框架

**说明**:
- ❌ 不使用 `pytest`、`unittest` 等测试框架
- ❌ 不添加 `tests/` 目录和测试文件
- ✅ 通过手动验证和实际运行确保正确性

**原因**:
- 研究原型，快速迭代优先
- 测试框架增加开发成本
- 实际数据验证比单元测试更有意义

---

#### 原则3：维持 step5 配置界面不变

**说明**:
- ❌ 不修改 `step5/cli.py` 的 CLI 参数
- ❌ 不修改 `step5/遗传算法因子挖掘.py` 的数据加载逻辑
- ✅ 所有配置通过 `core/gplearn/config.py` 管理
- ✅ 所有复杂逻辑在 `core/gplearn/` 内部实现

**架构边界**:
```
step5/ (稳定层，不修改)
    ├── cli.py              # 保持不变
    └── 遗传算法因子挖掘.py # 保持不变：加载完整数据

core/gplearn/ (改进层，可修改)
    ├── config.py           # 新增配置项
    ├── miner.py            # 内部切分、验证
    ├── fitness.py          # 新增适应度函数
    ├── operators.py        # 新增算子
    ├── data_cleaner.py     # 新建：数据清洗
    └── common/             # 按需修改
```

**原因**:
- step5 是稳定的 CLI 入口，不应频繁修改
- core/gplearn 是核心算法层，应该封装所有复杂逻辑
- 用户通过修改 `config.py` 即可控制行为

---

### 📋 基于原则的调整

**移除的问题**（基于原则不再适用）:
- ❌ CLI 灵活性低 → 通过 config.py 控制
- ❌ 缺乏日志系统 → 使用 print()
- ❌ 缺乏单元测试 → 手动验证
- ❌ 异常处理不足 → 使用 print 输出错误

**保留的核心问题**（仍需解决）:
- ✅ 缺乏样本外验证 (P0)
- ✅ 数据清洗粗糙 (P0)
- ✅ 适应度函数单一 (P0)
- ✅ GP 参数过小 (P1)
- ✅ 缺乏后验分析 (P1)
- ✅ 算子库不足 (P1)

---

## 当前问题清单

### P0 级别（金融正确性 - 必须解决）

#### 问题1：缺乏样本外验证 ⭐⭐⭐⭐⭐

**位置**: [core/gplearn/miner.py](../core/gplearn/miner.py)

**现状**:
```python
def run(self, features_df, target_df) -> List[str]:
    X, y, index, boundaries = self._prepare_data(features_df, target_df)
    self._train(X, y, index, boundaries)  # ← 全样本训练
    return self._export()
```

**问题**:
- 全样本训练导致 IC 被严重高估
- 回测 IC=0.05，实盘 IC 可能=-0.02
- 无法评估因子真实表现

**金融后果**:
- 挖掘出的因子可能只是历史数据拟合
- 实盘交易时可能产生严重亏损

**解决方案**: 在 miner.py 内部切分数据
```python
def run(self, features_df, target_df):
    # 1. 时间序列切分（在 core/gplearn 内部）
    train_features, train_target, val_features, val_target = \
        self._split_data(features_df, target_df)

    # 2. 训练
    X_train, y_train, index_train, boundaries_train = \
        self._prepare_data(train_features, train_target)
    self._train(X_train, y_train, index_train, boundaries_train)

    # 3. 验证（新增）
    self._validate(val_features, val_target)

    return self._export()
```

**工作量**: 1 天

---

#### 问题2：数据清洗粗糙 ⭐⭐⭐⭐⭐

**位置**: [core/gplearn/miner.py:95-119](../core/gplearn/miner.py)

**现状**:
```python
# 数据清洗在 _prepare_data() 方法中
features_clean = features_filtered[valid_mask].fillna(0)
```

**问题**:
1. **停牌处理错误**: 长期停牌股票的 ffill 产生虚假价格
2. **未处理极端值**: 没有 winsorize 或 MAD 处理
3. **未处理成交量缺失**: 应该用 NaN 标记，而非删除

**金融后果**:
- 因子可能被虚假数据污染
- 技术指标计算错误

**解决方案**: 新建 data_cleaner.py
```python
def clean_features(features_df, target_df):
    """标准数据清洗流程"""
    # 1. 删除停牌超过 N 天的股票
    # 2. 短期缺失 ffill（限制天数）
    # 3. 极值处理（3σ + MAD）
    # 4. 删除剩余 NaN
    return features_clean, target_clean
```

**工作量**: 2-3 天

---

#### 问题3：适应度函数单一 ⭐⭐⭐⭐

**位置**: [core/gplearn/fitness.py](../core/gplearn/fitness.py)

**现状**:
```python
@register_fitness(name="rank_ic")
@with_panel_convert(min_samples=100)
def rank_ic_fitness(y_true_panel, y_pred_panel):
    ic_series = y_pred_panel.corrwith(y_true_panel, axis=1, method="spearman")
    n_samples_per_date = y_pred_panel.notna().sum(axis=1)
    ic_mean = (ic_series * n_samples_per_date).sum() / n_samples_per_date.sum()
    return ic_mean  # ← 仅均值
```

**问题**:
1. **未考虑 IC 稳定性**: 只看均值，忽略标准差
2. **未考虑 IC 衰减**: 1 日 IC 高，5 日 IC 可能为负
3. **未考虑换手率**: 高换手率会显著侵蚀收益

**示例**:
```
因子 A: IC = [0.05, 0.04, 0.06, 0.05, 0.04]
      mean=0.048, std=0.008, IR=6.0 ← 优秀

因子 B: IC = [0.10, -0.05, 0.15, -0.08, 0.12]
      mean=0.048, std=0.11, IR=0.44 ← 劣质

当前框架认为 A 和 B 等价！
```

**解决方案**: 新增适应度函数
```python
@register_fitness(name="rank_ir")
def rank_ir_fitness(y_true_panel, y_pred_panel):
    """IR 适应度函数（IC 均值 / IC 标准差）"""
    ic_series = y_pred_panel.corrwith(y_true_panel, axis=1)
    return ic_series.mean() / (ic_series.std() + 1e-10)
```

**工作量**: 1 天

---

### P1 级别（算法能力 - 应该解决）

#### 问题4：GP 参数过小 ⭐⭐⭐

**位置**: [core/gplearn/config.py:85-102](../core/gplearn/config.py)

**现状**:
```python
population_size: int = 20       # 种群太小
generations: int = 2            # 迭代太少
n_components: int = 3           # 输出因子太少
```

**问题**:
- 探索能力严重不足（20×2×4=160 个表达式 vs 建议 500×20×6=60,000）
- 早熟收敛，停留在局部最优
- 因子数量太少，无法分散风险

**解决方案**: 修改 config.py
```python
population_size: int = 500       # 20 → 500
generations: int = 20            # 2 → 20
n_components: int = 10           # 3 → 10
```

**工作量**: 0.5 天

---

#### 问题5：缺乏后验分析 ⭐⭐⭐⭐

**位置**: [core/gplearn/miner.py:199-209](../core/gplearn/miner.py)

**现状**:
```python
def _export(self) -> List[str]:
    """导出因子表达式"""
    expressions = []
    for program_list in self._transformer._programs:
        for program in program_list:
            expressions.append(str(program))
    return expressions  # ← 仅输出表达式
```

**问题**: 无任何分析（IC/IR、分位数、换手率、回撤）

**解决方案**: 新增 analyze_factors() 方法
```python
def analyze_factors(self, features_df, target_df) -> pd.DataFrame:
    """因子后验分析"""
    # 1. IC/IR 统计
    # 2. 分位数收益
    # 3. 换手率
    # 4. 最大回撤
    return analysis_df
```

**工作量**: 2-3 天

---

#### 问题6：算子库不足 ⭐⭐⭐

**位置**: [core/gplearn/operators.py](../core/gplearn/operators.py)

**现状**（V2.1）:
- 基础算子：3 个（abs, sqrt, log）
- 时序算子：9 个（sma, ema, std, momentum, delta, max, min, ts_rank, corr）
- 截面算子：2 个（rank, zscore）✅ 已启用

**仍缺**:
- 技术指标：RSI, MACD, ATR, 布林带
- 高级算子：线性组合、条件逻辑

**解决方案**: 在 operators.py 新增算子
```python
@register_operator(name="rsi", category="time_series", arity=2)
@with_boundary_check
def rolling_rsi(arr, window):
    """相对强弱指标"""
    ...
```

**工作量**: 3-5 天

---

### P2 级别（系统优化 - 可以解决）

#### 问题7：全局变量并发安全未验证 ⭐⭐⭐

**位置**: [core/gplearn/common/state.py](../core/gplearn/common/state.py)

**问题**: V2.0 从 TLS 改为全局变量，但未验证并发安全性

**风险**: 多线程环境下可能产生竞争条件

**解决方案**: 添加并发测试

**工作量**: 1-2 天

---

## 修改边界与实施方案

### 架构原则

**核心约束**: step5/ 保持不变，所有改进在 core/gplearn/ 内部实现

```
step5/ (不变)
    ↓ CLI: --market, --start-date, --end-date, --random-state
    ↓ 加载完整数据（无切分）
    ↓
core/gplearn/ (修改区域)
    ↓ 内部处理：数据切分、清洗、训练、验证
    ↓
输出：训练集IC + 验证集IC + 表达式
```

---

### 具体实施方案

#### 方案1：样本外验证（P0）

**修改文件**: `miner.py`, `config.py`

**实现步骤**:
1. 在 `miner.py` 添加 `_split_data()` 方法
2. 在 `miner.py` 添加 `_validate()` 方法
3. 在 `config.py` 添加 `train_ratio` 和 `validation_enabled` 配置项

**关键代码**:
```python
# miner.py
def _split_data(self, features_df, target_df):
    """时间序列切分"""
    config = get_default_data_config()
    train_ratio = config.train_ratio

    dates = features_df.index.get_level_values(1).unique()
    split_idx = int(len(dates) * train_ratio)
    split_date = dates[split_idx]

    train_mask = features_df.index.get_level_values(1) <= split_date
    val_mask = features_df.index.get_level_values(1) > split_date

    return (
        features_df[train_mask], target_df[train_mask],
        features_df[val_mask], target_df[val_mask]
    )
```

**工作量**: 1 天

---

#### 方案2：数据清洗改进（P0）

**修改文件**: 新建 `data_cleaner.py`, `config.py`

**实现步骤**:
1. 创建 `data_cleaner.py`
2. 实现清洗函数（停牌检测、短期填充、极值处理）
3. 在 `config.py` 添加清洗配置项
4. 在 `miner.run()` 中调用清洗函数

**关键代码**:
```python
# data_cleaner.py
def clean_features(features_df, target_df):
    """标准数据清洗流程"""
    config = get_default_data_config()

    # 1. 删除长期停牌
    if config.drop_long_suspended:
        features_df, target_df = _drop_long_suspended(
            features_df, target_df,
            max_suspension_days=config.max_suspension_days
        )

    # 2. 短期填充
    if config.fill_short_missing:
        features_df = _fill_short_missing(
            features_df,
            limit=config.fill_limit
        )

    # 3. 极值处理
    if config.clip_outliers:
        features_df = _clip_outliers(
            features_df,
            method=config.clip_method,
            n_sigma=config.n_sigma
        )

    # 4. 删除剩余 NaN
    features_df = features_df.dropna()
    target_df = target_df.loc[features_df.index]

    return features_df, target_df
```

**工作量**: 2-3 天

---

#### 方案3：适应度函数改进（P0）

**修改文件**: `fitness.py`, `config.py`

**实现步骤**:
1. 在 `fitness.py` 添加 `rank_ir` 和 `decay_ic` 函数
2. 在 `config.py` 添加 `fitness_metric` 配置项
3. 在 `miner._train()` 中选择适应度函数

**关键代码**:
```python
# fitness.py
@register_fitness(name="rank_ir")
@with_panel_convert(min_samples=100)
def rank_ir_fitness(y_true_panel, y_pred_panel):
    """IR 适应度函数"""
    ic_series = y_pred_panel.corrwith(y_true_panel, axis=1)
    return ic_series.mean() / (ic_series.std() + 1e-10)

# config.py
@dataclass(frozen=True)
class GPConfig:
    fitness_metric: str = 'rank_ic'  # 'rank_ic', 'rank_ir', 'decay_ic'
```

**工作量**: 1 天

---

#### 方案4：GP 参数优化（P1）

**修改文件**: `config.py`

**实现步骤**: 修改默认值

**关键代码**:
```python
@dataclass(frozen=True)
class GPConfig:
    population_size: int = 500       # 20 → 500
    generations: int = 20            # 2 → 20
    hall_of_fame: int = 100          # 5 → 100
    n_components: int = 10           # 3 → 10
    tournament_size: int = 20        # 3 → 20
```

**工作量**: 0.5 天

---

#### 方案5：后验分析（P1）

**修改文件**: `miner.py`, `config.py`

**实现步骤**:
1. 在 `miner.py` 添加 `analyze_factors()` 方法
2. 添加辅助函数（分位数收益、换手率、回撤）
3. 在 `config.py` 添加 `auto_analyze` 配置项

**关键代码**:
```python
# miner.py
def analyze_factors(self, features_df, target_df) -> pd.DataFrame:
    """因子后验分析"""
    results = []

    for program in self._transformer._programs[0]:
        # 计算因子值
        X, y, index, boundaries = prepare_for_gp(features_df, target_df)
        factor_values = program.transform(X)

        # IC/IR 统计
        ic_series = ...
        ir = ...

        # 分位数收益
        quantile_returns = ...

        # 换手率
        turnover = ...

        results.append({
            'expression': str(program),
            'ic_mean': ic_series.mean(),
            'ir': ir,
            ...
        })

    return pd.DataFrame(results)
```

**工作量**: 2-3 天

---

#### 方案6：算子库扩展（P1）

**修改文件**: `operators.py`

**实现步骤**:
1. 添加 RSI 算子
2. 添加 MACD 算子
3. 添加 ATR 算子
4. 添加布林带算子

**关键代码**:
```python
@register_operator(name="rsi", category="time_series", arity=2)
@with_boundary_check
def rolling_rsi(arr, window):
    """相对强弱指标"""
    delta = pd.Series(arr).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi.values
```

**工作量**: 3-5 天

---

## 优先级矩阵

| 问题 | 优先级 | 工作量 | 修改文件 | 依赖 |
|------|--------|--------|---------|------|
| **样本外验证** | P0 | 1 天 | miner.py, config.py | - |
| **数据清洗** | P0 | 2-3 天 | data_cleaner.py（新建）, config.py | - |
| **适应度函数** | P0 | 1 天 | fitness.py, config.py | - |
| **GP 参数优化** | P1 | 0.5 天 | config.py | - |
| **后验分析** | P1 | 2-3 天 | miner.py, config.py | 样本外验证 |
| **算子库扩展** | P1 | 3-5 天 | operators.py | - |
| **并发安全** | P2 | 1-2 天 | state.py | - |

**总计**: 10-15 天（约 2-3 周）

---

## 实施路线图

### 阶段 0：金融正确性（1-2 周）- 必须首先完成

**目标**: 消除致命缺陷，确保因子真实有效

- ✅ 样本外验证（1 天）
- ✅ 数据清洗改进（2-3 天）
- ✅ 适应度函数扩展（1 天）

**交付标准**:
- 训练集 IC > 0.03，验证集 IC > 0.02
- 验证集 IC > 训练集 IC × 0.5（无严重过拟合）
- 可以复现实验结果

---

### 阶段 1：算法能力提升（1-2 周）- 建议完成

**目标**: 提高因子表达能力

- ✅ GP 参数优化（0.5 天）
- ✅ 后验分析（2-3 天）
- ✅ 算子库扩展（3-5 天）

**交付标准**:
- 挖掘速度提升 10x
- 生成完整的因子分析报告
- 支持 20+ 个算子

---

### 阶段 2：系统优化（1-2 周）- 可选

**目标**: 提高系统稳定性

- ✅ 并发安全验证（1-2 天）
- ✅ 性能优化（可选）

**交付标准**:
- 通过并发测试
- 多线程性能达到预期

---

## 向后兼容性保证

**原则**: 所有修改都通过配置文件控制，默认行为保持不变

1. **新增配置项都有默认值**
   ```python
   train_ratio: float = 0.7
   validation_enabled: bool = False   # 默认关闭
   auto_analyze: bool = False
   ```

2. **新方法不影响现有流程**
   ```python
   # 现有代码
   miner.run(features_df, target_df)  # 仍然有效

   # 新增功能（可选）
   analysis = miner.analyze_factors(features_df, target_df)
   ```

3. **配置文件可以覆盖**
   ```python
   # 用户可以在自己的代码中覆盖
   from core.gplearn.config import GPConfig

   custom_config = GPConfig(
       population_size=1000,
       generations=50,
       validation_enabled=True,
   )

   miner = FactorMiner(..., gp_config=custom_config)
   ```

---

## 风险提示

⚠️ **当前框架不能直接用于实盘交易！**

**原因**:
1. ❌ 缺乏样本外验证 → IC 被严重高估
2. ❌ 数据清洗粗糙 → 因子可能被虚假数据污染
3. ❌ 适应度函数单一 → 可能挖掘出劣质因子
4. ❌ 目标标签简化 → 未考虑交易成本和风险

**建议**:
- 在解决所有 P0 问题之前，**仅用于研究**
- 至少进行 6 个月的样本外验证
- 通过 paper trading 测试后再考虑实盘

---

## 使用建议

### 对于个人研究者

✅ **可以使用当前框架**，但必须：
1. 手动划分训练集/测试集（70%/30%）
2. 在测试集上验证 IC
3. 不要相信全样本训练的结果
4. 仅用于研究，不要用于实盘

**最小可行改进**（1 天）:
```python
# 手动划分训练集/测试集
train_start, train_end = "2020-01-01", "2022-12-31"
test_start, test_end = "2023-01-01", "2023-12-31"

# 训练集挖掘
train_features = D.features(..., start_time=train_start, end_time=train_end)
miner.run(train_features, train_target)

# 测试集验证
test_features = D.features(..., start_time=test_start, end_time=test_end)
# 注意：需要手动实现测试集验证逻辑
```

---

### 对于量化开发者

🔴 **必须先完成阶段 0**（1-2 周），然后：
1. 进行 6 个月样本外验证
2. 通过 paper trading 测试
3. 逐步完善配套系统

**预期投入**: 2-3 周全职开发

---

### 对于资产管理公司

🔴 **不建议直接使用**，建议：
1. 重新审视整个因子挖掘流程
2. 聘请专业量化工程师
3. 搭建完整的量化投资平台

---

## 总结

**当前状态（V2.1）**: 研究原型，技术实现优秀，但金融专业性不足

**核心价值**:
- ✅ 快速验证因子想法
- ✅ 算法研究和技术探索
- ✅ 教学和学习遗传编程

**核心缺陷**:
- ❌ 不能直接用于实盘交易
- ❌ 缺乏样本外验证
- ❌ 数据清洗粗糙
- ❌ 适应度函数单一

**改进方向**:
- 🔴 P0：金融正确性（样本外验证、数据清洗、适应度函数）
- 🟡 P1：算法能力（GP 参数、算子库、后验分析）
- 🟠 P2：系统稳定性（并发安全）

**预期投入**: 2-3 周全职开发

---

**相关文档**:
- 系统设计思想：[core/gplearn/DESIGN.md](../core/gplearn/DESIGN.md)
- CLI 使用说明：[step5/README.md](../step5/README.md)
