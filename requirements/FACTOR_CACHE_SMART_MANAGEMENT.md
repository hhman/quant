# 因子缓存智能管理需求文档

> **文档版本**: v1.3
> **创建日期**: 2026-01-18
> **最后更新**: 2026-01-18
> **需求类型**: 功能增强
> **优先级**: P0（高优先级）

> **核心逻辑**: 严格的表达式字符串匹配（相同表达式 → 替换，不同表达式 → 追加）

---

## 📋 目录

- [1. 需求背景](#1-需求背景)
- [2. 当前现状分析](#2-当前现状分析)
- [3. 问题识别](#3-问题识别)
- [4. 需求目标](#4-需求目标)
- [5. 使用场景分析](#5-使用场景分析)
- [6. 功能需求](#6-功能需求)
- [7. 技术方案](#7-技术方案)
- [8. 实施计划](#8-实施计划)
- [9. 附录](#9-附录)

---

## 1. 需求背景

### 1.1 业务背景

量化因子研究是一个**迭代式、增量式**的过程：

- 研究员需要不断测试新的因子表达式
- 需要保留历史研究成果以便对比分析
- 需要支持多因子并行研究和组合分析
- 计算成本高昂，应避免重复计算

### 1.2 当前痛点

**研究成果丢失问题**：
```bash
# 第1天：研究动量因子
python step1/cli.py --factor-formulas "Ref($close,60)/$close" ...
python step2/cli.py ...
python step3/cli.py ...
python step4/cli.py ...
# 📊 生成报告：动量因子表现良好

# 第2天：想研究成交量因子，并对比
python step1/cli.py --factor-formulas "Mean($volume,20)" ...
# ❌ 动量因子的数据被删除了！
# ❌ 无法回看第1天的结果
# ❌ 无法做因子对比
```

**核心问题**：当前 Cache 机制采用"替换模式"，每次运行都会覆盖之前的因子数据，导致研究成果丢失。

---

## 2. 当前现状分析

### 2.1 系统架构

```
Step0: 数据预处理
  ↓
Step1: 因子提取 → factor_std.parquet
  ↓
Step2: 因子中性化 → neutralized.parquet
  ↓
Step3: 因子收益率 → return_coef.parquet, return_tval.parquet
  ↓
Step4: 因子绩效 → ic.parquet, rank_ic.parquet, group_return.parquet, etc.
```

### 2.2 当前 Cache 机制

**文件命名规则**（保持不变）：
```
{market}_{start_date}_{end_date}__{data_type}.parquet

示例：
- csi300_20230101_20240101__factor_std.parquet
- csi300_20230101_20240101__neutralized.parquet
- csi300_20230101_20240101__return_coef.parquet
```

**当前写入策略**（`utils/cache_manager.py`）：
```python
# 收益率数据：跨市场复用
if data_type == "returns":
    if file_exists:
        return  # 跳过写入

# 其他数据：直接替换
if file_exists:
    print("🔄 替换已有文件")
else:
    print("💾 创建新文件")

df.to_parquet(path, compression=compression, index=True)
```

**核心问题**：
- ✅ 收益率处理正确（复用已有文件）
- ❌ 风格数据重复计算（每个市场都重新提取）
- ❌ 因子数据采用替换模式（研究成果丢失）

### 2.3 数据存储结构

**当前 Parquet 文件结构**：
```
Index: MultiIndex [instrument, datetime]
Columns:
  - factor_std: [Ref($close,60)/$close]  # 单列
  - neutralized: [Ref($close,60)/$close_neu]
  - return_coef: [Ref($close,60)/$close_ret_1d, ...]
```

**设计缺陷**：
- 每个文件只存储当前运行的因子
- 不支持增量添加新因子
- 无法保留研究历史

---

## 3. 问题识别

### 3.1 核心问题

| 问题编号 | 问题描述 | 影响 | 严重程度 |
|---------|---------|------|---------|
| P1 | 因子数据被覆盖 | 研究成果丢失 | 🔴 高 |
| P2 | 无法增量添加因子 | 每次只能测试单个因子 | 🟡 中 |
| P3 | 多因子对比困难 | 需要重复计算已有因子 | 🟡 中 |
| P4 | 重复计算浪费资源 | 相同因子重复计算 | 🟡 中 |
| P5 | 风格数据重复计算 | 每个市场都重新提取市值/行业数据 | 🟡 中 |

### 3.2 边界场景问题

| 场景 | 当前行为 | 问题 |
|------|---------|------|
| **新增因子** | 覆盖整个文件 | 旧因子丢失 |
| **替换因子** | 覆盖整个文件 | ✅ 符合预期 |
| **部分更新** | 不支持 | 无法只更新部分因子 |
| **跨市场复用** | ✅ 收益率正确 | 其他数据类型不支持 |
| **因子不存在** | 报错信息不友好 | ❌ 用户体验差 |

---

## 4. 需求目标

### 4.1 核心目标

**保留研究成果，支持增量研究**

1. ✅ **研究成果持久化**：因子数据不会因为新因子的添加而丢失
2. ✅ **增量添加因子**：支持在现有基础上添加新因子
3. ✅ **智能更新机制**：已存在的因子重新计算，新因子追加保存
4. ✅ **跨市场复用**：保持收益率数据的跨市场复用能力
5. ✅ **风格数据优化**：风格数据（市值、行业等）也支持跨市场复用，避免重复计算

### 4.2 设计原则

**两条铁律**（不可违背）：

1. ✅ **不修改 CLI 参数**：不新增 `--name`、`--append` 等参数
2. ✅ **不修改文件命名规则**：保持 `{market}_{dates}__{type}.parquet` 格式

**智能判断逻辑**：
```python
# 基于严格的表达式字符串匹配：
# - 表达式字符串完全相同 → 替换（重新计算）
# - 表达式字符串不同 → 追加（新因子）
# - 未请求的已存在因子 → 保留（不删除）

# 特殊数据处理：
# - returns（收益率）→ 统一保存为 all 市场，跨市场复用
# - styles（风格数据）→ 统一保存为 all 市场，跨市场复用
# - 其他数据 → 智能合并（追加/替换/保留）

# 因子身份定义：
# - 因子身份 = 完整的表达式字符串（精确匹配）
# - 示例：Ref($close,60)/$close ≠ Ref($close,30)/$close（不同表达式）
# - 示例：Ref($close,60)/$close = Ref($close,60)/$close（相同表达式）
```

### 4.3 非目标

**明确不做的事情**：
- ❌ 不提供因子版本管理（如 v1, v2）
- ❌ 不提供因子删除功能
- ❌ 不提供因子重命名功能
- ❌ 不修改文件命名规则

---

## 5. 使用场景分析

### 5.1 核心使用场景

#### 场景1：单因子探索（最常见）

```bash
# 第1天：研究60日动量因子
python step1/cli.py --factor-formulas "Ref($close,60)/$close" ...
# → 文件有1列：[Ref($close,60)/$close]

# 第2天：想研究30日动量因子（保留60日因子，做对比）
python step1/cli.py --factor-formulas "Ref($close,30)/$close" ...
# → Ref($close,30)/$close 是新表达式，追加新列

# 最终：文件有2列：[Ref($close,60)/$close, Ref($close,30)/$close]
```

**预期结果**：
- ✅ 追加新因子（`Ref($close,30)/$close` 是不同的表达式）
- ✅ 保留旧因子（`Ref($close,60)/$close` 不被删除）
- ✅ 可以对比两个不同参数的因子效果

**关键逻辑**：
- `Ref($close,60)/$close` ≠ `Ref($close,30)/$close`（字符串不同）
- → 判定为**不同因子**，应该**追加**

---

#### 场景2：增量添加因子（核心需求）

```bash
# 第1天：研究动量因子
python step1/cli.py --factor-formulas "Ref($close,60)/$close" ...
# → 文件有1列：[Ref($close,60)/$close]

# 第2天：想研究成交量因子（保留动量因子）
python step1/cli.py --factor-formulas "Mean($volume,20)" ...
# → 文件有2列：[Ref($close,60)/$close, Mean($volume,20)]
```

**预期结果**：
- ✅ 追加新因子
- ✅ 保留旧因子

---

#### 场景3：多因子一次性输入

```bash
# 一次性测试3个因子
python step1/cli.py \
  --factor-formulas "Ref($close,60)/$close;Mean($volume,20);MA($close,20)" ...

# → 文件有3列：[Ref($close,60)/$close, Mean($volume,20), MA($close,20)]
```

**预期结果**：
- ✅ 创建3个因子
- ✅ 保存到同一个文件

---

#### 场景4：混合场景（部分替换，部分追加，部分保留）

```bash
# 文件已有：[Ref($close,60)/$close, Mean($volume,20)]

# 用户运行：
python step1/cli.py --factor-formulas "Ref($close,60)/$close;Mean($volume,30)" ...

# → Ref($close,60)/$close：表达式相同 → 重新计算并替换
# → Mean($volume,20)：未请求 → 保留不变
# → Mean($volume,30)：表达式不同 → 新增追加

# 最终：[Ref($close,60)/$close(新), Mean($volume,20)(旧), Mean($volume,30)(新)]
```

**预期结果**：
- ✅ 智能判断哪些列要替换（表达式完全相同）
- ✅ 智能判断哪些列要追加（表达式不同）
- ✅ 智能判断哪些列要保留（未请求的已存在因子）

**关键逻辑**：
- `Ref($close,60)/$close`（新） = `Ref($close,60)/$close`（旧） → **替换**
- `Mean($volume,30)` ≠ `Mean($volume,20)` → **追加**
- `Mean($volume,20)` 未在请求中 → **保留**

---

#### 场景5：跨市场复用（已实现）

```bash
# csi300 市场
python step1/cli.py --market csi300 --factor-formulas "Ref($close,60)/$close" ...
# → 收益率保存为 all 市场文件
# → 风格数据保存为 all 市场文件

# sse50 市场（复用收益率和风格数据）
python step1/cli.py --market sse50 --factor-formulas "Mean($volume,20)" ...
# → 收益率文件被复用，不重复计算
# → 风格数据文件被复用，不重复计算
```

**预期结果**：
- ✅ 收益率跨市场复用
- ✅ 风格数据跨市场复用（市值、行业、流通市值）
- ✅ 风格数据覆盖 all 市场全部股票（非固定成份股）
- ✅ 其他数据类型智能合并

---

### 5.2 边界场景

#### 场景6：因子不存在

```bash
# Step1 只计算了 factor_1
python step1/cli.py --factor-formulas "factor_1" ...

# Step2 尝试计算 factor_1 和 factor_2
python step2/cli.py --factor-formulas "factor_1;factor_2" ...

# ❌ 报错：factor_2 不存在
```

**预期结果**：
- ✅ 友好的错误提示
- ✅ 明确指出缺失的因子
- ✅ 提示可用的因子

---

#### 场景7：部分列读取（性能优化）

```python
# 文件中有 [factor_1, factor_2, factor_3, factor_4]

# Step2 只需要 factor_1 和 factor_2
factor_df = cache_mgr.read_columns(['factor_1', 'factor_2'], 'factor_std')

# → 只读取指定的列，节省内存和时间
```

**预期结果**：
- ✅ 支持部分列读取
- ✅ 提升性能

---

## 6. 功能需求

### 6.1 智能写入功能

**需求编号**: FR-01

**需求描述**:
实现智能的 DataFrame 写入机制，自动判断因子列是追加、替换还是保留。

**功能规格**:

| 输入条件 | 输出行为 | 日志提示 |
|---------|---------|---------|
| 文件不存在 | 创建新文件 | `💾 创建新文件` |
| 新因子列 | 追加到现有文件 | `➕ 追加新因子 (N个)` |
| 已存在因子列 | 重新计算并替换 | `🔄 更新已有因子 (N个)` |
| 混合场景 | 智能合并 | `🔄 更新已有因子 (N个)`<br>`➕ 追加新因子 (M个)`<br>`✅ 保留已有因子 (K个)` |
| 完全相同 | 跳过写入 | `✅ 因子无变化，跳过写入` |

**技术约束**:
- 不修改 CLI 参数
- 不修改文件命名规则
- 只修改 `utils/cache_manager.py`

---

### 6.2 智能读取功能

**需求编号**: FR-02

**需求描述**:
提供友好的因子读取和检查功能，支持部分列读取和缺失检测。

**功能规格**:

#### FR-02.1: 增强 `read_dataframe` 方法

**输入**:
```python
cache_mgr.read_dataframe(
    data_type="factor_std",
    columns=["factor_1", "factor_2"]  # 可选：指定列
)
```

**行为**:
- `columns=None`: 读取全部列
- `columns=["factor_1", "factor_2"]`: 只读取指定列（性能优化）
- 请求的列不存在 → 抛出友好的错误信息

**错误信息格式**:
```
❌ 请求的因子不存在:
   缺失: ['factor_3', 'factor_4']
   可用: ['factor_1', 'factor_2', 'factor_5']
   文件: csi300_20230101_20240101__factor_std.parquet

💡 建议:
   - 先运行 step1 生成缺失的因子:
     python step1/cli.py --factor-formulas "factor_3;factor_4" ...
```

---

#### FR-02.2: 新增 `check_columns` 方法

**输入**:
```python
result = cache_mgr.check_columns(
    data_type="factor_std",
    required_columns=["factor_1", "factor_2", "factor_3"],
    verbose=True
)
```

**输出**:
```python
{
    'exists': False,  # 是否全部存在
    'missing': ['factor_3'],  # 缺失的列
    'available': ['factor_1', 'factor_2'],  # 可用的列
    'path': Path(.../factor_std.parquet)  # 文件路径
}
```

**日志输出**（`verbose=True`）:
```
⚠️  部分因子缺失:
   缺失: ['factor_3']
   可用: ['factor_1', 'factor_2']
```

---

### 6.3 风格数据跨市场复用

**需求编号**: FR-03

**需求描述**:
风格数据（styles: 市值、行业、流通市值）也应该支持跨市场复用，避免重复计算。

**当前问题**:
```python
# 当前实现（step1/因子提取与预处理.py）
instruments = D.instruments(market=market)  # 固定成份股

styles_df = pd.concat([total_mv, industry, float_mv], axis=1)
cache_mgr.write_dataframe(styles_df, "styles")
```

**问题分析**:
1. ❌ 每个市场都重新提取风格数据（csi300, sse50, csi500...）
2. ❌ 只包含特定市场的成份股，不包含全市场股票
3. ❌ 浪费计算资源和存储空间

**优化方案**:
```python
# 优化后实现
# 风格数据统一保存为 all 市场
if data_type == "styles":
    all_cache_mgr = CacheManager("all", self.start_date, self.end_date)
    path = all_cache_mgr.get_parquet_path(data_type)
    if verbose:
        print(f"    💾 风格数据保存为all市场: {path.name}")
        print("    ⚡ 其他市场可复用此文件")
```

**技术实现**:
1. 使用 `all` 市场提取风格数据（覆盖全市场股票）
2. 统一保存为 `all_{dates}__styles.parquet`
3. 检测到文件存在时跳过写入（复用已有数据）

**功能规格**:

| 输入条件 | 输出行为 | 日志提示 |
|---------|---------|---------|
| 文件不存在 | 创建新文件（all市场，全股票） | `💾 风格数据保存为all市场: all_xxx__styles.parquet`<br>`⚡ 其他市场可复用此文件` |
| 文件存在 | 跳过写入，复用已有数据 | `✅ 风格数据文件已存在，复用已有文件` |

**读取逻辑**:
```python
def read_dataframe(self, data_type: str, columns=None):
    if data_type == "styles":
        # 优先读取 all 市场的风格数据
        all_styles_path = self.CACHE_DIR / f"all_{self.start_date_compact}_{self.end_date_compact}__styles.parquet"
        if all_styles_path.exists():
            if self.market != "all":
                print("    ⚡ 复用all市场风格数据 (跨市场复用)")
            path = all_styles_path
        else:
            path = self.get_parquet_path(data_type)
    else:
        path = self.get_parquet_path(data_type)

    return pd.read_parquet(path, columns=columns)
```

**预期效果**:
```bash
# csi300 市场
python step1/cli.py --market csi300 --factor-formulas "Ref(\$close,60)/\$close" ...
# 输出：
#   💾 风格数据保存为all市场: all_20230101_20240101__styles.parquet
#   ⚡ 其他市场可复用此文件

# sse50 市场（复用风格数据）
python step1/cli.py --market sse50 --factor-formulas "Mean(\$volume,20)" ...
# 输出：
#   ✅ 风格数据文件已存在，复用已有文件
```

---

### 6.4 全链路支持

**需求编号**: FR-04

**需求描述**:
Step1/2/3/4 都自动继承智能写入机制，无需修改业务代码。

**技术实现**:
- 所有数据类型统一使用 `cache_mgr.write_dataframe()`
- 智能合并逻辑在 `cache_manager.py` 中统一实现
- Step 代码无需修改，自动支持增量更新

**影响的数据类型**:
- `factor_raw` ✅ (智能合并)
- `factor_std` ✅ (智能合并)
- `neutralized` ✅ (智能合并)
- `return_coef` ✅ (智能合并)
- `return_tval` ✅ (智能合并)
- `ic` ✅ (智能合并)
- `rank_ic` ✅ (智能合并)
- `group_return` ✅ (智能合并)
- `autocorr` ✅ (智能合并)
- `turnover` ✅ (智能合并)
- `returns` ✅ (跨市场复用，已实现)
- `styles` ✅ (跨市场复用，新增)

---

## 7. 技术方案

### 7.1 架构设计

**核心思想**: 逻辑集中，单一职责

```
┌─────────────────────────────────────────┐
│         cache_manager.py                │
│  ┌───────────────────────────────────┐  │
│  │  write_dataframe()                │  │
│  │  - 智能合并逻辑                   │  │
│  │  - 列级增删改                     │  │
│  │  - 友好日志                       │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  read_dataframe()                 │  │
│  │  - 部分列读取                     │  │
│  │  - 友好错误提示                   │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  check_columns()                  │  │
│  │  - 列存在性检查                   │  │
│  │  - 缺失分析                       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│         Step1/2/3/4                    │
│  - 业务逻辑                            │
│  - 调用 cache_mgr                      │
│  - 无需关心存储细节                    │
└─────────────────────────────────────────┘
```

### 7.2 核心算法

#### 智能合并算法

```python
def smart_merge(existing_df, new_df, verbose=True):
    """
    智能合并两个 DataFrame（基于严格的表达式字符串匹配）

    核心原则：
    - 因子身份 = 完整的表达式字符串
    - 相同表达式 → 替换（重新计算）
    - 不同表达式 → 追加（新因子）
    - 未请求的已存在因子 → 保留（不删除）

    算法：
    1. 提取列名集合（表达式字符串）
    2. 计算交集（要替换）和差集（要追加）
    3. 删除要替换的列
    4. 追加所有新列
    5. 保留未涉及的列

    复杂度: O(N * M)，N=行数，M=列数
    """
    existing_cols = set(existing_df.columns)
    new_cols = set(new_df.columns)

    # 分类（基于字符串精确匹配）
    to_replace = existing_cols & new_cols    # 交集：相同表达式 → 替换
    to_append = new_cols - existing_cols     # 差集：不同表达式 → 追加
    to_keep = existing_cols - new_cols       # 差集：未请求 → 保留

    # 合并
    result_df = existing_df.drop(columns=list(to_replace))
    result_df = pd.concat([result_df, new_df], axis=1)

    return result_df, {
        'replaced': list(to_replace),
        'appended': list(to_append),
        'kept': list(to_keep)
    }
```

**关键示例**：
```python
# 示例1：不同表达式 = 追加
existing_cols = ["Ref($close,60)/$close"]
new_cols = ["Ref($close,30)/$close"]
# → to_replace = []（无交集）
# → to_append = ["Ref($close,30)/$close"]
# → to_keep = ["Ref($close,60)/$close"]

# 示例2：相同表达式 = 替换
existing_cols = ["Ref($close,60)/$close"]
new_cols = ["Ref($close,60)/$close"]
# → to_replace = ["Ref($close,60)/$close"]
# → to_append = []
# → to_keep = []
```

#### 列存在性检查算法

```python
def check_columns(path, required_columns):
    """
    检查请求的列是否都存在

    优化：只读元数据，不读数据
    """
    import pyarrow.parquet as pq

    # 只读 schema（元数据），不读数据
    schema = pq.read_schema(path)
    existing_cols = set(schema.names)

    missing_cols = set(required_columns) - existing_cols

    return {
        'exists': len(missing_cols) == 0,
        'missing': sorted(missing_cols),
        'available': sorted(existing_cols)
    }
```

### 7.3 性能优化

| 优化点 | 技术方案 | 性能提升 |
|-------|---------|---------|
| **元数据读取** | 只读 schema，不读数据 | 100x（大文件） |
| **部分列读取** | 使用 `columns` 参数 | 2-10x |
| **增量更新** | 只计算变化的因子 | 2-5x |
| **风格数据复用** | 跨市场复用，提取一次全市场数据 | 3-10x（取决于市场数量） |
| **收益率复用** | 跨市场复用，避免重复计算 | 3-10x（取决于市场数量） |

---

## 8. 实施计划

### 8.1 实施阶段

#### 阶段1：核心功能实现（预计 1.5 小时）

**任务列表**:

| 任务编号 | 任务描述 | 预计时间 | 负责人 |
|---------|---------|---------|--------|
| T-1.1 | 备份现有代码 | 5分钟 | - |
| T-1.2 | 修改 `write_dataframe` 方法（添加 styles 跨市场复用） | 25分钟 | - |
| T-1.3 | 增强 `read_dataframe` 方法（添加 styles 跨市场读取） | 15分钟 | - |
| T-1.4 | 新增 `check_columns` 方法 | 15分钟 | - |
| T-1.5 | 修改 Step1 代码（使用 all 市场提取 styles） | 10分钟 | - |
| T-1.6 | 代码格式化和注释 | 5分钟 | - |

**产出**:
- ✅ 修改后的 `utils/cache_manager.py`
- ✅ 修改后的 `step1/因子提取与预处理.py`

---

#### 阶段2：文档更新（预计 30 分钟）

**任务列表**:

| 任务编号 | 任务描述 | 预计时间 |
|---------|---------|---------|
| T-2.1 | 更新测试计划（TEST_PLAN.md） | 15分钟 |
| T-2.2 | 更新使用文档（README.md） | 15分钟 |

**产出**:
- ✅ 更新后的文档

---

### 8.2 验收标准

#### 功能验收

- ✅ FR-01: 智能写入功能正常
  - 新因子自动追加
  - 已有因子自动替换
  - 混合场景正确处理
- ✅ FR-02: 智能读取功能正常
  - 部分列读取正确
  - 错误提示友好
- ✅ FR-03: 风格数据跨市场复用正常
  - 风格数据统一保存为 all 市场
  - 覆盖全市场股票（非固定成份股）
  - 跨市场自动复用
- ✅ FR-04: 全链路支持正常
  - Step1/2/3/4 都无需修改（除 Step1 的 styles 提取部分）
  - 自动继承智能机制

#### 性能验收

- ✅ 100个因子文件，追加1列 < 10秒
- ✅ 元数据读取 < 0.1秒
- ✅ 无性能退化
- ✅ 风格数据跨市场复用生效（多市场场景性能提升 3-10x）

#### 兼容性验收

- ✅ 不修改 CLI 参数
- ✅ 不修改文件命名规则
- ✅ 向后兼容

---

## 9. 附录

### 9.1 术语表

| 术语 | 定义 |
|-----|------|
| **因子列** | Parquet 文件中存储的因子数据列（如 `Ref($close,60)/$close`） |
| **因子身份** | 完整的表达式字符串（精确匹配），`Ref($close,60)` ≠ `Ref($close,30)` |
| **智能追加** | 检测到不同的表达式字符串 → 追加新列 |
| **智能替换** | 检测到相同的表达式字符串 → 重新计算并替换 |
| **智能合并** = 智能追加 + 智能替换 + 保留旧列 |
| **元数据** | Parquet 文件的 schema（列名、类型等） |
| **跨市场复用** | 不同市场共享同一个收益率/风格数据文件 |
| **风格数据** | 市值（$total_mv）、行业（$industry）、流通市值（$float_mv）等基础数据 |

---

### 9.2 参考资料

- [Pandas DataFrame 文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [PyParquet 文档](https://arrow.apache.org/docs/python/parquet.html)
- 当前 `utils/cache_manager.py` 源码
- 当前 `tests/TEST_PLAN.md`

---

### 9.3 变更历史

| 版本 | 日期 | 变更内容 | 作者 |
|-----|------|---------|------|
| v1.0 | 2026-01-18 | 初始版本 | Claude |
| v1.1 | 2026-01-18 | 新增 FR-03：风格数据跨市场复用需求 | Claude |
| v1.2 | 2026-01-18 | 移除测试相关章节，简化文档结构 | Claude |
| v1.3 | 2026-01-18 | **重大修正**：明确因子身份 = 严格的表达式字符串匹配，修正所有场景描述 | Claude |

---

**文档结束**
