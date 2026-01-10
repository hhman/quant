# Step1: 因子提取与预处理

## 📋 概述

**Step1** 是量化因子分析流程的核心步骤，负责从Qlib数据中提取因子值，并进行去极值和标准化处理，为后续的因子分析做好准备。

### 核心功能

1. **因子提取** - 使用Qlib表达式引擎计算因子值
2. **去极值处理** - 使用3σ标准去除异常值
3. **Z-Score标准化** - 对因子进行截面标准化
4. **多周期收益率计算** - 计算不同持仓周期的未来收益率
5. **风格变量提取** - 提取市值、行业等风格因子

### 依赖关系

- **前置依赖**: Step0（Qlib数据）
- **后续依赖**: Step2（因子中性化）依赖Step1的因子数据
- **独立执行**: 可独立运行，但需要Qlib数据

## 📥 输入数据

### 必需数据

1. **Qlib数据目录** (`provider_uri`)
   - 由Step0生成
   - 默认路径: `cache/qlib_data`

2. **市场定义**
   - 股票池: `--market` (如 csi300)
   - 对应的成分股文件: `qlib_data/instruments/{market}.txt`

### Qlib字段

Step1会自动提取以下基础字段:

**价格字段**:
- `$open` - 开盘价
- `$high` - 最高价
- `$low` - 最低价
- `$close` - 收盘价
- `$volume` - 成交量
- `$amount` - 成交额

**市值字段**:
- `$total_mv` - 总市值
- `$float_mv` - 流通市值

**行业字段**:
- `$industry` - 行业分类（整数ID）

## ⚙️ 输入参数

### CLI参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识（股票池） |
| `--start-date` | str | None | 否 | 起始日期 |
| `--end-date` | str | None | 否 | 结束日期 |
| `--factor-formulas` | str | None | **是** | 因子表达式列表，逗号分隔 |
| `--periods` | str | None | **是** | 收益率周期，逗号分隔 |
| `--provider-uri` | str | "~/data/qlib_data" | 否 | Qlib数据路径 |
| `--cache-dir` | str | "cache" | 否 | Cache目录 |
| `--force-regenerate` | flag | False | 否 | 强制重新生成 |
| `--verbose` | flag | False | 否 | 显示详细输出 |
| `--dry-run` | flag | False | 否 | 模拟运行 |

### 因子表达式 (`--factor-formulas`)

**格式**: Qlib表达式字符串，逗号分隔

**支持的运算**:
- 基础运算: `+`, `-`, `*`, `/`
- 函数: `Ref()`, `Mean()`, `Std()`, `Max()`, `Min()`, `Rank()`, `Abs()`
- 条件: `If()`
- 时间序列: `Mom()`, ` Roc()`, `Sum()`

**示例**:
```bash
# 单个因子
--factor-formulas "Ref($close,60)/$close"

# 多个因子
--factor-formulas "Ref($close,60)/$close,MOM($close,20),Mean($volume,5)"

# 复杂表达式
--factor-formulas "Rank($close)-Rank($total_mv)"
```

**常用因子库**:
- 动量因子: `Ref($close,20)/$close - 1`
- 波动率因子: `Std($close,20)/Mean($close,20)`
- 换手率因子: `$volume/Mean($volume,20) - 1`
- 市值因子: `Rank($total_mv)`
- 反转因子: `-Ref($close,5)/$close`

### 收益率周期 (`--periods`)

**格式**: 逗号分隔的周期字符串

**支持的周期**:
- `1d` - 1日
- `1w` - 5日（约1周）
- `1m` - 20日（约1月）
- `1q` - 60日（约1季度）
- `1y` - 252日（约1年）

**示例**:
```bash
# 单周期
--periods 1d

# 多周期
--periods 1d,1w,1m,1q

# 长周期
--periods 1m,1q,1y
```

### Step1专用参数

**必需参数说明**:
- `--factor-formulas`: **必须指定**，因子表达式列表
- `--periods`: **必须指定**，收益率周期列表

**可选参数说明**:
- `--start-date`: 不指定则使用Qlib数据的全部日期范围
- `--end-date`: 不指定则使用Qlib数据的全部日期范围
- `--market`: 默认为 "csi300"，需在 `qlib_data/instruments/` 中有对应文件

## 📤 输出结果

### 目录结构

```
cache/
├── factor_raw.parquet              # 原始因子值（去极值后）
├── factor_standardized.parquet     # 标准化因子值
├── data_styles.parquet             # 风格变量（市值、行业）
├── data_returns.parquet            # 未来收益率
└── step1_metadata.json             # Step1元数据
```

### 输出文件说明

#### 1. 原始因子值 (`factor_raw.parquet`)

**格式**: MultiIndex DataFrame

**索引**:
- `instrument` - 股票代码
- `datetime` - 日期

**列**: 因子表达式列
- 每个因子表达式对应一列
- 列名为表达式字符串

**值**: 去极值后的因子值

**示例**:
```
                              Ref($close,60)/$close  MOM($close,20)
instrument  datetime
SH600000    2020-01-02              0.987654        0.023456
            2020-01-03              0.989012        0.024567
SZ000001    2020-01-02              1.012345        0.034567
            2020-01-03              1.013456        0.035678
```

#### 2. 标准化因子值 (`factor_standardized.parquet`)

**格式**: 与 `factor_raw.parquet` 相同

**处理**:
- 对每个因子进行Z-Score标准化
- 公式: `(x - mean) / std`
- 按日期截面计算均值和标准差

**值**: 标准化后的因子值（均值≈0，标准差≈1）

#### 3. 风格变量 (`data_styles.parquet`)

**格式**: MultiIndex DataFrame

**列**:
- `$total_mv` - 总市值
- `$float_mv` - 流通市值
- `$industry` - 行业ID（整数）

**用途**:
- Step2用于中性化回归
- Step3用于收益回归

#### 4. 未来收益率 (`data_returns.parquet`)

**格式**: MultiIndex DataFrame

**列**: `ret_{period}`
- `ret_1d` - 1日收益率
- `ret_1w` - 5日收益率
- `ret_1m` - 20日收益率
- 等等...

**计算**:
- `ret_1d = (下日收盘 / 今日收盘) - 1`
- `ret_5d = (5日后收盘 / 今日收盘) - 1`

**对数收益率**:
- 使用对数收益率: `ln(未来价格/当前价格)`

**示例**:
```
                              ret_1d    ret_1w    ret_1m
instrument  datetime
SH600000    2020-01-02     0.01234   0.05678   0.08901
            2020-01-03    -0.00890   0.03456   0.06789
```

#### 5. Step1元数据 (`step1_metadata.json`)

```json
{
  "market": "csi300",
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "factor_formulas": [
    "Ref($close,60)/$close",
    "MOM($close,20)"
  ],
  "periods": {
    "1d": 1,
    "1w": 5,
    "1m": 20
  },
  "generated_at": "2025-01-10T12:00:00Z",
  "stage": "step1"
}
```

## 🚀 使用示例

### 基本用法

```bash
# 提取单因子，单周期
python step1/因子提取与预处理.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close" \
  --periods 1d \
  --provider-uri cache/qlib_data
```

### 多因子多周期

```bash
# 提取多个因子，多个收益率周期
python step1/因子提取与预处理.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close,MOM($close,20),Mean($volume,5)" \
  --periods 1d,1w,1m \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --verbose
```

### 使用全部数据

```bash
# 不指定日期范围，使用Qlib数据的全部日期
python step1/因子提取与预处理.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close" \
  --periods 1d,1w,1m
```

### 模拟运行

```bash
# 查看配置信息，不实际执行
python step1/因子提取与预处理.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close" \
  --periods 1d \
  --dry-run
```

### 强制重新生成

```bash
# 覆盖已有cache
python step1/因子提取与预处理.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close" \
  --periods 1d \
  --force-regenerate
```

## 🔄 执行流程

Step1按以下顺序执行:

1. **初始化Qlib**
   - 加载Qlib数据
   - 设置region (REG_CN)

2. **校验参数**
   - 检查 `--factor-formulas` 和 `--periods` 是否指定
   - 检查cache是否存在（除非 `--force-regenerate`）

3. **计算因子值**
   - 使用Qlib表达式引擎
   - 逐日计算因子值
   - 支持多因子并行计算

4. **去极值处理**
   - 使用3σ标准
   - 截面处理（按日期）
   - 保留大部分数据（约99.7%）

5. **Z-Score标准化**
   - 按日期截面计算均值和标准差
   - 标准化因子值
   - 便于后续分析

6. **提取风格变量**
   - 总市值、流通市值
   - 行业分类
   - 用于中性化

7. **计算未来收益率**
   - 计算多个周期的收益率
   - 对数收益率
   - 用于回归分析

8. **保存数据**
   - 4个parquet文件
   - 元数据JSON
   - 支持智能子集匹配

## ⚠️ 注意事项

### 因子表达式

1. **语法正确**: 确保表达式符合Qlib语法
2. **字段存在**: 引用的字段必须在Qlib数据中存在
3. **时间对齐**: 注意时间序列函数的对齐方式
4. **复杂数度**: 复杂表达式计算时间较长

### 数据质量

1. **缺失值**: Qlib会自动处理部分缺失值
2. **异常值**: 3σ去极值会处理大部分异常
3. **标准化**: Z-Score后因子分布接近标准正态

### 性能优化

1. **多因子**: 一次性计算多个因子效率更高
2. **内存占用**: 大数据集注意内存使用
3. **并行计算**: Qlib支持部分并行计算

### Cache管理

1. **增量更新**: 不支持增量更新，需全部重新计算
2. **子集匹配**: Step2-4可使用Step1的子集
3. **强制覆盖**: 使用 `--force-regenerate` 覆盖已有cache

## 🔗 后续步骤

Step1完成后，可以继续执行:

```bash
# Step2: 因子中性化
python step2/因子中性化.py --market csi300
```

或使用子集:

```bash
# 只分析部分因子
python step2/因子中性化.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close"

# 只分析部分周期
python step2/因子中性化.py \
  --market csi300 \
  --periods 1d

# 只分析部分时间段
python step2/因子中性化.py \
  --market csi300 \
  --start-date 2022-01-01 \
  --end-date 2023-01-01
```

## 🤖 Agent Skill转换指南

### Skill定义

**Skill名称**: `factor_extraction`

**功能描述**: 从Qlib数据中提取因子，进行去极值和标准化

### 输入参数

```python
{
    "market": "csi300",                      # str: 市场标识
    "start_date": "2020-01-01",              # str: 起始日期
    "end_date": "2024-12-31",                # str: 结束日期
    "factor_formulas": [                     # List[str]: 因子表达式
        "Ref($close,60)/$close",
        "MOM($close,20)"
    ],
    "periods": ["1d", "1w", "1m"],           # List[str]: 收益率周期
    "provider_uri": "cache/qlib_data",       # str: Qlib数据路径
    "cache_dir": "cache",                    # str: 输出目录
    "force_regenerate": false,               # bool: 强制重新生成
    "verbose": false                         # bool: 详细输出
}
```

### 输出数据

```python
{
    "factor_raw": "cache/factor_raw.parquet",           # str: 原始因子文件
    "factor_standardized": "cache/factor_standardized.parquet",  # str: 标准化因子文件
    "data_styles": "cache/data_styles.parquet",         # str: 风格变量文件
    "data_returns": "cache/data_returns.parquet",       # str: 收益率文件
    "step1_metadata": "cache/step1_metadata.json",      # str: 元数据文件
    "status": "success",                                # str: 执行状态
    "message": "因子提取完成，共2个因子"                  # str: 执行信息
}
```

### 依赖关系

- **前置**: Step0（Qlib数据）
- **后置**: Step2（因子中性化）、Step3（因子收益）

### 执行方式

```bash
# Python脚本
python step1/因子提取与预处理.py [参数]
```

### 验证方法

```python
import pandas as pd
from pathlib import Path

def verify_step1(cache_dir="cache"):
    """验证Step1输出"""
    factor_file = Path(cache_dir) / "factor_standardized.parquet"
    returns_file = Path(cache_dir) / "data_returns.parquet"
    metadata_file = Path(cache_dir) / "step1_metadata.json"

    assert factor_file.exists(), "因子文件不存在"
    assert returns_file.exists(), "收益率文件不存在"
    assert metadata_file.exists(), "元数据文件不存在"

    # 检查数据格式
    factor_df = pd.read_parquet(factor_file)
    assert factor_df.index.names == ["instrument", "datetime"], "索引格式不正确"

    # 检查标准化
    means = factor_df.mean()
    stds = factor_df.std()
    assert all(abs(means) < 0.1), "因子标准化不正确（均值应接近0）"
    assert all(abs(stds - 1.0) < 0.1), "因子标准化不正确（标准差应接近1）"

    return True
```

## 📚 相关文件

- `step1/因子提取与预处理.py` - 主脚本
- `cli_config.py` - CLI参数解析
- `cache_manager.py` - Cache管理
- `factor_analysis.py` - 因子处理函数（ext_out_3std, z_score）
- `metadata.py` - 元数据管理
