# Step2: 因子中性化

## 📋 概述

**Step2** 是量化因子分析流程的关键步骤，负责对标准化后的因子进行行业和市值中性化处理，消除因子中的风格暴露，使因子更加纯粹。

### 核心功能

1. **行业中性化** - 剔除因子中的行业效应
2. **市值中性化** - 剔除因子中的市值效应
3. **WLS回归** - 使用加权最小二乘法，权重为流通市值平方根
4. **智能子集匹配** - 支持从Step1 cache中选取因子子集

### 依赖关系

- **前置依赖**: Step1（标准化因子数据）
- **后续依赖**: Step3（因子收益率）、Step4（因子绩效评估）
- **独立执行**: 依赖Step1，支持子集匹配

## 📥 输入数据

### 必需数据

1. **Step1输出文件**:
   - `factor_standardized.parquet` - 标准化因子值
   - `data_styles.parquet` - 风格变量（市值、行业）
   - `step1_metadata.json` - Step1元数据

2. **Cache目录** (`--cache-dir`)
   - 默认: `cache`

### 数据格式要求

**输入索引**: MultiIndex (instrument, datetime)

**输入列**:
- 因子列: 需要中性化的因子
- 风格列:
  - `$total_mv` - 总市值
  - `$industry` - 行业ID
  - `$float_mv` - 流通市值（用于权重）

## ⚙️ 输入参数

### CLI参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识（需与Step1一致） |
| `--start-date` | str | None | 否 | 起始日期（子集） |
| `--end-date` | str | None | 否 | 结束日期（子集） |
| `--factor-formulas` | str | None | 否 | 因子表达式列表（子集） |
| `--periods` | str | None | 否 | 收益率周期（子集） |
| `--cache-dir` | str | "cache" | 否 | Cache目录 |
| `--verbose` | flag | False | 否 | 显示详细输出 |
| `--dry-run` | flag | False | 否 | 模拟运行 |

### 参数说明

#### 市场标识 (`--market`)

**用途**: 校验Step1 cache的市场标识

**示例**:
```bash
# 使用csi300市场的cache
--market csi300
```

#### 时间范围 (`--start-date`, `--end-date`)

**用途**: 从Step1 cache中选取时间段

**默认**: 使用Step1的全部时间范围

**示例**:
```bash
# 只分析2022-2023年
--start-date 2022-01-01 --end-date 2023-01-01
```

#### 因子子集 (`--factor-formulas`)

**用途**: 从Step1 cache中选取部分因子

**默认**: 使用Step1的全部因子

**示例**:
```bash
# 只中性化一个因子
--factor-formulas "Ref($close,60)/$close"

# 中性化多个因子
--factor-formulas "Ref($close,60)/$close,MOM($close,20)"
```

**注意**: 指定的因子必须在Step1 cache中存在

#### 周期子集 (`--periods`)

**用途**: 从Step1 cache中选取部分周期

**默认**: 使用Step1的全部周期

**示例**:
```bash
# 只使用1d周期
--periods 1d
```

**注意**: Step2不使用周期参数，但会传递给后续step

## 📤 输出结果

### 目录结构

```
cache/
├── factor_行业市值中性化.parquet    # 中性化后的因子
└── step2_metadata.json              # Step2元数据
```

### 输出文件说明

#### 1. 中性化因子 (`factor_行业市值中性化.parquet`)

**格式**: MultiIndex DataFrame

**索引**:
- `instrument` - 股票代码
- `datetime` - 日期

**列**: 因子表达式列
- 与Step1的因子列相同
- 只包含指定的因子子集

**值**: 回归残差（中性化后的因子）

**示例**:
```
                              Ref($close,60)/$close  MOM($close,20)
instrument  datetime
SH600000    2020-01-02              0.012345        -0.023456
            2020-01-03              0.034567        -0.012345
SZ000001    2020-01-02             -0.023456         0.045678
            2020-01-03             -0.012345         0.034567
```

**统计特性**:
- 均值 ≈ 0（中性化后）
- 与市值、行业的相关性接近0
- 保留了因子的独特信息

#### 2. Step2元数据 (`step2_metadata.json`)

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
  "neutralization_method": "industry_marketcap",
  "regression_weights": "sqrt(float_mv)",
  "generated_at": "2025-01-10T12:00:00Z",
  "stage": "step2"
}
```

## 🚀 使用示例

### 基本用法

```bash
# 使用Step1的全部数据
python step2/因子中性化.py --market csi300
```

### 使用子集匹配

```bash
# 只分析部分因子
python step2/因子中性化.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close"

# 只分析部分时间段
python step2/因子中性化.py \
  --market csi300 \
  --start-date 2022-01-01 \
  --end-date 2023-01-01

# 组合使用子集
python step2/因子中性化.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close,MOM($close,20)" \
  --start-date 2022-01-01 \
  --end-date 2023-01-01 \
  --verbose
```

### 模拟运行

```bash
# 查看将要执行的操作
python step2/因子中性化.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close" \
  --dry-run
```

## 🔄 执行流程

Step2按以下顺序执行:

1. **校验Step1 cache**
   - 检查市场标识是否匹配
   - 检查因子子集是否在cache中
   - 检查时间范围是否在cache内
   - 支持智能子集匹配

2. **加载数据**
   - 加载标准化因子 (`factor_standardized.parquet`)
   - 加载风格变量 (`data_styles.parquet`)
   - 应用时间范围和因子子集过滤

3. **逐日中性化**
   - 按日期分组
   - 对每个日期执行中性化回归

4. **中性化回归**
   - 回归模型: `factor ~ log(total_mv) + industry_dummies`
   - 权重: `sqrt(float_mv)`
   - 提取残差作为中性化后的因子

5. **保存结果**
   - 中性化因子文件
   - 元数据文件
   - 记录使用的子集

## ⚠️ 注意事项

### 回归方法

1. **WLS回归**: 使用加权最小二乘法
2. **权重**: 流通市值平方根
3. **行业哑变量**: 使用one-hot编码
4. **截距项**: 自动添加

### 数据质量

1. **缺失值处理**: 回归前删除缺失值
2. **样本要求**: 样本数需大于参数数
3. **多重共线性**: 行业和市值可能存在相关性
4. **异常值**: 极端值可能影响回归结果

### 性能优化

1. **逐日处理**: 内存占用低
2. **并行计算**: 可并行处理不同日期
3. **大数据集**: 注意回归时间

### 中性化效果

1. **检查残差**: 中性化后因子应与风格变量无关
2. **R²检验**: 回归R²不应太高（<0.5为宜）
3. **残差分布**: 应接近正态分布

## 🔗 后续步骤

Step2完成后，可以继续执行:

```bash
# Step3: 因子收益率计算
python step3/因子收益率计算.py --market csi300

# Step4: 因子绩效评估
python step4/因子绩效评估.py --market csi300
```

或使用子集:

```bash
# 只分析Step2使用的因子子集
python step3/因子收益率计算.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close"
```

## 🤖 Agent Skill转换指南

### Skill定义

**Skill名称**: `factor_neutralization`

**功能描述**: 对标准化因子进行行业和市值中性化

### 输入参数

```python
{
    "market": "csi300",                      # str: 市场标识
    "start_date": "2020-01-01",              # str: 起始日期（可选）
    "end_date": "2024-12-31",                # str: 结束日期（可选）
    "factor_formulas": [                     # List[str]: 因子子集（可选）
        "Ref($close,60)/$close"
    ],
    "periods": ["1d", "1w", "1m"],           # List[str]: 周期子集（可选）
    "cache_dir": "cache",                    # str: Cache目录
    "verbose": false                         # bool: 详细输出
}
```

### 输出数据

```python
{
    "factor_neutralized": "cache/factor_行业市值中性化.parquet",  # str: 中性化因子文件
    "step2_metadata": "cache/step2_metadata.json",                # str: 元数据文件
    "status": "success",                                           # str: 执行状态
    "message": "因子中性化完成，共1个因子"                           # str: 执行信息
}
```

### 依赖关系

- **前置**: Step1（标准化因子、风格变量）
- **后置**: Step3（因子收益率）、Step4（因子绩效评估）

### 执行方式

```bash
# Python脚本
python step2/因子中性化.py [参数]
```

### 验证方法

```python
import pandas as pd
import numpy as np
from pathlib import Path

def verify_step2(cache_dir="cache"):
    """验证Step2输出"""
    factor_file = Path(cache_dir) / "factor_行业市值中性化.parquet"
    styles_file = Path(cache_dir) / "data_styles.parquet"
    metadata_file = Path(cache_dir) / "step2_metadata.json"

    assert factor_file.exists(), "中性化因子文件不存在"
    assert styles_file.exists(), "风格变量文件不存在"
    assert metadata_file.exists(), "元数据文件不存在"

    # 检查中性化效果
    factor_df = pd.read_parquet(factor_file)
    styles_df = pd.read_parquet(styles_file)

    # 合并数据
    merged = factor_df.join(styles_df)

    # 检查与市值的相关性
    for factor_col in factor_df.columns:
        corr = merged[factor_col].corr(np.log(merged["$total_mv"]))
        assert abs(corr) < 0.1, f"{factor_col} 与市值相关性过高: {corr}"

    # 检查均值接近0
    means = factor_df.mean()
    assert all(abs(means) < 0.1), "中性化因子均值应接近0"

    return True
```

## 📚 相关文件

- `step2/因子中性化.py` - 主脚本
- `cli_config.py` - CLI参数解析
- `cache_manager.py` - Cache管理（智能子集匹配）
- `factor_analysis.py` - 中性化函数（neutralize_industry_marketcap）
- `metadata.py` - 元数据管理
