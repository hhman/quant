# Step3: 因子收益率计算

## 📋 概述

**Step3** 通过回归分析计算因子对未来收益率的预测能力，评估因子的有效性和显著性。

### 核心功能
1. **因子收益回归** - 计算因子对收益率的边际贡献
2. **WLS回归** - 使用流通市值加权
3. **t值检验** - 评估因子收益的统计显著性
4. **多周期分析** - 支持不同持仓周期的收益率

### 依赖关系
- **前置依赖**: Step1（标准化因子、收益率数据、风格变量）
- **后续依赖**: Step4（因子绩效评估）
- **独立执行**: 依赖Step1，支持子集匹配

## 📥 输入数据
- `factor_standardized.parquet` - 标准化因子
- `data_styles.parquet` - 风格变量
- `data_returns.parquet` - 未来收益率

## ⚙️ 输入参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识 |
| `--start-date` | str | None | 否 | 起始日期（子集） |
| `--end-date` | str | None | 否 | 结束日期（子集） |
| `--factor-formulas` | str | None | 否 | 因子子集 |
| `--periods` | str | None | 否 | 周期子集 |
| `--cache-dir` | str | "cache" | 否 | Cache目录 |
| `--verbose` | flag | False | 否 | 详细输出 |
| `--dry-run` | flag | False | 否 | 模拟运行 |

## 📤 输出结果

### 输出文件
```
cache/
├── factor_回归收益率.parquet          # 因子收益率时间序列
├── factor_回归t值.parquet             # t值时间序列
├── factor_回归收益率_summary.xlsx     # 收益率汇总统计
├── factor_回归t值_summary.xlsx        # t值汇总统计
└── step3_metadata.json               # Step3元数据
```

### 回归模型
```
ret = β0 + β1*factor + β2*log(total_mv) + Σβj*industry_j + ε
```

**权重**: `sqrt(float_mv)`

### 输出说明

**因子收益率时间序列**:
- 索引: `datetime`
- 列: `{factor}_{ret_period}` (如 "Ref($close,60)/$close_ret_1d")
- 值: 因子收益率系数β

**t值时间序列**:
- 索引: `datetime`
- 列: 同上
- 值: t统计量

**汇总统计**:
- 因子收益率均值
- 因子收益率序列t检验
- |t|均值
- |t|>2占比

## 🚀 使用示例

```bash
# 基本用法
python step3/因子收益率计算.py --market csi300

# 使用子集
python step3/因子收益率计算.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close" \
  --start-date 2022-01-01 \
  --end-date 2023-01-01

# 详细输出
python step3/因子收益率计算.py --market csi300 --verbose
```

## 🔄 执行流程
1. 校验Step1 cache（支持子集匹配）
2. 加载数据（因子、风格、收益率）
3. 逐日回归（因子 × 收益率周期）
4. 保存结果（系数、t值、汇总）

## ⚠️ 注意事项
- **回归质量**: 检查R²，过低说明拟合质量差
- **t值**: |t|>2表示显著（95%置信度）
- **样本要求**: 每日回归需足够样本

## 🔗 后续步骤
```bash
# Step4: 因子绩效评估
python step4/因子绩效评估.py --market csi300
```

## 🤖 Agent Skill转换指南

### Skill定义
**Skill名称**: `factor_return_analysis`

### 输入参数
```python
{
    "market": "csi300",
    "start_date": "2020-01-01",  # 可选
    "end_date": "2024-12-31",    # 可选
    "factor_formulas": [...],    # 可选
    "periods": [...],            # 可选
    "cache_dir": "cache"
}
```

### 输出数据
```python
{
    "factor_returns": "cache/factor_回归收益率.parquet",
    "factor_tvalues": "cache/factor_回归t值.parquet",
    "returns_summary": "cache/factor_回归收益率_summary.xlsx",
    "tvalues_summary": "cache/factor_回归t值_summary.xlsx",
    "step3_metadata": "cache/step3_metadata.json",
    "status": "success"
}
```

### 验证方法
```python
import pandas as pd
from pathlib import Path

def verify_step3(cache_dir="cache"):
    factor_file = Path(cache_dir) / "factor_回归收益率.parquet"
    assert factor_file.exists()

    df = pd.read_parquet(factor_file)
    assert df.index.name == "datetime"

    # 检查t值显著性
    t_file = Path(cache_dir) / "factor_回归t值.parquet"
    t_df = pd.read_parquet(t_file)
    significant_rate = (t_df.abs() > 2).sum() / len(t_df)
    assert significant_rate > 0.5, "因子显著性不足"

    return True
```

## 📚 相关文件
- `step3/因子收益率计算.py` - 主脚本
- `factor_analysis.py` - 因子收益回归函数
- `cache_manager.py` - Cache管理
