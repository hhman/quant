# T5: IC评估新增t统计量 - 需求文档

## 📋 需求概述

### 需求背景
当前框架IC评估输出包含均值、标准差、IR、IC>0占比等指标，但缺少t统计量。华泰论文标准（第5页P5）要求计算IC的t检验。

### 需求目标
在IC汇总统计中新增t统计量，用于判断IC是否显著异于0。

### 设计原则
- **最小改动原则**：在现有_summary函数中新增
- **统一应用原则**：同时应用于IC和RankIC

---

## 🔧 改动方：core/factor_analysis.py

### 文件：`core/factor_analysis.py`

#### 修改点：_summary函数新增t统计量

**位置**：Line 280-297

**原代码**：
```python
def _summary(series: pd.Series) -> pd.Series:
    s = pd.Series(series).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    mean = s.mean()
    std = s.std()
    icir = mean / std if std != 0 else np.nan
    win_rate = (s > 0).sum() / len(s)
    gt002_rate = (s.abs() > 0.02).sum() / len(s)
    return pd.Series({
        "IC序列均值": mean,
        "IC序列标准差": std,
        "IR比率": icir,
        "IC>0占比": win_rate,
        "|IC|>0.02占比": gt002_rate,
    })
```

**修改为**：
```python
def _summary(series: pd.Series) -> pd.Series:
    """IC序列统计摘要（含t统计量）"""
    s = pd.Series(series).dropna()
    if s.empty:
        return pd.Series(dtype=float)

    mean = s.mean()
    std = s.std()
    T = len(s)  # 样本数

    # 新增：t统计量
    # 公式：t = mean(IC) / std(IC) × √T
    t_stat = mean / std * np.sqrt(T) if std != 0 else np.nan

    icir = mean / std if std != 0 else np.nan
    win_rate = (s > 0).sum() / len(s)
    gt002_rate = (s.abs() > 0.02).sum() / len(s)

    return pd.Series({
        "IC序列均值": mean,
        "IC序列标准差": std,
        "IR比率": icir,
        "IC t统计量": t_stat,  # 新增
        "IC>0占比": win_rate,
        "|IC|>0.02占比": gt002_rate,
    })
```

---

## 📊 t统计量说明

### 计算公式
```
t = mean(IC) / std(IC) × √T
```

**参数说明**：
- `mean(IC)`：IC序列均值
- `std(IC)`：IC序列标准差
- `T`：IC序列样本数
- `√T`：根号下样本数

### 统计意义
- **t > 2**：IC在95%置信水平下显著（拒绝"IC=0"的原假设）
- **t < -2**：IC在95%置信水平下显著为负
- **|t| < 2**：IC不显著（可能由随机波动导致）

### 示例
假设某因子的IC序列：
- IC均值 = 0.03
- IC标准差 = 0.15
- 样本数 T = 1000

计算：
```
t = 0.03 / 0.15 × √1000 = 0.2 × 31.62 = 6.32
```

**结论**：t = 6.32 > 2，IC显著为正，因子有效。

---

## ✅ 验收标准

### 功能验收
1. ✅ IC汇总输出包含"IC t统计量"列
2. ✅ RankIC汇总输出包含"IC t统计量"列
3. ✅ t统计量计算公式正确（mean/std × √T）

### 数据质量验收
1. ✅ t统计量数值合理（通常在-10到10之间）
2. ✅ 显著因子的t统计量绝对值>2

### 回归验收
1. ✅ Step4正常运行
2. ✅ Excel汇总报告正确输出

---

## 📝 输出示例

### Excel汇总报告格式

**ic_summary.xlsx**：

| 因子 | IC序列均值 | IC序列标准差 | IR比率 | IC t统计量 | IC>0占比 | |IC|>0.02占比 |
|------|-----------|-------------|--------|-----------|---------|------------|
| PE_TTM | 0.030 | 0.150 | 0.200 | 6.32 | 0.550 | 0.420 |

**rank_ic_summary.xlsx**：

| 因子 | IC序列均值 | IC序列标准差 | IR比率 | IC t统计量 | IC>0占比 | |IC|>0.02占比 |
|------|-----------|-------------|--------|-----------|---------|------------|
| PE_TTM | 0.025 | 0.140 | 0.179 | 5.65 | 0.580 | 0.380 |

---

## 📚 参考文档

- 华泰多因子系列2：单因子测试之估值类因子.pdf（第5页P5，IC的t检验）
- `core/factor_analysis.py`（Line 280-297：_summary函数）
- `core/factor_analysis.py`（Line 255-317：summarize_ic函数）
