# Step4: 因子绩效评估

## 📋 概述

**Step4** 是量化因子分析流程的最后一步，全面评估因子的预测能力和绩效表现，包括IC分析、分组收益、自相关性和换手率等。

### 核心功能
1. **IC/RankIC分析** - 评估因子与收益率的相关性
2. **分组收益分析** - 评估因子分组的收益表现
3. **自相关性分析** - 评估因子的稳定性
4. **换手率分析** - 评估因子的稳定性
5. **可视化图表** - 生成绩效分析图表

### 依赖关系
- **前置依赖**: Step1（收益率数据）、Step2（中性化因子）
- **后续依赖**: 无（最后一步）
- **独立执行**: 依赖Step1和Step2，支持子集匹配

## 📥 输入数据
- `data_returns.parquet` - 未来收益率（Step1）
- `factor_行业市值中性化.parquet` - 中性化因子（Step2）

## ⚙️ 输入参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识 |
| `--start-date` | str | None | 否 | 起始日期（子集） |
| `--end-date` | str | None | 否 | 结束日期（子集） |
| `--factor-formulas` | str | None | 否 | 因子子集 |
| `--periods` | str | None | 否 | 周期子集 |
| `--provider-uri` | str | "~/data/qlib_data" | 否 | Qlib数据路径 |
| `--cache-dir` | str | "cache" | 否 | Cache目录 |
| `--verbose` | flag | False | 否 | 详细输出 |
| `--dry-run` | flag | False | 否 | 模拟运行 |

## 📤 输出结果

### 输出文件
```
cache/
├── ic_summary.xlsx                  # IC汇总统计
├── rank_ic_summary.xlsx             # RankIC汇总统计
├── group_return_summary.xlsx        # 分组收益汇总
├── autocorr_summary.xlsx            # 自相关性汇总
├── turnover_summary.xlsx            # 换手率汇总
├── graphs/                          # 可视化图表
│   └── {factor}_{ret_period}/
│       ├── group_return.html
│       ├── pred_ic.html
│       ├── pred_autocorr.html
│       ├── pred_turnover.html
│       └── score_ic.html
└── step4_metadata.json              # Step4元数据
```

### 评估指标说明

#### 1. IC分析 (Information Coefficient)
- **IC**: 因子值与未来收益率的相关系数
- **RankIC**: 因子排名与收益率排名的相关系数
- **IC均值**: 平均IC，|IC|>0.03为有效
- **IC标准差**: IC的波动性
- **ICIR**: IC均值/IC标准差，>0.5为稳定
- **IC>0占比**: IC为正的占比，>0.5为方向正确

#### 2. 分组收益
- **多空收益**: Top组 - Bottom组收益
- **最大回撤**: 多空收益的最大回撤
- **夏普比率**: 收益/波动，>1为优秀
- **胜率**: 正收益日占比

#### 3. 自相关性
- **1日自相关**: 因子t与t-1的相关性
- **5日自相关**: 因子t与t-5的相关性
- **均值**: 平均自相关系数

#### 4. 换手率
- **1日换手率**: 因子Top/Bottom组1日换手率
- **5日换手率**: 因子Top/Bottom组5日换手率
- **平均换手率**: 时间平均

## 🚀 使用示例

```bash
# 基本用法
python step4/因子绩效评估.py \
  --market csi300 \
  --provider-uri cache/qlib_data

# 使用子集
python step4/因子绩效评估.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close" \
  --periods 1d,1w \
  --start-date 2022-01-01 \
  --end-date 2023-01-01 \
  --provider-uri cache/qlib_data

# 详细输出
python step4/因子绩效评估.py \
  --market csi300 \
  --provider-uri cache/qlib_data \
  --verbose
```

## 🔄 执行流程
1. 校验Step2 cache（中性化因子）
2. 加载Step1的收益率数据
3. 计算IC/RankIC
4. 计算分组收益
5. 计算自相关性
6. 计算换手率
7. 生成可视化图表
8. 保存结果和元数据

## ⚠️ 注意事项
- **IC阈值**: |IC均值|>0.03为有效因子
- **ICIR**: >0.5为因子稳定
- **换手率**: 过高说明因子不稳定
- **可视化**: 需要plotly库

## 🔗 完整流程示例

```bash
# Step0: 数据预处理
bash step0/step0.sh --start-date 2008-01-01 --end-date 2024-12-31

# Step1: 因子提取
python step1/因子提取与预处理.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close,MOM($close,20)" \
  --periods 1d,1w,1m \
  --provider-uri cache/qlib_data

# Step2: 因子中性化
python step2/因子中性化.py --market csi300

# Step3: 因子收益率计算
python step3/因子收益率计算.py --market csi300

# Step4: 因子绩效评估
python step4/因子绩效评估.py \
  --market csi300 \
  --provider-uri cache/qlib_data
```

## 🤖 Agent Skill转换指南

### Skill定义
**Skill名称**: `factor_evaluation`

### 输入参数
```python
{
    "market": "csi300",
    "start_date": "2020-01-01",  # 可选
    "end_date": "2024-12-31",    # 可选
    "factor_formulas": [...],    # 可选
    "periods": [...],            # 可选
    "provider_uri": "cache/qlib_data",
    "cache_dir": "cache"
}
```

### 输出数据
```python
{
    "ic_summary": "cache/ic_summary.xlsx",
    "rank_ic_summary": "cache/rank_ic_summary.xlsx",
    "group_return_summary": "cache/group_return_summary.xlsx",
    "autocorr_summary": "cache/autocorr_summary.xlsx",
    "turnover_summary": "cache/turnover_summary.xlsx",
    "graphs_dir": "cache/graphs/",
    "step4_metadata": "cache/step4_metadata.json",
    "status": "success",
    "message": "因子评估完成"
}
```

### 依赖关系
- **前置**: Step1（收益率）、Step2（中性化因子）
- **后置**: 无（最后一步）

### 验证方法
```python
import pandas as pd
from pathlib import Path

def verify_step4(cache_dir="cache"):
    ic_file = Path(cache_dir) / "ic_summary.xlsx"
    assert ic_file.exists()

    ic_df = pd.read_excel(ic_file, index_col=0)

    # 检查IC有效性
    for factor in ic_df.index:
        ic_mean = ic_df.loc[factor, "IC均值"]
        assert abs(ic_mean) > 0.03, f"{factor} IC均值过低: {ic_mean}"

    # 检查图表
    graphs_dir = Path(cache_dir) / "graphs"
    assert graphs_dir.exists()

    return True
```

## 📚 相关文件
- `step4/因子绩效评估.py` - 主脚本
- `factor_analysis.py` - 绩效评估函数
- `cache_manager.py` - Cache管理
- `metadata.py` - 元数据管理

## 📊 评估结果解读

### 优秀因子标准
- **IC均值**: |IC| > 0.05
- **ICIR**: > 0.5
- **IC>0占比**: > 0.6
- **多空收益**: 年化 > 5%
- **夏普比率**: > 1.0
- **最大回撤**: < 20%
- **换手率**: 日换手 < 30%

### 因子分类
- **高IC低换手**: 稳定优秀因子
- **高IC高换手**: 需要考虑交易成本
- **低IC低换手**: 可能适合组合
- **低IC高换手**: 不建议使用
