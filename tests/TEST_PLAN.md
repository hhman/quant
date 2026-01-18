# 量化因子分析流程 - 测试计划

## ⚠️ 测试执行人员必读（最高优先级）

### 三条铁律（严格执行，不得违反）

1. **严格按照文档命令执行，不擅自修改参数**
2. **执行所有验证步骤，不得跳过**
3. **遇到错误立即停下来，等用户确认后再执行**

### 违反后果
- 擅自修改命令 → 测试无效，需要重新执行
- 跳过验证步骤 → 无法确认功能是否正常
- 遇到错误继续执行 → 可能掩盖真实问题

---

## 测试前准备

### 清理缓存
```bash
rm -rf .cache/
```

### 数据检查
确保以下目录存在：
- `stock/` - 股票日线数据
- `index/` - 指数日线数据
- `finance/` - 财务数据

## 测试依赖关系

```
Test 1 (Step0)
   ↓
Test 2 (Step1) ←────────┐
   ↓                     │
Test 3 (Step2)           │ Test 6 (Cache机制)
   ↓                     │
Test 4 (Step3) ←─────────┤
   ↓                     │
Test 5 (Step4)           │ Test 7 (错误检测)
```

## 错误处理原则

遇到错误立即停止，向用户报告错误信息、可能原因、影响范围，等待用户明确指令后重新执行。

## Test 1: Step0 数据预处理

**命令**:
```bash
conda activate quant && bash step0/cli.sh \
  --start-date 2023-01-01 \
  --end-date 2024-01-01
```

**验证**:
```bash
ls -d .cache/qlib_data && echo "✅ 通过" || echo "❌ 失败"
```

## Test 2: Step1 因子提取

**命令**:
```bash
conda activate quant && python step1/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Ref(\$close,60)/\$close"
```

**验证**:
```bash
ls .cache/csi300_20230101_20240101__factor_raw.parquet && \
ls .cache/csi300_20230101_20240101__factor_std.parquet && \
ls .cache/all_20230101_20240101__returns.parquet && \
ls .cache/csi300_20230101_20240101__styles.parquet && \
echo "✅ 通过" || echo "❌ 失败"
```

## Test 3: Step2 因子中性化

**命令**:
```bash
conda activate quant && python step2/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Ref(\$close,60)/\$close"
```

**验证**:
```bash
ls .cache/csi300_20230101_20240101__neutralized.parquet && \
echo "✅ 通过" || echo "❌ 失败"
```

## Test 4: Step3 因子收益率

**命令**:
```bash
conda activate quant && python step3/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Ref(\$close,60)/\$close"
```

**验证**:
```bash
ls .cache/csi300_20230101_20240101__return_coef.parquet && \
ls .cache/csi300_20230101_20240101__return_tval.parquet && \
ls .cache/csi300_20230101_20240101__return_coef_summary.xlsx && \
ls .cache/csi300_20230101_20240101__return_tval_summary.xlsx && \
echo "✅ 通过" || echo "❌ 失败"
```

## Test 5: Step4 因子绩效评估

**命令**:
```bash
conda activate quant && python step4/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Ref(\$close,60)/\$close"
```

**验证**:
```bash
ls .cache/csi300_20230101_20240101__ic.parquet && \
ls .cache/csi300_20230101_20240101__rank_ic.parquet && \
ls .cache/csi300_20230101_20240101__group_return.parquet && \
ls .cache/csi300_20230101_20240101__autocorr.parquet && \
ls .cache/csi300_20230101_20240101__turnover.parquet && \
ls .cache/csi300_20230101_20240101__ic_summary.xlsx && \
ls .cache/csi300_20230101_20240101__rank_ic_summary.xlsx && \
ls .cache/csi300_20230101_20240101__group_return_summary.xlsx && \
ls .cache/csi300_20230101_20240101__autocorr_summary.xlsx && \
ls .cache/csi300_20230101_20240101__turnover_summary.xlsx && \
ls -d .cache/graphs && \
echo "✅ 通过" || echo "❌ 失败"
```

## Test 6: Cache机制验证

### 场景1: 跨市场复用

**命令**:
```bash
# 1. 记录执行前的文件时间
stat -f "%Sm" .cache/all_20230101_20240101__returns.parquet

# 2. 执行sse50市场
conda activate quant && python step1/cli.py \
  --market sse50 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Ref(\$close,60)/\$close"

# 3. 记录执行后的文件时间
stat -f "%Sm" .cache/all_20230101_20240101__returns.parquet
```

**验证**:
```bash
# 对比步骤1和步骤3的文件时间，应该完全相同
# 如果时间相同，说明文件被复用，没有重新计算
```

**预期**: 执行后文件时间与执行前相同（说明复用了cache）

### 场景2: 智能追加因子

**命令**:
```bash
# 1. 记录执行前的文件大小
stat -f "%z" .cache/csi300_20230101_20240101__factor_std.parquet

# 2. 添加新因子
conda activate quant && python step1/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Mean(\$volume,20)"

# 3. 记录执行后的文件大小
stat -f "%z" .cache/csi300_20230101_20240101__factor_std.parquet

# 4. 检查因子数量
python -c "import pandas as pd; df = pd.read_parquet('.cache/csi300_20230101_20240101__factor_std.parquet'); print(f'因子列数: {df.shape[1]}')"
```

**验证**:
```bash
# 步骤3的文件大小应该大于步骤1的文件大小
# 步骤4应该输出: 因子列数: 2
```

**预期**: 文件大小增加，因子列数从1变为2

### 场景3: 智能替换因子

**命令**:
```bash
# 1. 记录执行前的文件时间
stat -f "%Sm" .cache/csi300_20230101_20240101__factor_std.parquet

# 2. 执行完全相同的命令（与场景2相同）
conda activate quant && python step1/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Mean(\$volume,20)"

# 3. 记录执行后的文件时间
stat -f "%Sm" .cache/csi300_20230101_20240101__factor_std.parquet
```

**验证**:
```bash
# 对比步骤1和步骤3的文件时间，时间应该已更新（系统检测到cache已存在，删除旧文件并重新计算）
# 对比步骤1和步骤3的文件大小，应该基本不变（因子数量相同）
```

**预期**: 执行后文件时间已更新，文件大小基本不变（说明系统重新计算了因子，而非直接复用）

## Test 7: 错误检测验证

### 场景1: 不存在的日期范围

**命令**:
```bash
conda activate quant && python step2/cli.py \
  --market csi300 \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --factor-formulas "Ref(\$close,60)/\$close"
```

**验证**:
```bash
# 命令应该失败
# 检查错误输出中是否包含关键词
```

**预期**: 命令失败，错误信息包含"不存在"或"empty"或"为空"

### 场景2: 不存在的市场

**命令**:
```bash
conda activate quant && python step2/cli.py \
  --market csi500 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Ref(\$close,60)/\$close"
```

**验证**:
```bash
# 命令应该失败
# 检查错误输出中是否包含"不在cache中"或"不存在"
```

**预期**: 命令失败，错误信息包含"不在cache中"或"不存在"

### 场景3: 不存在的因子

**命令**:
```bash
conda activate quant && python step2/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Mean(\$volume,30)"
```

**验证**:
```bash
# 命令应该失败
# 检查错误输出中是否包含"不在cache中"或"不存在"
```

**预期**: 命令失败，错误信息包含"不在cache中"或"不存在"

## 预期测试时间

- Test 1-5: 7-11分钟
- Test 6-7: 15-25分钟
- 总计: 22-36分钟
