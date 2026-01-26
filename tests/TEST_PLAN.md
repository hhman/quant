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
conda activate quant && python step0/cli.py \
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
ls .cache/all_20230101_20240101__styles.parquet && \
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

### 场景1: 跨市场复用收益率数据

**目的**: 验证不同市场复用 all 市场的 returns 和 styles 文件

**命令与验证**:
```bash
# 1. 记录执行前的文件时间戳
RETURNS_BEFORE=$(stat -f "%m" .cache/all_20230101_20240101__returns.parquet 2>/dev/null || echo "0")
STYLES_BEFORE=$(stat -f "%m" .cache/all_20230101_20240101__styles.parquet 2>/dev/null || echo "0")
echo "执行前时间戳:"
echo "  returns: $RETURNS_BEFORE"
echo "  styles: $STYLES_BEFORE"

# 2. 执行 sse50 市场（不同市场）
echo ""
echo "执行 sse50 市场..."
conda activate quant && python step1/cli.py \
  --market sse50 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Ref(\$close,60)/\$close" 2>&1 | grep -E "(复用|收益率|风格数据|保存)"

# 3. 记录执行后的文件时间戳
RETURNS_AFTER=$(stat -f "%m" .cache/all_20230101_20240101__returns.parquet 2>/dev/null || echo "0")
STYLES_AFTER=$(stat -f "%m" .cache/all_20230101_20240101__styles.parquet 2>/dev/null || echo "0")
echo ""
echo "执行后时间戳:"
echo "  returns: $RETURNS_AFTER"
echo "  styles: $STYLES_AFTER"

# 4. 验证：文件时间戳应该不变（说明复用了cache）
echo ""
if [ "$RETURNS_BEFORE" = "$RETURNS_AFTER" ] && [ "$STYLES_BEFORE" = "$STYLES_AFTER" ]; then
  echo "✅ 通过：returns和styles文件复用成功（时间戳未变化）"
else
  echo "❌ 失败：文件时间戳发生变化，可能未正确复用cache"
fi
```

**预期**:
- 日志包含"复用all市场收益率数据"或"复用all市场风格数据"
- 文件时间戳执行前后保持不变（说明复用了已有文件）

---

### 场景2: 智能追加新因子

**目的**: 验证向已有文件追加新因子的功能

**命令与验证**:
```bash
# 1. 检查当前因子列数
BEFORE_COUNT=$(python -c "import pandas as pd; df = pd.read_parquet('.cache/csi300_20230101_20240101__factor_std.parquet'); print(df.shape[1])")
echo "执行前列数: $BEFORE_COUNT"

# 2. 添加新因子（与Test2中的因子不同）
echo ""
echo "执行追加新因子..."
conda activate quant && python step1/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Mean(\$volume,20)" 2>&1 | grep -E "(追加|因子)"

# 3. 检查追加后的因子列数和列名
AFTER_COUNT=$(python -c "import pandas as pd; df = pd.read_parquet('.cache/csi300_20230101_20240101__factor_std.parquet'); print(df.shape[1])")
FACTOR_COLS=$(python -c "import pandas as pd; df = pd.read_parquet('.cache/csi300_20230101_20240101__factor_std.parquet'); print(list(df.columns))")
echo ""
echo "执行后列数: $AFTER_COUNT"
echo "因子列: $FACTOR_COLS"

# 4. 验证
echo ""
if [ "$AFTER_COUNT" -gt "$BEFORE_COUNT" ]; then
  echo "✅ 通过：成功追加新因子（列数从$BEFORE_COUNT增加到$AFTER_COUNT）"
else
  echo "❌ 失败：列数未增加"
fi
```

**预期**:
- 日志包含"追加新因子 (1个)"
- 执行前1列，执行后2列
- 因子列包含 `Ref($close,60)/$close` 和 `Mean($volume,20)`

---

### 场景3: 智能替换已存在因子

**目的**: 验证重新计算已存在因子的功能

**命令与验证**:
```bash
# 1. 记录执行前的文件时间戳和列数
BEFORE_TS=$(stat -f "%m" .cache/csi300_20230101_20240101__factor_std.parquet)
BEFORE_COUNT=$(python -c "import pandas as pd; df = pd.read_parquet('.cache/csi300_20230101_20240101__factor_std.parquet'); print(df.shape[1])")
echo "执行前时间戳: $BEFORE_TS"
echo "执行前列数: $BEFORE_COUNT"

# 2. 重新计算相同因子（触发替换逻辑）
echo ""
echo "执行因子替换..."
conda activate quant && python step1/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Mean(\$volume,20)" 2>&1 | grep -E "(更新|替换|因子)"

# 3. 等待1秒后记录执行后的时间戳和列数
sleep 1
AFTER_TS=$(stat -f "%m" .cache/csi300_20230101_20240101__factor_std.parquet)
AFTER_COUNT=$(python -c "import pandas as pd; df = pd.read_parquet('.cache/csi300_20230101_20240101__factor_std.parquet'); print(df.shape[1])")
echo ""
echo "执行后时间戳: $AFTER_TS"
echo "执行后列数: $AFTER_COUNT"

# 4. 验证
echo ""
if [ "$AFTER_TS" -gt "$BEFORE_TS" ] && [ "$AFTER_COUNT" -eq "$BEFORE_COUNT" ]; then
  echo "✅ 通过：因子已更新（时间戳变化，列数保持为$AFTER_COUNT）"
else
  echo "❌ 失败：时间戳或列数不符合预期"
fi
```

**预期**:
- 日志包含"更新已有因子 (1个): Mean($volume,20)"
- 文件时间戳更新
- 因子列数保持不变（仍为2）

---

## Test 7: 错误检测验证

### 场景1: 不存在的日期范围（无cache数据）

**目的**: 验证当请求的日期范围没有对应cache时，系统是否报错

**命令与验证**:
```bash
echo "测试不存在的日期范围..."
echo ""

# 执行命令并捕获输出
OUTPUT=$(conda activate quant && python step2/cli.py \
  --market csi300 \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --factor-formulas "Ref(\$close,60)/\$close" 2>&1)

# 显示输出（前20行）
echo "$OUTPUT" | head -20
echo ""

# 验证：检查错误信息关键词
if echo "$OUTPUT" | grep -q "Cache 文件不存在"; then
  echo "✅ 通过：正确检测到cache不存在的错误"
else
  echo "❌ 失败：未检测到预期错误"
fi
```

**预期**: 命令失败（退出码非0），错误信息包含"Cache 文件不存在"

---

### 场景2: 不存在的因子

**目的**: 验证请求的因子在cache中不存在时的错误处理

**命令与验证**:
```bash
echo "测试不存在的因子..."
echo ""

# 执行命令并捕获输出
OUTPUT=$(conda activate quant && python step2/cli.py \
  --market csi300 \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --factor-formulas "Mean(\$volume,30)" 2>&1)

# 显示输出（前20行）
echo "$OUTPUT" | head -20
echo ""

# 验证：检查错误信息关键词
if echo "$OUTPUT" | grep -q -E "(不在cache中|不存在|未找到)"; then
  echo "✅ 通过：正确检测到因子不存在的错误"
else
  echo "❌ 失败：未检测到预期错误"
fi
```

**预期**: 命令失败，错误信息包含"不在cache中"或"不存在"

## 预期测试时间

- Test 1-5: 7-11分钟
- Test 6-7: 15-25分钟
- 总计: 22-36分钟
