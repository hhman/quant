#!/bin/bash
# Phase 1: Step2基本运行测试
# 测试因子中性化流程

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
MARKET=${MARKET:-"csi300"}
START_DATE=${START_DATE:-"2023-01-01"}
END_DATE=${END_DATE:-"2025-12-31"}
# 使用ROC60因子（60日反转因子）
FACTOR_FORMULA=${FACTOR_FORMULA:-"Ref(\$close,60)/\$close"}
PERIODS=${PERIODS:-"1d,1w,1m"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}

# 测试开始
log_test_start "Step2基本运行测试"

# 检查Step1是否已执行
log_info "检查Step1输出..."
if [[ ! -f "$CACHE_DIR/step1_metadata.json" ]]; then
    log_error "Step1元数据不存在，请先运行Step1"
    exit 1
fi

# 验证Step1的cache文件
if ! verify_step1_cache "$CACHE_DIR"; then
    log_error "Step1 cache文件不完整"
    exit 1
fi

test_pass "Step1输出检查通过"

# 执行Step2
log_info "执行Step2因子中性化..."
log_info "市场: $MARKET"
log_info "时间范围: [$START_DATE, $END_DATE]"
log_info "因子表达式: $FACTOR_FORMULA"
log_info "周期: $PERIODS"

python "$PROJECT_ROOT/step2/因子中性化.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "$FACTOR_FORMULA" \
    --periods "$PERIODS" \
    --cache-dir "$CACHE_DIR" \
    --verbose

# 检查执行结果
if [[ $? -eq 0 ]]; then
    test_pass "Step2执行成功"
else
    test_fail "Step2执行失败"
    exit 1
fi

# 验证输出文件
log_info "验证Step2输出..."
verify_step2_cache "$CACHE_DIR"

# 验证元数据
log_info "验证Step2元数据..."
check_metadata_field "$CACHE_DIR/step2_metadata.json" "stage" "step2"
check_metadata_field "$CACHE_DIR/step2_metadata.json" "market" "$MARKET"

# 检查中性化因子文件
log_info "检查中性化因子数据..."

if command -v python &> /dev/null; then
    file="factor_行业市值中性化.parquet"
    if [[ -f "$CACHE_DIR/$file" ]]; then
        shape=$(python -c "import pandas as pd; df = pd.read_parquet('$CACHE_DIR/$file'); print(f'{df.shape[0]}行 x {df.shape[1]}列')" 2>/dev/null)
        log_success "$file: $shape"
        test_pass "$file可读取"

        # 检查数据统计
        stats=$(python -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('$CACHE_DIR/$file')
print(f'均值: {df.mean().mean():.4f}')
print(f'标准差: {df.std().mean():.4f}')
print(f'缺失率: {df.isna().mean().mean():.2%}')
print(f'最小值: {df.min().min():.4f}')
print(f'最大值: {df.max().max():.4f}')
" 2>/dev/null)
        log_info "数据统计:\n$stats"
    fi
else
    log_warning "Python不可用，跳过数据内容检查"
fi

# 测试摘要
echo ""
print_test_summary
