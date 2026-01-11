#!/bin/bash
# Phase 1: Step4基本运行测试
# 测试因子绩效评估流程

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
PROVIDER_URI=${PROVIDER_URI:-"$PROJECT_ROOT/cache/qlib_data"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}

# 测试开始
log_test_start "Step4基本运行测试"

# 检查Step2是否已执行
log_info "检查Step2输出..."
if [[ ! -f "$CACHE_DIR/step2_metadata.json" ]]; then
    log_error "Step2元数据不存在，请先运行Step2"
    exit 1
fi

# 验证Step2的cache文件
if ! verify_step2_cache "$CACHE_DIR"; then
    log_error "Step2 cache文件不完整"
    exit 1
fi

test_pass "Step2输出检查通过"

# 执行Step4
log_info "执行Step4因子绩效评估..."
log_info "市场: $MARKET"
log_info "时间范围: [$START_DATE, $END_DATE]"
log_info "因子表达式: $FACTOR_FORMULA"
log_info "周期: $PERIODS"

python "$PROJECT_ROOT/step4/因子绩效评估.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "$FACTOR_FORMULA" \
    --periods "$PERIODS" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI" \
    --verbose

# 检查执行结果
if [[ $? -eq 0 ]]; then
    test_pass "Step4执行成功"
else
    test_fail "Step4执行失败"
    exit 1
fi

# 验证输出文件
log_info "验证Step4输出..."
verify_step4_cache "$CACHE_DIR"

# 验证元数据
log_info "验证Step4元数据..."
check_metadata_field "$CACHE_DIR/step4_metadata.json" "stage" "step4"
check_metadata_field "$CACHE_DIR/step4_metadata.json" "market" "$MARKET"

# 检查绩效评估结果文件
log_info "检查绩效评估结果..."

if command -v python &> /dev/null; then
    # 检查parquet文件
    for file in factor_ic.parquet factor_rank_ic.parquet factor_group_return.parquet factor_autocorr.parquet factor_turnover.parquet; do
        if [[ -f "$CACHE_DIR/$file" ]]; then
            shape=$(python -c "import pandas as pd; df = pd.read_parquet('$CACHE_DIR/$file'); print(f'{df.shape[0]}行 x {df.shape[1]}列')" 2>/dev/null)
            log_success "$file: $shape"
            test_pass "$file可读取"
        fi
    done

    # 检查可视化图表目录
    if [[ -d "$CACHE_DIR/graphs" ]]; then
        graph_count=$(find "$CACHE_DIR/graphs" -name "*.html" | wc -l)
        log_success "可视化图表: $graph_count 个HTML文件"
        test_pass "可视化图表已生成"
    fi
else
    log_warning "Python不可用，跳过数据内容检查"
fi

# 测试摘要
echo ""
print_test_summary
