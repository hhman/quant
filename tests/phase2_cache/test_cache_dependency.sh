#!/bin/bash
# Phase 2: Cache依赖链测试
# 测试Step之间的依赖关系

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
MARKET=${MARKET:-"csi300"}
START_DATE=${START_DATE:-"2023-01-01"}
END_DATE=${END_DATE:-"2025-12-31"}
PROVIDER_URI=${PROVIDER_URI:-"$PROJECT_ROOT/cache/qlib_data"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}

# 测试开始
log_test_start "Cache依赖链测试"

# ========== 测试2.3.1: 完整流程依赖链 ==========
log_info "测试2.3.1: 完整流程依赖链 (Step0→Step1→Step2→Step3→Step4)"

# 清理所有cache，从头开始
log_warning "清理所有cache文件..."
clean_all_cache "$CACHE_DIR"

# Step0
log_info ""
log_info "执行Step0..."
if [[ -d "$PROJECT_ROOT/raw_data" ]]; then
    bash "$PROJECT_ROOT/step0/step0.sh" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --raw-data-dir "$PROJECT_ROOT/raw_data" \
        --cache-dir "$CACHE_DIR" \
        --qlib-src-dir "$PROJECT_ROOT/qlib_src"

    if [[ $? -eq 0 ]]; then
        test_pass "Step0执行成功"
        verify_step0_output "$CACHE_DIR"
    else
        test_fail "Step0执行失败"
        log_warning "Step0失败，跳过后续测试"
        exit 1
    fi
else
    log_warning "原始数据目录不存在，跳过Step0"
    log_info "假设Qlib数据已准备好..."
fi

# Step1
log_info ""
log_info "执行Step1..."
python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d,1w,1m" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "Step1执行成功"
    verify_step1_cache "$CACHE_DIR"
else
    test_fail "Step1执行失败"
    exit 1
fi

# Step2（依赖Step1）
log_info ""
log_info "执行Step2 (依赖Step1)..."
python "$PROJECT_ROOT/step2/因子中性化.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "Step2执行成功"
    verify_step2_cache "$CACHE_DIR"
else
    test_fail "Step2执行失败"
    exit 1
fi

# Step3（依赖Step1）
log_info ""
log_info "执行Step3 (依赖Step1)..."
python "$PROJECT_ROOT/step3/因子收益率计算.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "Step3执行成功"
    verify_step3_cache "$CACHE_DIR"
else
    test_fail "Step3执行失败"
    exit 1
fi

# Step4（依赖Step1和Step2）
log_info ""
log_info "执行Step4 (依赖Step1和Step2)..."
python "$PROJECT_ROOT/step4/因子绩效评估.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI"

if [[ $? -eq 0 ]]; then
    test_pass "Step4执行成功"
    verify_step4_cache "$CACHE_DIR"
else
    test_fail "Step4执行失败"
    exit 1
fi

# 验证元数据文件都被正确保存
log_info ""
log_info "验证元数据文件..."
for step in step0 step1 step2 step3 step4; do
    if [[ -f "$CACHE_DIR/${step}_metadata.json" ]]; then
        check_metadata_field "$CACHE_DIR/${step}_metadata.json" "stage" "$step"
    fi
done

# 显示依赖关系图
log_info ""
log_info "依赖关系总结:"
log_info "  Step0 → Step1 (Qlib数据)"
log_info "  Step1 → Step2 (标准化因子)"
log_info "  Step1 → Step3 (因子+收益率+风格)"
log_info "  Step1 → Step4 (收益率数据)"
log_info "  Step2 → Step4 (中性化因子)"

# 验证所有cache文件完整性
log_info ""
log_info "Cache文件完整性检查:"
all_cache_files=(
    "factor_raw.parquet"
    "factor_standardized.parquet"
    "factor_行业市值中性化.parquet"
    "factor_回归收益率.parquet"
    "factor_回归t值.parquet"
    "factor_ic.parquet"
    "factor_rank_ic.parquet"
    "factor_group_return.parquet"
    "factor_autocorr.parquet"
    "factor_turnover.parquet"
    "data_returns.parquet"
    "data_styles.parquet"
)

missing_files=0
for file in "${all_cache_files[@]}"; do
    if [[ -f "$CACHE_DIR/$file" ]]; then
        test_pass "✓ $file"
    else
        test_fail "✗ $file 缺失"
        ((missing_files++))
    fi
done

if [[ $missing_files -eq 0 ]]; then
    log_success "所有cache文件完整"
else
    log_warning "有 $missing_files 个cache文件缺失"
fi

# 测试摘要
echo ""
print_test_summary
