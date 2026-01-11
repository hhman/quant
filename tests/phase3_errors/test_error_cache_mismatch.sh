#!/bin/bash
# Phase 3: Cache不匹配错误测试
# 测试市场、日期、因子不匹配的错误处理

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
MARKET=${MARKET:-"csi300"}
START_DATE=${START_DATE:-"2023-01-01"}
END_DATE=${END_DATE:-"2024-12-31"}
PROVIDER_URI=${PROVIDER_URI:-"$PROJECT_ROOT/cache/qlib_data"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}

# 测试开始
log_test_start "Cache不匹配错误测试"

# ========== 测试3.1.1: 市场不匹配 ==========
log_info "测试3.1.1: 市场不匹配错误处理"

# 先用csi300生成cache
log_info "Step1: 使用csi300生成cache..."
python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "csi300" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" \
    --verbose

if [[ $? -eq 0 ]]; then
    test_pass "Step1 (csi300) 执行成功"

    # Step2尝试用sse50（应该报错）
    log_info "Step2: 尝试使用sse50 (应该报错)..."
    output=$(python "$PROJECT_ROOT/step2/因子中性化.py" \
        --market "sse50" \
        --cache-dir "$CACHE_DIR" 2>&1)

    exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        test_pass "正确检测到市场不匹配错误"

        # 检查错误信息是否清晰
        if [[ "$output" == *"市场不匹配"* ]] || [[ "$output" == *"market"* ]]; then
            test_pass "错误信息清晰"
        else
            log_warning "错误信息可能不够清晰"
        fi
    else
        test_fail "应该检测到市场不匹配但未报错"
    fi
else
    test_fail "Step1执行失败"
fi

# ========== 测试3.1.2: 日期范围超出cache ==========
log_info ""
log_info "测试3.1.2: 日期范围超出cache错误处理"

# 用2023-2024生成cache
log_info "Step1: 使用2023-2024生成cache..."
python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2023-01-01" \
    --end-date "2024-12-31" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "Step1 (2023-2024) 执行成功"

    # Step2请求2025年数据（应该报错）
    log_info "Step2: 请求2025年数据 (应该报错)..."
    output=$(python "$PROJECT_ROOT/step2/因子中性化.py" \
        --market "$MARKET" \
        --start-date "2025-01-01" \
        --end-date "2025-12-31" \
        --cache-dir "$CACHE_DIR" 2>&1)

    exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        test_pass "正确检测到日期范围超出错误"

        # 检查错误信息
        if [[ "$output" == *"日期范围"* ]] || [[ "$output" == *"超出cache"* ]]; then
            test_pass "错误信息包含日期范围说明"
        fi
    else
        test_fail "应该检测到日期范围超出但未报错"
    fi
else
    test_fail "Step1执行失败"
fi

# ========== 测试3.1.3: 因子不在cache中 ==========
log_info ""
log_info "测试3.1.3: 因子不在cache中错误处理"

# 用ROC60生成cache
log_info "Step1: 使用ROC60生成cache..."
python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "Step1 (ROC60) 执行成功"

    # Step2请求MA20因子（应该报错）
    log_info "Step2: 请求MA20因子 (应该报错)..."
    output=$(python "$PROJECT_ROOT/step2/因子中性化.py" \
        --market "$MARKET" \
        --factor-formulas "MA(\$close,20)" \
        --cache-dir "$CACHE_DIR" 2>&1)

    exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        test_pass "正确检测到因子不存在错误"

        # 检查错误信息
        if [[ "$output" == *"缺少因子"* ]] || [[ "$output" == *"cache缺少"* ]]; then
            test_pass "错误信息包含因子缺失说明"
        fi
    else
        test_fail "应该检测到因子不存在但未报错"
    fi
else
    test_fail "Step1执行失败"
fi

# 测试摘要
echo ""
print_test_summary
