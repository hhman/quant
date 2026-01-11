#!/bin/bash
# Phase 3: Step1必需参数错误测试
# 测试Step1缺少必需参数时的错误处理

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
MARKET=${MARKET:-"csi300"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}
PROVIDER_URI=${PROVIDER_URI:-"$PROJECT_ROOT/cache/qlib_data"}

# 测试开始
log_test_start "Step1必需参数错误测试"

# ========== 测试3.3.1: 缺少factor-formulas ==========
log_info "测试3.3.1: 缺少factor-formulas参数"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --periods "1d" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI" 2>&1)

exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到缺少factor-formulas参数"

    # 检查错误信息
    if [[ "$output" == *"factor-formulas"* ]] || [[ "$output" == *"因子表达式"* ]]; then
        test_pass "错误信息明确指出缺少factor-formulas"
    fi

    if [[ "$output" == *"必须使用"* ]] || [[ "$output" == *"必须指定"* ]]; then
        test_pass "错误信息说明是必需参数"
    fi
else
    test_fail "应该检测到缺少factor-formulas但未报错"
fi

# ========== 测试3.3.2: 缺少periods ==========
log_info ""
log_info "测试3.3.2: 缺少periods参数"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI" 2>&1)

exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到缺少periods参数"

    # 检查错误信息
    if [[ "$output" == *"periods"* ]] || [[ "$output" == *"周期"* ]]; then
        test_pass "错误信息明确指出缺少periods"
    fi

    if [[ "$output" == *"必须使用"* ]] || [[ "$output" == *"必须指定"* ]]; then
        test_pass "错误信息说明是必需参数"
    fi
else
    test_fail "应该检测到缺少periods但未报错"
fi

# ========== 测试3.3.3: 同时缺少两个必需参数 ==========
log_info ""
log_info "测试3.3.3: 同时缺少factor-formulas和periods"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI" 2>&1)

exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到缺少必需参数"

    # 检查错误信息是否提到两个参数
    if [[ "$output" == *"factor-formulas"* ]] && [[ "$output" == *"periods"* ]]; then
        test_pass "错误信息同时提到两个缺少的参数"
    elif [[ "$output" == *"factor-formulas"* ]] || [[ "$output" == *"periods"* ]]; then
        test_pass "错误信息提到至少一个缺少的参数"
    fi
else
    test_fail "应该检测到缺少必需参数但未报错"
fi

# ========== 测试3.3.4: 验证正确参数可以成功 ==========
log_info ""
log_info "测试3.3.4: 验证提供正确参数后可以成功运行"

# 先确保有Step0的输出
if [[ -f "$CACHE_DIR/step0_metadata.json" ]]; then
    python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
        --market "$MARKET" \
        --start-date "2023-01-01" \
        --end-date "2024-12-31" \
        --factor-formulas "Ref(\$close,60)/\$close" \
        --periods "1d" \
        --cache-dir "$CACHE_DIR" \
        --provider-uri "$PROVIDER_URI"

    if [[ $? -eq 0 ]]; then
        test_pass "提供正确参数后Step1成功执行"
    else
        test_fail "Step1执行失败"
    fi
else
    log_warning "Step0未执行，跳过验证测试"
fi

# 测试摘要
echo ""
print_test_summary
