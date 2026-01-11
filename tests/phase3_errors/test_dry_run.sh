#!/bin/bash
# Phase 3: Dry-run模式测试
# 测试各step的--dry-run参数

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
MARKET=${MARKET:-"csi300"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}
PROVIDER_URI=${PROVIDER_URI:-"$PROJECT_ROOT/cache/qlib_data"}

# 记录测试前的cache状态
log_info "记录测试前cache状态..."
before_cache_files=$(ls -1 "$CACHE_DIR" 2>/dev/null | wc -l)

# 测试开始
log_test_start "Dry-run模式测试"

# ========== 测试3.4.1: Step1的dry-run模式 ==========
log_info "测试3.4.1: Step1 dry-run模式"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2023-01-01" \
    --end-date "2024-12-31" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI" \
    --dry-run \
    --verbose 2>&1)

exit_code=$?

# dry-run不应该报错
if [[ $exit_code -eq 0 ]]; then
    test_pass "Step1 dry-run执行成功"

    # 检查输出是否显示参数信息
    if [[ "$output" == *"模拟运行"* ]] || [[ "$output" == *"dry-run"* ]]; then
        test_pass "输出显示模拟运行信息"
    fi

    if [[ "$output" == *"$MARKET"* ]]; then
        test_pass "输出显示市场参数"
    fi

    if [[ "$output" == *"Ref(\$close,60)/\$close"* ]]; then
        test_pass "输出显示因子表达式"
    fi

    # 验证没有生成新文件
    after_cache_files=$(ls -1 "$CACHE_DIR" 2>/dev/null | wc -l)
    if [[ $after_cache_files -eq $before_cache_files ]]; then
        test_pass "没有生成新文件"
    else
        test_fail "dry-run模式不应该生成文件"
    fi
else
    test_fail "Step1 dry-run执行失败"
fi

# ========== 测试3.4.2: Step2的dry-run模式 ==========
log_info ""
log_info "测试3.4.2: Step2 dry-run模式"

output=$(python "$PROJECT_ROOT/step2/因子中性化.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR" \
    --dry-run \
    --verbose 2>&1)

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    test_pass "Step2 dry-run执行成功"

    if [[ "$output" == *"模拟运行"* ]] || [[ "$output" == *"dry-run"* ]]; then
        test_pass "输出显示模拟运行信息"
    fi
else
    test_fail "Step2 dry-run执行失败"
fi

# ========== 测试3.4.3: Step3的dry-run模式 ==========
log_info ""
log_info "测试3.4.3: Step3 dry-run模式"

output=$(python "$PROJECT_ROOT/step3/因子收益率计算.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR" \
    --dry-run \
    --verbose 2>&1)

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    test_pass "Step3 dry-run执行成功"

    if [[ "$output" == *"模拟运行"* ]] || [[ "$output" == *"dry-run"* ]]; then
        test_pass "输出显示模拟运行信息"
    fi
else
    test_fail "Step3 dry-run执行失败"
fi

# ========== 测试3.4.4: Step4的dry-run模式 ==========
log_info ""
log_info "测试3.4.4: Step4 dry-run模式"

output=$(python "$PROJECT_ROOT/step4/因子绩效评估.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI" \
    --dry-run \
    --verbose 2>&1)

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    test_pass "Step4 dry-run执行成功"

    if [[ "$output" == *"模拟运行"* ]] || [[ "$output" == *"dry-run"* ]]; then
        test_pass "输出显示模拟运行信息"
    fi
else
    test_fail "Step4 dry-run执行失败"
fi

# ========== 测试3.4.5: 批量测试所有step ==========
log_info ""
log_info "测试3.4.5: 批量dry-run所有step"

all_passed=true
for step in \
    "step1/因子提取与预处理.py" \
    "step2/因子中性化.py" \
    "step3/因子收益率计算.py" \
    "step4/因子绩效评估.py"
do
    log_info "  测试: $step"
    python "$PROJECT_ROOT/$step" \
        --market "$MARKET" \
        --cache-dir "$CACHE_DIR" \
        --dry-run >/dev/null 2>&1

    if [[ $? -eq 0 ]]; then
        log_success "    $step dry-run成功"
    else
        log_error "    $step dry-run失败"
        all_passed=false
    fi
done

if $all_passed; then
    test_pass "所有step dry-run测试通过"
else
    test_fail "部分step dry-run测试失败"
fi

# 测试摘要
echo ""
print_test_summary
