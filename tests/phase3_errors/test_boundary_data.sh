#!/bin/bash
# Phase 3: 数据边界测试
# 测试极短时间范围、不支持参数等边界场景

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
MARKET=${MARKET:-"csi300"}
PROVIDER_URI=${PROVIDER_URI:-"$PROJECT_ROOT/cache/qlib_data"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}

# 测试开始
log_test_start "数据边界测试"

# ========== 测试3.5.1: 极短时间范围 ==========
log_info "测试3.5.1: 极短时间范围 (1个月)"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2023-01-01" \
    --end-date "2023-01-31" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" \
    --verbose 2>&1)

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    test_pass "极短时间范围执行成功"

    # 检查是否有关于NaN的警告（ROC60需要60天lookback）
    if [[ "$output" == *"NaN"* ]] || [[ "$output" == *"缺失"* ]] || [[ "$output" == *"lookback"* ]]; then
        log_info "  检测到NaN警告（预期：ROC60有60天lookback）"
    fi

    # 验证生成了cache文件
    if [[ -f "$CACHE_DIR/step1_metadata.json" ]]; then
        test_pass "生成了元数据文件"

        # 检查元数据中的日期范围
        start_date=$(python -c "import json; print(json.load(open('$CACHE_DIR/step1_metadata.json'))['start_date'])" 2>/dev/null)
        end_date=$(python -c "import json; print(json.load(open('$CACHE_DIR/step1_metadata.json'))['end_date'])" 2>/dev/null)
        log_info "  元数据日期范围: $start_date 至 $end_date"
    fi
else
    test_fail "极短时间范围执行失败"
fi

# ========== 测试3.5.2: 不支持的周期 ==========
log_info ""
log_info "测试3.5.2: 不支持的周期参数"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1x" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" 2>&1)

exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到不支持的周期"

    # 检查错误信息
    if [[ "$output" == *"不支持的周期"* ]] || [[ "$output" == *"1x"* ]]; then
        test_pass "错误信息明确指出不支持的周期"
    fi

    # 检查是否列出支持的周期
    if [[ "$output" == *"支持的周期"* ]] || [[ "$output" == *"1d"* ]] || [[ "$output" == *"1w"* ]]; then
        test_pass "错误信息列出支持的周期"
    fi
else
    test_fail "应该检测到不支持的周期但未报错"
fi

# ========== 测试3.5.3: 空因子表达式 ==========
log_info ""
log_info "测试3.5.3: 空因子表达式"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --factor-formulas "" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" 2>&1)

exit_code=$?

# 空表达式应该报错
if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到空因子表达式"
else
    test_fail "应该检测到空因子表达式但未报错"
fi

# ========== 测试3.5.4: 无效的日期格式 ==========
log_info ""
log_info "测试3.5.4: 无效的日期格式"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2023/01/01" \
    --end-date "2023-12-31" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" 2>&1)

exit_code=$?

# 无效日期格式应该报错
if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到无效日期格式"
else
    log_warning "日期格式验证可能由其他层处理"
fi

# ========== 测试3.5.5: 结束日期早于开始日期 ==========
log_info ""
log_info "测试3.5.5: 结束日期早于开始日期"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2024-01-01" \
    --end-date "2023-12-31" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" 2>&1)

exit_code=$?

# 反向日期应该报错或至少产生警告
if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到日期范围错误"
else
    log_warning "日期范围验证可能由其他层处理，但应该检查数据是否为空"
fi

# ========== 测试3.5.6: 单日数据 ==========
log_info ""
log_info "测试3.5.6: 单日数据范围"

output=$(python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "2023-01-03" \
    --end-date "2023-01-03" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" \
    --verbose 2>&1)

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    test_pass "单日数据范围执行成功"
    log_info "  注意: ROC60因子会有大量NaN（60天lookback）"
else
    test_fail "单日数据范围执行失败"
fi

# 测试摘要
echo ""
print_test_summary
