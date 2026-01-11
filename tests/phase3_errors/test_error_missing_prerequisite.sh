#!/bin/bash
# Phase 3: 缺失前置step错误测试
# 测试直接运行后续step而不运行前置step的错误处理

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
MARKET=${MARKET:-"csi300"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}

# 测试开始
log_test_start "缺失前置step错误测试"

# ========== 测试3.2.1: 缺少step1直接运行step2 ==========
log_info "测试3.2.1: 缺少step1直接运行step2"

# 清空cache元数据
log_warning "清理Step元数据..."
rm -f "$CACHE_DIR"/step*_metadata.json

if [[ ! -f "$CACHE_DIR/step1_metadata.json" ]]; then
    test_pass "Step1元数据已清理"
else
    test_fail "清理失败"
    exit 1
fi

# 尝试直接运行step2（应该报错）
log_info "尝试运行Step2 (没有Step1元数据)..."
output=$(python "$PROJECT_ROOT/step2/因子中性化.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR" 2>&1)

exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到缺少Step1元数据"

    # 检查错误信息
    if [[ "$output" == *"找不到"* ]] || [[ "$output" == *"step1"* ]] || [[ "$output" == *"metadata"* ]]; then
        test_pass "错误信息提示缺少Step1"
    fi

    # 检查是否提示先运行step1
    if [[ "$output" == *"先运行"* ]] || [[ "$output" == *"请先"* ]]; then
        test_pass "错误信息提示解决方法"
    fi
else
    test_fail "应该检测到缺少Step1但未报错"
fi

# ========== 测试3.2.2: 缺少step2直接运行step4 ==========
log_info ""
log_info "测试3.2.2: 缺少step2直接运行step4"

# 只生成step1，不生成step2
log_info "Step1: 生成cache..."
START_DATE="2023-01-01"
END_DATE="2024-12-31"
PROVIDER_URI="$PROJECT_ROOT/cache/qlib_data"

python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "Step1执行成功"
else
    test_fail "Step1执行失败"
    exit 1
fi

# 确保step2元数据不存在
rm -f "$CACHE_DIR/step2_metadata.json"

# 尝试直接运行step4（应该报错）
log_info "尝试运行Step4 (没有Step2元数据)..."
output=$(python "$PROJECT_ROOT/step4/因子绩效评估.py" \
    --market "$MARKET" \
    --cache-dir "$CACHE_DIR" \
    --provider-uri "$PROVIDER_URI" 2>&1)

exit_code=$?

if [[ $exit_code -ne 0 ]]; then
    test_pass "正确检测到缺少Step2元数据"

    # 检查错误信息
    if [[ "$output" == *"step2"* ]] || [[ "$output" == *"中性化"* ]]; then
        test_pass "错误信息提到Step2"
    fi
else
    test_fail "应该检测到缺少Step2但未报错"
fi

# 测试摘要
echo ""
print_test_summary
