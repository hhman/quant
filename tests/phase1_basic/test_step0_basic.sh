#!/bin/bash
# Phase 1: Step0基本运行测试
# 测试Step0数据预处理流程

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# 配置参数
START_DATE=${START_DATE:-"2023-01-01"}
END_DATE=${END_DATE:-"2025-12-31"}
STOCK_DIR=${STOCK_DIR:-"$PROJECT_ROOT/stock"}
INDEX_DIR=${INDEX_DIR:-"$PROJECT_ROOT/index"}
FINANCE_DIR=${FINANCE_DIR:-"$PROJECT_ROOT/finance"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}
QLIB_SRC_DIR=${QLIB_SRC_DIR:-"$PROJECT_ROOT/qlib_src"}

# 测试开始
log_test_start "Step0基本运行测试"

# 检查原始数据目录
log_info "检查原始数据目录..."

# 检查必要目录
check_dir_exists "$STOCK_DIR" "股票CSV目录" || exit 1
check_dir_exists "$INDEX_DIR" "指数CSV目录" || exit 1
check_dir_exists "$FINANCE_DIR" "财务CSV目录" || exit 1

# 检查qlib_src目录
if [[ ! -d "$QLIB_SRC_DIR" ]]; then
    log_error "Qlib脚本目录不存在: $QLIB_SRC_DIR"
    log_info "提示: 需要qlib源码目录中的dump_bin.py等脚本"
    exit 1
fi

# 执行Step0
log_info "执行Step0数据预处理..."
log_info "时间范围: [$START_DATE, $END_DATE]"
log_info "股票数据: $STOCK_DIR"
log_info "指数数据: $INDEX_DIR"
log_info "财务数据: $FINANCE_DIR"
log_info "Cache目录: $CACHE_DIR"

bash "$PROJECT_ROOT/step0/step0.sh" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --stock-dir "$STOCK_DIR" \
    --index-dir "$INDEX_DIR" \
    --finance-dir "$FINANCE_DIR" \
    --cache-dir "$CACHE_DIR" \
    --qlib-src-dir "$QLIB_SRC_DIR" \
    --verbose

# 检查执行结果
if [[ $? -eq 0 ]]; then
    test_pass "Step0执行成功"
else
    test_fail "Step0执行失败"
    exit 1
fi

# 验证输出文件
log_info "验证Step0输出..."
verify_step0_output "$CACHE_DIR"

# 验证元数据
log_info "验证Step0元数据..."
check_metadata_field "$CACHE_DIR/step0_metadata.json" "stage" "step0"
check_metadata_field "$CACHE_DIR/step0_metadata.json" "start_date" "$START_DATE"
check_metadata_field "$CACHE_DIR/step0_metadata.json" "end_date" "$END_DATE"

# 检查Qlib数据目录结构
log_info "检查Qlib数据目录结构..."
if [[ -d "$CACHE_DIR/qlib_data/features" ]]; then
    file_count=$(find "$CACHE_DIR/qlib_data/features" -name "*.csv" | wc -l)
    log_success "Qlib features目录包含 $file_count 个CSV文件"
    test_pass "Qlib features目录存在"
else
    test_fail "Qlib features目录不存在"
fi

# 检查instruments文件
if [[ -d "$CACHE_DIR/qlib_data/instruments" ]]; then
    log_success "Qlib instruments目录包含以下文件:"
    ls -1 "$CACHE_DIR/qlib_data/instruments" | head -5
    if [[ $(ls -1 "$CACHE_DIR/qlib_data/instruments" | wc -l) -gt 5 ]]; then
        log_info "... (共$(ls -1 "$CACHE_DIR/qlib_data/instruments" | wc -l)个文件)"
    fi
fi

# 测试摘要
echo ""
print_test_summary
