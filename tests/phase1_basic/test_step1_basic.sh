#!/bin/bash
# Phase 1: Step1基本运行测试
# 测试因子提取与预处理流程

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
log_test_start "Step1基本运行测试"

# 检查Step0是否已执行
log_info "检查Step0输出..."
if [[ ! -f "$CACHE_DIR/step0_metadata.json" ]]; then
    log_error "Step0元数据不存在，请先运行Step0"
    exit 1
fi

if [[ ! -d "$PROVIDER_URI" ]]; then
    log_error "Qlib数据目录不存在: $PROVIDER_URI"
    log_info "请先运行Step0生成Qlib数据"
    exit 1
fi

test_pass "Step0输出检查通过"

# 执行Step1
log_info "执行Step1因子提取与预处理..."
log_info "市场: $MARKET"
log_info "时间范围: [$START_DATE, $END_DATE]"
log_info "因子表达式: $FACTOR_FORMULA"
log_info "周期: $PERIODS"

python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "$FACTOR_FORMULA" \
    --periods "$PERIODS" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" \
    --verbose

# 检查执行结果
if [[ $? -eq 0 ]]; then
    test_pass "Step1执行成功"
else
    test_fail "Step1执行失败"
    exit 1
fi

# 验证输出文件
log_info "验证Step1输出..."
verify_step1_cache "$CACHE_DIR"

# 验证元数据
log_info "验证Step1元数据..."
check_metadata_field "$CACHE_DIR/step1_metadata.json" "stage" "step1"
check_metadata_field "$CACHE_DIR/step1_metadata.json" "market" "$MARKET"
check_metadata_field "$CACHE_DIR/step1_metadata.json" "start_date" "$START_DATE"
check_metadata_field "$CACHE_DIR/step1_metadata.json" "end_date" "$END_DATE"

# 检查parquet文件的基本信息
log_info "检查parquet文件..."

if command -v python &> /dev/null; then
    for file in factor_raw.parquet factor_standardized.parquet data_returns.parquet data_styles.parquet; do
        if [[ -f "$CACHE_DIR/$file" ]]; then
            shape=$(python -c "import pandas as pd; df = pd.read_parquet('$CACHE_DIR/$file'); print(f'{df.shape[0]}行 x {df.shape[1]}列')" 2>/dev/null)
            log_success "$file: $shape"
            test_pass "$file可读取"
        fi
    done
else
    log_warning "Python不可用，跳过parquet文件内容检查"
fi

# 测试摘要
echo ""
print_test_summary
