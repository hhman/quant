#!/bin/bash
# Phase 1: Step3基本运行测试
# 测试因子收益率计算流程

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
log_test_start "Step3基本运行测试"

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

# 执行Step3
log_info "执行Step3因子收益率计算..."
log_info "市场: $MARKET"
log_info "时间范围: [$START_DATE, $END_DATE]"
log_info "因子表达式: $FACTOR_FORMULA"
log_info "周期: $PERIODS"

python "$PROJECT_ROOT/step3/因子收益率计算.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "$FACTOR_FORMULA" \
    --periods "$PERIODS" \
    --cache-dir "$CACHE_DIR" \
    --verbose

# 检查执行结果
if [[ $? -eq 0 ]]; then
    test_pass "Step3执行成功"
else
    test_fail "Step3执行失败"
    exit 1
fi

# 验证输出文件
log_info "验证Step3输出..."
verify_step3_cache "$CACHE_DIR"

# 验证元数据
log_info "验证Step3元数据..."
check_metadata_field "$CACHE_DIR/step3_metadata.json" "stage" "step3"
check_metadata_field "$CACHE_DIR/step3_metadata.json" "market" "$MARKET"

# 检查回归结果文件
log_info "检查回归结果数据..."

if command -v python &> /dev/null; then
    for file in factor_回归收益率.parquet factor_回归t值.parquet; do
        if [[ -f "$CACHE_DIR/$file" ]]; then
            shape=$(python -c "import pandas as pd; df = pd.read_parquet('$CACHE_DIR/$file'); print(f'{df.shape[0]}行 x {df.shape[1]}列')" 2>/dev/null)
            log_success "$file: $shape"
            test_pass "$file可读取"
        fi
    done

    # 检查汇总Excel文件
    for file in factor_回归收益率_summary.xlsx factor_回归t值_summary.xlsx; do
        if [[ -f "$CACHE_DIR/$file" ]]; then
            log_success "$file 存在"
            test_pass "$file可读取"
        fi
    done
else
    log_warning "Python不可用，跳过数据内容检查"
fi

# 测试摘要
echo ""
print_test_summary
