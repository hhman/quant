#!/bin/bash
# Phase 2: Cache组合子集匹配测试
# 测试因子+周期+时间范围三重子集同时生效

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
log_test_start "Cache组合子集匹配测试"

# ========== 测试2.2.1: 因子+周期+时间三重子集 ==========
log_info "测试2.2.1: 因子+周期+时间三重子集匹配"

# 先生成包含多因子、多周期、长时间范围的cache
log_info "Step1: 生成完整cache (多因子、多周期、2023-2025)..."
python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "Ref(\$close,60)/\$close,Ref(\$close,20)/\$close,MA(\$close,10)" \
    --periods "1d,1w,1m" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "完整cache生成成功"

    # 检查原始cache
    if command -v python &> /dev/null; then
        log_info "原始Step1 cache内容:"
        python -c "
import pandas as pd
df = pd.read_parquet('$CACHE_DIR/factor_standardized.parquet')
print(f'  因子列: {list(df.columns)}')
dates = df.index.get_level_values('datetime').unique()
print(f'  日期范围: {dates.min()} 至 {dates.max()}')
" 2>/dev/null

        ret_df = pd.read_parquet('$CACHE_DIR/data_returns.parquet')
        print(f"  周期列: {list(ret_df.columns)}")
    fi

    # Step2使用三重子集：单因子+单周期+单年
    log_info ""
    log_info "Step2: 使用三重子集 (ROC60 + 1d周期 + 2024年)..."
    python "$PROJECT_ROOT/step2/因子中性化.py" \
        --market "$MARKET" \
        --factor-formulas "Ref(\$close,60)/\$close" \
        --periods "1d" \
        --start-date "2024-01-01" \
        --end-date "2024-12-31" \
        --cache-dir "$CACHE_DIR" \
        --verbose

    if [[ $? -eq 0 ]]; then
        test_pass "三重子集匹配成功"

        # 验证输出确实应用了三重过滤
        if command -v python &> /dev/null; then
            log_info ""
            log_info "验证过滤结果..."

            # 检查因子列
            factor_cols=$(python -c "
import pandas as pd
df = pd.read_parquet('$CACHE_DIR/factor_行业市值中性化.parquet')
print(','.join(df.columns.tolist()))
" 2>/dev/null)

            log_info "  输出因子列: $factor_cols"

            if [[ "$factor_cols" == *"Ref(\$close,60)/\$close"* ]]; then
                test_pass "  ✓ 只包含请求的因子"
            else
                test_fail "  ✗ 因子过滤不正确"
            fi

            # 检查日期范围
            date_range=$(python -c "
import pandas as pd
df = pd.read_parquet('$CACHE_DIR/factor_行业市值中性化.parquet')
dates = df.index.get_level_values('datetime').unique()
print(f'{dates.min()} 至 {dates.max()}')
" 2>/dev/null)

            log_info "  输出日期范围: $date_range"

            if [[ "$date_range" == "2024"* ]]; then
                test_pass "  ✓ 日期范围正确截取"
            else
                test_fail "  ✗ 日期截取不正确"
            fi

            # 检查行数（应该明显少于原始数据）
            original_count=$(python -c "
import pandas as pd
df = pd.read_parquet('$CACHE_DIR/factor_standardized.parquet')
print(len(df))
" 2>/dev/null)

            filtered_count=$(python -c "
import pandas as pd
df = pd.read_parquet('$CACHE_DIR/factor_行业市值中性化.parquet')
print(len(df))
" 2>/dev/null)

            log_info "  数据行数: 原始=$original_count, 过滤后=$filtered_count"

            if [[ $filtered_count -lt $original_count ]]; then
                test_pass "  ✓ 数据量正确减少"
            else
                test_fail "  ✗ 数据过滤可能未生效"
            fi
        fi
    else
        test_fail "三重子集匹配失败"
    fi
else
    test_fail "完整cache生成失败"
fi

# 测试摘要
echo ""
print_test_summary
