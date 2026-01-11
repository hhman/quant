#!/bin/bash
# Phase 2: Cache子集匹配测试
# 测试因子、周期、时间范围的独立子集匹配

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
log_test_start "Cache基础子集匹配测试"

# ========== 测试2.1.1: 因子子集匹配 ==========
log_info "测试2.1.1: 因子子集匹配"

# 先生成包含多个因子的cache
log_info "Step1: 生成包含多个因子的cache..."
python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "Ref(\$close,60)/\$close,Ref(\$close,20)/\$close" \
    --periods "1d,1w,1m" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR" \
    --verbose

if [[ $? -eq 0 ]]; then
    test_pass "多因子Step1执行成功"

    # Step2只用其中一个因子
    log_info "Step2: 只使用ROC60因子..."
    python "$PROJECT_ROOT/step2/因子中性化.py" \
        --market "$MARKET" \
        --factor-formulas "Ref(\$close,60)/\$close" \
        --cache-dir "$CACHE_DIR" \
        --verbose

    if [[ $? -eq 0 ]]; then
        test_pass "因子子集匹配成功"

        # 验证输出只包含请求的因子
        if command -v python &> /dev/null; then
            factor_cols=$(python -c "
import pandas as pd
df = pd.read_parquet('$CACHE_DIR/factor_行业市值中性化.parquet')
print(','.join(df.columns.tolist()))
" 2>/dev/null)

            log_info "输出因子列: $factor_cols"
            if [[ "$factor_cols" == *"Ref(\$close,60)/\$close"* ]]; then
                test_pass "输出包含请求的因子"
            fi
            if [[ "$factor_cols" != *"Ref(\$close,20)/\$close"* ]]; then
                test_pass "输出不包含未请求的因子"
            fi
        fi
    else
        test_fail "因子子集匹配失败"
    fi
else
    test_fail "多因子Step1执行失败"
fi

# ========== 测试2.1.2: 周期子集匹配 ==========
log_info ""
log_info "测试2.1.2: 周期子集匹配"

# 重新生成cache（确保有多个周期）
log_info "Step1: 生成包含多周期的cache..."
python "$PROJECT_ROOT/step1/因子提取与预处理.py" \
    --market "$MARKET" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --factor-formulas "Ref(\$close,60)/\$close" \
    --periods "1d,1w,1m" \
    --provider-uri "$PROVIDER_URI" \
    --cache-dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    test_pass "多周期Step1执行成功"

    # Step2只用1d周期
    log_info "Step2: 只使用1d周期..."
    python "$PROJECT_ROOT/step2/因子中性化.py" \
        --market "$MARKET" \
        --periods "1d" \
        --cache-dir "$CACHE_DIR" \
        --verbose

    if [[ $? -eq 0 ]]; then
        test_pass "周期子集匹配成功"
    else
        test_fail "周期子集匹配失败"
    fi
else
    test_fail "多周期Step1执行失败"
fi

# ========== 测试2.1.3: 时间范围子集匹配 ==========
log_info ""
log_info "测试2.1.3: 时间范围子集匹配"

# 重新生成cache
log_info "Step1: 生成2023-2025的cache..."
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

    # Step2只用2024年数据
    log_info "Step2: 只使用2024年数据..."
    python "$PROJECT_ROOT/step2/因子中性化.py" \
        --market "$MARKET" \
        --start-date "2024-01-01" \
        --end-date "2024-12-31" \
        --cache-dir "$CACHE_DIR" \
        --verbose

    if [[ $? -eq 0 ]]; then
        test_pass "时间范围子集匹配成功"

        # 验证数据确实被截取
        if command -v python &> /dev/null; then
            date_range=$(python -c "
import pandas as pd
df = pd.read_parquet('$CACHE_DIR/factor_行业市值中性化.parquet')
dates = df.index.get_level_values('datetime').unique()
print(f'{dates.min()} 至 {dates.max()}')
" 2>/dev/null)

            log_info "输出数据日期范围: $date_range"
            if [[ "$date_range" == *"2024"* ]]; then
                test_pass "日期范围正确截取"
            fi
        fi
    else
        test_fail "时间范围子集匹配失败"
    fi
else
    test_fail "Step1执行失败"
fi

# 测试摘要
echo ""
print_test_summary
