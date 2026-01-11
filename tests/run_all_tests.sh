#!/bin/bash
# 量化因子分析流程 - 主测试入口脚本
# 整合所有测试，自动激活conda环境，生成测试报告

set -e

# ========== 默认参数 ==========
CONDA_ENV=${CONDA_ENV:-"quant"}
PHASE=""
STEP=""
NO_CLEANUP=0
VERBOSE=0

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 测试配置
MARKET=${MARKET:-"csi300"}
START_DATE=${START_DATE:-"2023-01-01"}
END_DATE=${END_DATE:-"2025-12-31"}
FACTOR_FORMULA=${FACTOR_FORMULA:-"Ref(\$close,60)/\$close"}
PERIODS=${PERIODS:-"1d,1w,1m"}
CACHE_DIR=${CACHE_DIR:-"$PROJECT_ROOT/cache"}
STOCK_DIR=${STOCK_DIR:-"$PROJECT_ROOT/stock"}
INDEX_DIR=${INDEX_DIR:-"$PROJECT_ROOT/index"}
FINANCE_DIR=${FINANCE_DIR:-"$PROJECT_ROOT/finance"}
QLIB_SRC_DIR=${QLIB_SRC_DIR:-"$PROJECT_ROOT/qlib_src"}

# 输出文件
REPORT_FILE="$SCRIPT_DIR/test_report.txt"
LOG_FILE="$SCRIPT_DIR/test_log.txt"

# ========== 解析CLI参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --no-cleanup)
            NO_CLEANUP=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --market)
            MARKET="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash run_all_tests.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --conda-env NAME     Conda环境名称 (default: quant)"
            echo "  --phase N            只运行指定phase (1, 2, 3)"
            echo "  --step NAME          只运行指定step (step0, step1, step2, step3, step4)"
            echo "  --no-cleanup         保留测试结果，不清理"
            echo "  --verbose, -v        显示详细输出"
            echo "  --market MARKET      股票池 (default: csi300)"
            echo "  --start-date DATE    起始日期 (default: 2023-01-01)"
            echo "  --end-date DATE      结束日期 (default: 2025-12-31)"
            echo "  --help, -h           显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  # 运行所有测试"
            echo "  bash tests/run_all_tests.sh"
            echo ""
            echo "  # 只运行Phase 1（基本通过性）"
            echo "  bash tests/run_all_tests.sh --phase 1"
            echo ""
            echo "  # 只运行Step1测试"
            echo "  bash tests/run_all_tests.sh --step step1"
            echo ""
            echo "  # 使用不同的conda环境"
            echo "  bash tests/run_all_tests.sh --conda-env myenv"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1 (使用 --help 查看帮助)"
            exit 1
            ;;
    esac
done

# ========== 加载测试工具函数 ==========
source "$PROJECT_ROOT/tests/lib/test_utils.sh"

# ========== Conda环境函数 ==========

check_conda_env() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda未安装或不在PATH中"
        log_info "请先安装Conda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    if ! conda info --envs 2>/dev/null | grep -q "^${CONDA_ENV} "; then
        log_error "Conda环境 '${CONDA_ENV}' 不存在"
        log_info "可用环境:"
        conda info --envs 2>/dev/null | grep -v "^#" | grep -v "^$"
        exit 1
    fi

    log_success "Conda环境 '${CONDA_ENV}' 存在"
}

activate_conda_env() {
    log_info "激活conda环境: ${CONDA_ENV}"
    eval "$(conda shell.bash hook 2>/dev/null)"

    conda activate ${CONDA_ENV} 2>/dev/null

    if [[ "${CONDA_DEFAULT_ENV}" == "${CONDA_ENV}" ]]; then
        log_success "当前环境: ${CONDA_DEFAULT_ENV}"
    else
        log_error "激活conda环境失败"
        exit 1
    fi
}

# ========== 测试执行函数 ==========

run_phase1_tests() {
    log_test_start "Phase 1: 基本运行测试"

    local phase1_dir="$PROJECT_ROOT/tests/phase1_basic"
    local tests=(
        "test_step0_basic.sh"
        "test_step1_basic.sh"
        "test_step2_basic.sh"
        "test_step3_basic.sh"
        "test_step4_basic.sh"
    )

    # 如果指定了特定step，只运行该step
    if [[ -n "$STEP" ]]; then
        tests=("test_${STEP}.sh")
    fi

    for test_script in "${tests[@]}"; do
        local test_path="$phase1_dir/$test_script"
        if [[ -f "$test_path" ]]; then
            log_info "运行: $test_script"
            export MARKET START_DATE END_DATE FACTER_FORMULA PERIODS \
                   CACHE_DIR STOCK_DIR INDEX_DIR FINANCE_DIR QLIB_SRC_DIR
            bash "$test_path" 2>&1 | tee -a "$LOG_FILE"
            echo ""
        else
            log_warning "测试脚本不存在: $test_path"
        fi
    done
}

run_phase2_tests() {
    log_test_start "Phase 2: Cache机制测试"

    local phase2_dir="$PROJECT_ROOT/tests/phase2_cache"
    local tests=(
        "test_cache_subset.sh"
        "test_cache_combination.sh"
        "test_cache_dependency.sh"
    )

    for test_script in "${tests[@]}"; do
        local test_path="$phase2_dir/$test_script"
        if [[ -f "$test_path" ]]; then
            log_info "运行: $test_script"
            export MARKET START_DATE END_DATE FACTER_FORMULA PERIODS \
                   CACHE_DIR RAW_DATA_DIR QLIB_SRC_DIR
            bash "$test_path" 2>&1 | tee -a "$LOG_FILE"
            echo ""
        else
            log_warning "测试脚本不存在: $test_path"
        fi
    done
}

run_phase3_tests() {
    log_test_start "Phase 3: 错误处理测试"

    local phase3_dir="$PROJECT_ROOT/tests/phase3_errors"
    local tests=(
        "test_error_cache_mismatch.sh"
        "test_error_missing_prerequisite.sh"
        "test_error_step1_required_params.sh"
        "test_dry_run.sh"
        "test_boundary_data.sh"
    )

    for test_script in "${tests[@]}"; do
        local test_path="$phase3_dir/$test_script"
        if [[ -f "$test_path" ]]; then
            log_info "运行: $test_script"
            export MARKET START_DATE END_DATE FACTER_FORMULA PERIODS \
                   CACHE_DIR RAW_DATA_DIR QLIB_SRC_DIR
            bash "$test_path" 2>&1 | tee -a "$LOG_FILE"
            echo ""
        else
            log_warning "测试脚本不存在: $test_path"
        fi
    done
}

run_all_phases() {
    # 根据PHASE参数决定运行哪些测试
    if [[ -z "$PHASE" ]]; then
        # 运行所有phase
        run_phase1_tests
        run_phase2_tests
        run_phase3_tests
    elif [[ "$PHASE" == "1" ]]; then
        run_phase1_tests
    elif [[ "$PHASE" == "2" ]]; then
        run_phase2_tests
    elif [[ "$PHASE" == "3" ]]; then
        run_phase3_tests
    else
        log_error "无效的phase编号: $PHASE (应为1, 2, 或3)"
        exit 1
    fi
}

generate_report() {
    log_info "生成测试报告..."

    {
        echo "======================================"
        echo "  量化因子分析流程 - 测试报告"
        echo "======================================"
        echo ""
        echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Conda环境: ${CONDA_ENV}"
        echo "市场: ${MARKET}"
        echo "时间范围: [${START_DATE}, ${END_DATE}]"
        echo "因子: ${FACTOR_FORMULA}"
        echo "周期: ${PERIODS}"
        echo ""
        echo "======================================"
        echo "测试结果"
        echo "======================================"
        echo "总测试数: $TESTS_TOTAL"
        echo "通过: $TESTS_PASSED"
        echo "失败: $TESTS_FAILED"
        echo ""

        if [[ $TESTS_FAILED -eq 0 ]]; then
            echo "✅ 所有测试通过!"
        else
            echo "❌ 有 $TESTS_FAILED 个测试失败"
        fi

        echo ""
        echo "详细日志请查看: $LOG_FILE"
        echo ""
    } > "$REPORT_FILE"

    log_success "测试报告已保存: $REPORT_FILE"

    # 显示报告摘要
    cat "$REPORT_FILE"
}

# ========== 主测试流程 ==========

main() {
    echo "========================================"
    echo "  📊 量化因子分析流程 - 测试套件"
    echo "========================================"
    echo ""

    # 清空之前的测试结果
    > "$LOG_FILE"
    reset_test_counters

    # 检查并激活conda环境
    log_info "检查Conda环境..."
    check_conda_env
    activate_conda_env
    echo ""

    # 检查Python环境
    log_info "检查Python环境..."
    python --version
    echo ""

    # 导出环境变量供子测试使用
    export MARKET
    export START_DATE
    export END_DATE
    export FACTER_FORMULA
    export PERIODS
    export CACHE_DIR
    export STOCK_DIR
    export INDEX_DIR
    export FINANCE_DIR
    export QLIB_SRC_DIR
    export PROVIDER_URI="$PROJECT_ROOT/cache/qlib_data"

    # 显示测试配置
    log_info "测试配置:"
    log_info "  市场: $MARKET"
    log_info "  时间范围: [$START_DATE, $END_DATE]"
    log_info "  因子: $FACTOR_FORMULA"
    log_info "  周期: $PERIODS"
    log_info "  Cache目录: $CACHE_DIR"
    if [[ $VERBOSE -eq 1 ]]; then
        log_info "  详细模式: 启用"
    fi
    echo ""

    # 执行测试
    run_all_phases

    # 生成报告
    generate_report

    # 清理选项
    if [[ $NO_CLEANUP -eq 0 ]]; then
        log_info "保留测试结果 (使用 --no-cleanup 选项保留)"
    else
        log_info "NO_CLEANUP已启用，保留所有测试文件"
    fi

    # 返回测试结果
    if [[ $TESTS_FAILED -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# 运行主流程
main "$@"
