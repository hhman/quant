#!/bin/bash
# 测试工具函数库
# 提供通用的测试辅助函数

# 颜色定义
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# 测试结果统计
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# ========== 日志函数 ==========

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_test_start() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}测试:${NC} $*"
    echo -e "${BLUE}========================================${NC}"
}

# ========== 测试结果记录 ==========

test_pass() {
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
    log_success "$@"
}

test_fail() {
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
    log_error "$@"
}

# ========== 文件检查函数 ==========

# 检查文件是否存在
check_file_exists() {
    local file=$1
    local description=${2:-"文件"}
    if [[ -f "$file" ]]; then
        test_pass "$description 存在: $file"
        return 0
    else
        test_fail "$description 不存在: $file"
        return 1
    fi
}

# 检查目录是否存在
check_dir_exists() {
    local dir=$1
    local description=${2:-"目录"}
    if [[ -d "$dir" ]]; then
        test_pass "$description 存在: $dir"
        return 0
    else
        test_fail "$description 不存在: $dir"
        return 1
    fi
}

# 检查多个文件是否存在
check_files_exist() {
    local all_passed=true
    for file in "$@"; do
        if ! check_file_exists "$file"; then
            all_passed=false
        fi
    done
    $all_passed
}

# ========== JSON元数据检查函数 ==========

# 从JSON文件获取字段值
get_json_field() {
    local json_file=$1
    local field=$1

    if command -v jq &> /dev/null; then
        jq -r ".$field" "$json_file" 2>/dev/null
    else
        log_warning "jq未安装，使用grep解析JSON"
        grep -o "\"$field\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$json_file" |
            cut -d'"' -f4
    fi
}

# 检查元数据文件中的字段值
check_metadata_field() {
    local metadata_file=$1
    local field=$2
    local expected=$3

    if [[ ! -f "$metadata_file" ]]; then
        test_fail "元数据文件不存在: $metadata_file"
        return 1
    fi

    local actual
    if command -v jq &> /dev/null; then
        actual=$(jq -r ".$field" "$metadata_file" 2>/dev/null)
    else
        actual=$(grep -o "\"$field\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$metadata_file" |
                 cut -d'"' -f4)
    fi

    if [[ "$actual" == "$expected" ]]; then
        test_pass "元数据字段 $field = $expected"
        return 0
    else
        test_fail "元数据字段 $field: 期望=$expected, 实际=$actual"
        return 1
    fi
}

# ========== Cache验证函数 ==========

# 验证Step1的cache文件
verify_step1_cache() {
    local cache_dir=${1:-"cache"}
    local all_passed=true

    log_info "验证Step1 cache文件..."

    check_file_exists "$cache_dir/factor_raw.parquet" "原始因子文件" || all_passed=false
    check_file_exists "$cache_dir/factor_standardized.parquet" "标准化因子文件" || all_passed=false
    check_file_exists "$cache_dir/data_returns.parquet" "收益率数据文件" || all_passed=false
    check_file_exists "$cache_dir/data_styles.parquet" "风格数据文件" || all_passed=false
    check_file_exists "$cache_dir/step1_metadata.json" "Step1元数据文件" || all_passed=false

    $all_passed
}

# 验证Step2的cache文件
verify_step2_cache() {
    local cache_dir=${1:-"cache"}
    local all_passed=true

    log_info "验证Step2 cache文件..."

    check_file_exists "$cache_dir/factor_行业市值中性化.parquet" "中性化因子文件" || all_passed=false
    check_file_exists "$cache_dir/step2_metadata.json" "Step2元数据文件" || all_passed=false

    $all_passed
}

# 验证Step3的cache文件
verify_step3_cache() {
    local cache_dir=${1:-"cache"}
    local all_passed=true

    log_info "验证Step3 cache文件..."

    check_file_exists "$cache_dir/factor_回归收益率.parquet" "回归收益率文件" || all_passed=false
    check_file_exists "$cache_dir/factor_回归t值.parquet" "回归t值文件" || all_passed=false
    check_file_exists "$cache_dir/factor_回归收益率_summary.xlsx" "回归收益率汇总" || all_passed=false
    check_file_exists "$cache_dir/factor_回归t值_summary.xlsx" "回归t值汇总" || all_passed=false
    check_file_exists "$cache_dir/step3_metadata.json" "Step3元数据文件" || all_passed=false

    $all_passed
}

# 验证Step4的cache文件
verify_step4_cache() {
    local cache_dir=${1:-"cache"}
    local all_passed=true

    log_info "验证Step4 cache文件..."

    check_file_exists "$cache_dir/factor_ic.parquet" "IC文件" || all_passed=false
    check_file_exists "$cache_dir/factor_rank_ic.parquet" "RankIC文件" || all_passed=false
    check_file_exists "$cache_dir/factor_group_return.parquet" "分组收益文件" || all_passed=false
    check_file_exists "$cache_dir/factor_autocorr.parquet" "自相关文件" || all_passed=false
    check_file_exists "$cache_dir/factor_turnover.parquet" "换手率文件" || all_passed=false
    check_dir_exists "$cache_dir/graphs" "可视化图表目录" || all_passed=false
    check_file_exists "$cache_dir/step4_metadata.json" "Step4元数据文件" || all_passed=false

    $all_passed
}

# 验证Step0的输出
verify_step0_output() {
    local cache_dir=${1:-"cache"}
    local all_passed=true

    log_info "验证Step0输出..."

    check_dir_exists "$cache_dir/qlib_data" "Qlib数据目录" || all_passed=false
    check_dir_exists "$cache_dir/step0_temp" "Step0临时目录" || all_passed=false
    check_file_exists "$cache_dir/step0_metadata.json" "Step0元数据文件" || all_passed=false

    if [[ -d "$cache_dir/qlib_data/instruments" ]]; then
        check_file_exists "$cache_dir/qlib_data/instruments/csi300.txt" "CSI300成分股文件" || all_passed=false
    else
        test_fail "Qlib instruments目录不存在"
        all_passed=false
    fi

    $all_passed
}

# ========== Conda环境函数 ==========

# 检查conda环境是否存在
check_conda_env() {
    local env_name=$1
    if conda info --envs 2>/dev/null | grep -q "^${env_name} "; then
        return 0
    else
        log_error "Conda环境 '${env_name}' 不存在"
        log_info "可用环境:"
        conda info --envs 2>/dev/null | grep -v "^#" | grep -v "^$"
        return 1
    fi
}

# 激活conda环境
activate_conda_env() {
    local env_name=$1
    log_info "激活conda环境: ${env_name}"

    # 初始化conda
    eval "$(conda shell.bash hook 2>/dev/null)"

    # 激活环境
    conda activate ${env_name} 2>/dev/null

    # 验证激活成功
    if [[ "${CONDA_DEFAULT_ENV}" == "${env_name}" ]]; then
        log_success "当前环境: ${CONDA_DEFAULT_ENV}"
        return 0
    else
        log_error "激活conda环境失败"
        return 1
    fi
}

# ========== 清理函数 ==========

# 清理cache目录中的特定文件
clean_cache_files() {
    local cache_dir=${1:-"cache"}
    local pattern=${2:-"*"}

    log_warning "清理文件: $cache_dir/$pattern"
    rm -f ${cache_dir}/${pattern}
    log_success "清理完成"
}

# 清理所有元数据文件
clean_metadata() {
    local cache_dir=${1:-"cache"}
    clean_cache_files "$cache_dir" "step*_metadata.json"
}

# 清理所有cache文件
clean_all_cache() {
    local cache_dir=${1:-"cache"}
    log_warning "清理所有cache文件: $cache_dir"
    rm -rf ${cache_dir}/*.parquet ${cache_dir}/*.xlsx ${cache_dir}/step*_metadata.json
    log_success "清理完成"
}

# ========== 测试报告函数 ==========

print_test_summary() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}测试摘要${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "总测试数: $TESTS_TOTAL"
    echo -e "通过: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "失败: ${RED}$TESTS_FAILED${NC}"

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}所有测试通过!${NC}"
        return 0
    else
        echo -e "${RED}有测试失败!${NC}"
        return 1
    fi
}

# 重置测试计数器
reset_test_counters() {
    TESTS_PASSED=0
    TESTS_FAILED=0
    TESTS_TOTAL=0
}

# ========== 参数验证函数 ==========

# 验证必需的命令是否存在
check_required_commands() {
    local missing_commands=()
    for cmd in "$@"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done

    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_error "缺少必需的命令: ${missing_commands[*]}"
        return 1
    fi
    return 0
}

# 验证Python包是否已安装
check_python_package() {
    local package=$1
    python -c "import ${package}" 2>/dev/null
    return $?
}

# ========== 执行超时控制 ==========

# 带超时执行命令
run_with_timeout() {
    local timeout=$1
    shift
    local command="$@"

    log_info "执行命令 (超时: ${timeout}秒): $command"

    if command -v timeout &> /dev/null; then
        timeout "$timeout" $command
    else
        log_warning "timeout命令不可用，直接执行命令"
        $command
    fi
}

# ========== 输出捕获 ==========

# 执行命令并捕获输出
run_and_capture() {
    local output_file=$1
    shift
    local command="$@"

    log_info "执行命令并捕获输出到: $output_file"
    $command 2>&1 | tee "$output_file"
    return ${PIPESTATUS[0]}
}
