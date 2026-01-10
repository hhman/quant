#!/bin/bash
# Step0: 数据预处理 - 从原始CSV到Qlib格式
# 功能: 清洗CSV数据 + 转换为Qlib格式 + 数据校验
# 基于run_dump_all.sh改造,支持CLI风格参数

set -e  # 遇到错误立即退出

# ========== 默认参数 ==========
START_DATE=${START_DATE:-"2008-01-01"}
END_DATE=${END_DATE:-"2025-01-01"}
RAW_DATA_DIR=${RAW_DATA_DIR:-"raw_data"}
CACHE_DIR=${CACHE_DIR:-"cache"}
QLIB_SRC_DIR=${QLIB_SRC_DIR:-"qlib_src"}
VERBOSE=${VERBOSE:-0}
DRY_RUN=${DRY_RUN:-0}

# ========== 解析CLI参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date) START_DATE="$2"; shift 2 ;;
        --end-date) END_DATE="$2"; shift 2 ;;
        --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --qlib-src-dir) QLIB_SRC_DIR="$2"; shift 2 ;;
        --verbose) VERBOSE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help)
            echo "用法: bash step0/step0.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --start-date DATE         起始日期 (default: 2008-01-01)"
            echo "  --end-date DATE           结束日期 (default: 2025-01-01)"
            echo "  --raw-data-dir DIR        原始CSV数据目录 (default: raw_data)"
            echo "  --cache-dir DIR           Cache目录 (default: cache)"
            echo "  --qlib-src-dir DIR        Qlib脚本目录 (default: qlib_src)"
            echo "  --verbose                 显示详细输出"
            echo "  --dry-run                 模拟运行"
            echo "  --help                    显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  bash step0/step0.sh --start-date 2008-01-01 --end-date 2025-01-01 --verbose"
            echo "  bash step0/step0.sh --raw-data-dir my_data --cache-dir my_cache"
            exit 0
            ;;
        *) echo "❌ 未知参数: $1 (使用 --help 查看帮助)"; exit 1 ;;
    esac
done

# ========== 目录配置 ==========
STOCK_DIR="${RAW_DATA_DIR}/stock"
INDEX_DIR="${RAW_DATA_DIR}/index"
FIN_DIR="${RAW_DATA_DIR}/finance"
OUTPUT_DIR="${CACHE_DIR}/step0_temp"
QLIB_DIR="${CACHE_DIR}/qlib_data"

# ========== 打印配置信息 ==========
echo "📊 Step0: 数据预处理"
echo "  时间范围: [$START_DATE, $END_DATE]"
echo "  原始数据目录: $RAW_DATA_DIR"
echo "    - 股票CSV: $STOCK_DIR"
echo "    - 指数CSV: $INDEX_DIR"
echo "    - 财务CSV: $FIN_DIR"
echo "  临时输出目录: $OUTPUT_DIR"
echo "  Qlib数据目录: $QLIB_DIR"
echo "  Qlib脚本目录: $QLIB_SRC_DIR"

if [ "$VERBOSE" -eq 1 ]; then
    echo ""
    echo "🔍 详细模式已启用"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "🔍 模拟运行模式 - 不执行实际操作"
    echo ""
    echo "将要执行的操作:"
    echo "  1. 清洗日线CSV数据 (data_loader.py)"
    echo "  2. 转换为Qlib二进制格式 (dump_bin.py)"
    echo "  3. 复制指数成分股文件"
    echo "  4. 透视财务数据 (pit_loader.py)"
    echo "  5. 转换财务数据为Qlib格式 (dump_pit.py)"
    echo "  6. 校验数据完整性 (check_dump_bin.py, check_data_health.py)"
    exit 0
fi

echo ""

# ========== 检查必要目录是否存在 ==========
if [ ! -d "$STOCK_DIR" ]; then
    echo "❌ 错误: 股票CSV目录不存在: $STOCK_DIR"
    exit 1
fi

if [ ! -d "$INDEX_DIR" ]; then
    echo "❌ 错误: 指数CSV目录不存在: $INDEX_DIR"
    exit 1
fi

if [ ! -d "$FIN_DIR" ]; then
    echo "❌ 错误: 财务CSV目录不存在: $FIN_DIR"
    exit 1
fi

# ========== Step0.1: 清洗日线数据 ==========
echo "⚙️  Step0.1: 清洗日线CSV数据..."
python "$(dirname "$0")/日线数据清洗.py" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --stock-dir "$STOCK_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-dir "$OUTPUT_DIR"

if [ "$VERBOSE" -eq 1 ]; then
    echo "  ✓ 数据清洗完成"
fi

# ========== Step0.2: 转换为Qlib二进制格式 ==========
echo "⚙️  Step0.2: 转换为Qlib二进制格式..."
python "$QLIB_SRC_DIR/scripts/dump_bin.py" dump_all \
  --data_path "${OUTPUT_DIR}/features" \
  --qlib_dir "$QLIB_DIR" \
  --include_fields "open,high,low,close,volume,amount,industry,total_mv,float_mv,factor"

if [ "$VERBOSE" -eq 1 ]; then
    echo "  ✓ Qlib二进制格式转换完成"
fi

# ========== Step0.3: 复制membership文件 ==========
echo "⚙️  Step0.3: 复制指数成分股文件..."
mkdir -p "${QLIB_DIR}/instruments"
cp ${OUTPUT_DIR}/instruments/* ${QLIB_DIR}/instruments/

if [ "$VERBOSE" -eq 1 ]; then
    echo "  ✓ 成分股文件复制完成"
    ls -lh "${QLIB_DIR}/instruments/" | head -5
    if [ $(ls "${QLIB_DIR}/instruments/" | wc -l) -gt 5 ]; then
        echo "  ... (共$(ls "${QLIB_DIR}/instruments/" | wc -l)个文件)"
    fi
fi

# ========== Step0.4: 透视财务数据 ==========
echo "⚙️  Step0.4: 透视财务数据..."
python "$(dirname "$0")/财务数据透视.py" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --input-dir "$FIN_DIR" \
  --output-dir "$OUTPUT_DIR"

if [ "$VERBOSE" -eq 1 ]; then
    echo "  ✓ 财务数据透视完成"
fi

# ========== Step0.5: 转换财务数据为Qlib格式 ==========
echo "⚙️  Step0.5: 转换财务数据为Qlib格式..."
python "$QLIB_SRC_DIR/scripts/dump_pit.py" \
  --csv_path "${OUTPUT_DIR}/financial" \
  --qlib_dir "$QLIB_DIR"

if [ "$VERBOSE" -eq 1 ]; then
    echo "  ✓ 财务数据转换完成"
fi

# ========== Step0.6: 数据质量校验 ==========
echo "⚙️  Step0.6: 校验数据完整性..."
python "$QLIB_SRC_DIR/scripts/check_dump_bin.py" check \
  --qlib_dir "$QLIB_DIR" \
  --csv_path "${OUTPUT_DIR}/features"

python "$QLIB_SRC_DIR/scripts/check_data_health.py" check_data \
  --qlib_dir "$QLIB_DIR"

if [ "$VERBOSE" -eq 1 ]; then
    echo "  ✓ 数据校验完成"
fi

# ========== 保存元数据 ==========
echo "📝 保存Step0元数据..."
mkdir -p "$CACHE_DIR"
cat > "${CACHE_DIR}/step0_metadata.json" << EOF
{
  "start_date": "$START_DATE",
  "end_date": "$END_DATE",
  "raw_data_dir": "$RAW_DATA_DIR",
  "stock_dir": "$STOCK_DIR",
  "index_dir": "$INDEX_DIR",
  "finance_dir": "$FIN_DIR",
  "qlib_dir": "$QLIB_DIR",
  "qlib_src_dir": "$QLIB_SRC_DIR",
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "stage": "step0"
}
EOF

echo "  ✓ 元数据已保存: ${CACHE_DIR}/step0_metadata.json"

echo ""
echo "✅ Step0完成!"
echo "   Qlib数据目录: $QLIB_DIR"
echo "   元数据文件: ${CACHE_DIR}/step0_metadata.json"
echo ""
echo "下一步:"
echo "  python step1/因子提取与预处理.py --start-date $START_DATE --end-date $END_DATE --market csi300 --factor-formulas ... --periods ..."
