#!/bin/bash
# Step0: 数据预处理 - 从原始CSV到Qlib格式

set -e

# ========== 解析CLI参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date) START_DATE="$2"; shift 2 ;;
        --end-date) END_DATE="$2"; shift 2 ;;
        --help)
            echo "用法: bash step0/cli.sh --start-date DATE --end-date DATE"
            echo ""
            echo "必选参数:"
            echo "  --start-date DATE         起始日期 (YYYY-MM-DD)"
            echo "  --end-date DATE           结束日期 (YYYY-MM-DD)"
            exit 0
            ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

# 检查必选参数
if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
    echo "❌ 错误: 必须指定 --start-date 和 --end-date"
    exit 1
fi

# ========== 目录配置 ==========
STOCK_DIR="stock"
INDEX_DIR="index"
FIN_DIR="finance"
CACHE_DIR=".cache"
OUTPUT_DIR="${CACHE_DIR}/step0_temp"
QLIB_DIR="${CACHE_DIR}/qlib_data"
QLIB_SRC_DIR="qlib_src"

# ========== 检查必要目录 ==========
if [ ! -d "$STOCK_DIR" ]; then echo "❌ 股票CSV目录不存在: $STOCK_DIR"; exit 1; fi
if [ ! -d "$INDEX_DIR" ]; then echo "❌ 指数CSV目录不存在: $INDEX_DIR"; exit 1; fi
if [ ! -d "$FIN_DIR" ]; then echo "❌ 财务CSV目录不存在: $FIN_DIR"; exit 1; fi

# ========== Step0.1: 清洗日线数据 ==========
python "$(dirname "$0")/日线数据清洗.py" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --stock-dir "$STOCK_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-dir "$OUTPUT_DIR"

# ========== Step0.2: 转换为Qlib二进制格式 ==========
python "$QLIB_SRC_DIR/scripts/dump_bin.py" dump_all \
  --data_path "${OUTPUT_DIR}" \
  --qlib_dir "$QLIB_DIR" \
  --include_fields "open,high,low,close,volume,amount,industry,total_mv,float_mv,factor"

# ========== Step0.3: 复制membership文件 ==========
mkdir -p "${QLIB_DIR}/instruments"
if ls "${OUTPUT_DIR}"/instruments/*.txt 1> /dev/null 2>&1; then
    cp "${OUTPUT_DIR}"/instruments/*.txt "${QLIB_DIR}/instruments/"
else
    echo "❌ 未找到成分股文件: ${OUTPUT_DIR}/instruments/*.txt"
    exit 1
fi

# ========== Step0.4: 透视财务数据 ==========
python "$(dirname "$0")/财务数据透视.py" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --finance-dir "$FIN_DIR" \
  --output-dir "$OUTPUT_DIR/financial"

# ========== Step0.5: 转换财务数据为Qlib格式 ==========
python "$QLIB_SRC_DIR/scripts/dump_pit.py" \
  --csv_path "${OUTPUT_DIR}/financial" \
  --qlib_dir "$QLIB_DIR"

# ========== Step0.6: 数据质量校验 ==========
python "$QLIB_SRC_DIR/scripts/check_dump_bin.py" check \
  --qlib_dir "$QLIB_DIR" \
  --csv_path "${OUTPUT_DIR}"

python "$QLIB_SRC_DIR/scripts/check_data_health.py" check_data \
  --qlib_dir "$QLIB_DIR"

echo "✅ Step0完成!"
