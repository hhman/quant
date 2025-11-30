START_DATE=${1:-"2008-01-01"}
END_DATE=${2:-"2025-01-01"}
STOCK_DIR=${3:-"stock"}
INDEX_DIR=${4:-"index"}
OUTPUT_DIR=${5:-"output"}

python data_loader.py \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --stock-dir "${STOCK_DIR}" \
  --index-dir "${INDEX_DIR}" \
  --output-dir "${OUTPUT_DIR}"

python qlib_src/scripts/dump_bin.py dump_all \
  --data_path "${OUTPUT_DIR}/features" \
  --qlib_dir "${OUTPUT_DIR}/qlib_data" \
  --include_fields "open,high,low,close,volume,amount,industry,total_mv,factor"

cp ${OUTPUT_DIR}/instruments/* ${OUTPUT_DIR}/qlib_data/instruments

python qlib_src/scripts/check_dump_bin.py check \
  --qlib_dir "${OUTPUT_DIR}/qlib_data" \
  --csv_path "${OUTPUT_DIR}/features"

python qlib_src/scripts/check_data_health.py check_data \
  --qlib_dir "${OUTPUT_DIR}/qlib_data"
