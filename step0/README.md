# Step0: 数据预处理

## 📋 概述

**Step0** 是量化因子分析流程的第一步，负责将原始CSV格式的金融数据转换为Qlib可用的二进制格式。

### 核心功能

1. **日线数据清洗** - 处理股票和指数的日线OHLCV数据
2. **Qlib格式转换** - 将清洗后的数据转换为Qlib二进制格式
3. **财务数据透视** - 将宽表财务数据转换为PIT (Point-in-Time) 长表格式
4. **数据质量校验** - 验证转换后数据的完整性和一致性

### 依赖关系

- **前置依赖**: 原始CSV数据（股票、指数、财务）
- **后续依赖**: Step1（因子提取）依赖Step0生成的Qlib数据
- **独立执行**: 作为数据预处理流程，可独立运行

## 📥 输入数据

### 目录结构要求

```
raw_data/
├── stock/          # 股票日线CSV数据
├── index/          # 指数日线CSV数据
└── finance/        # 财务数据CSV（宽表）
```

### 股票CSV格式 (`stock/`)

**必需列**:
- `股票代码` - 如 "SZ000001"
- `股票名称` - 如 "平安银行"
- `交易日期` - 格式: YYYYMMDD
- `开盘价`, `最高价`, `最低价`, `收盘价`, `前收盘价`
- `成交量`, `成交额`
- `总市值`, `流通市值`
- `沪深300成分股`, `上证50成分股`, `中证500成分股`, `中证1000成分股`, `中证2000成分股`, `创业板指成分股`
- `新版申万一级行业名称`

**编码**: GBK
**跳过行数**: 1 (第一行为标题)

### 指数CSV格式 (`index/`)

**必需列**:
- 日期列: `candle_end_time` 或 `date`
- 代码列: `index_code`, `symbol` 或 `code`
- OHLCV: `open`, `high`, `low`, `close`, `volume`, `amount`

**列名**: 不区分大小写

### 财务CSV格式 (`finance/`)

**必需列**:
- `stock_code` - 股票代码
- `report_date` - 报告期 (YYYYMMDD)
- `publish_date` - 公告日 (YYYYMMDD)
- 财务指标列 - 如 `净利润`, `营业收入` 等

**编码**: GBK
**跳过行数**: 1

## ⚙️ 输入参数

### CLI参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--start-date` | str | "2008-01-01" | 否 | 起始日期 |
| `--end-date` | str | "2025-01-01" | 否 | 结束日期 |
| `--raw-data-dir` | str | "raw_data" | 否 | 原始CSV数据目录 |
| `--cache-dir` | str | "cache" | 否 | Cache目录 |
| `--qlib-src-dir` | str | "qlib_src" | 否 | Qlib脚本目录 |
| `--verbose` | flag | False | 否 | 显示详细输出 |
| `--dry-run` | flag | False | 否 | 模拟运行，不实际执行 |

### 自动生成的子目录

脚本会自动在 `raw_data_dir` 下查找以下子目录:
- `stock/` - 股票CSV
- `index/` - 指数CSV
- `finance/` - 财务CSV

## 📤 输出结果

### 目录结构

```
cache/
├── step0_temp/              # 临时数据
│   ├── features/            # 清洗后的日线数据
│   │   ├── SH600000.csv
│   │   └── SZ000001.csv
│   ├── instruments/         # 指数成分股文件
│   │   ├── csi300.txt
│   │   ├── sse50.txt
│   │   └── ...
│   ├── financial/           # 透视后的财务数据
│   │   ├── SH600000.csv
│   │   └── SZ000001.csv
│   └── industry_mapping.json # 行业名称映射
│
├── qlib_data/               # Qlib二进制数据
│   ├── instruments/         # 成分股文件（复制）
│   └── [Qlib数据文件]
│
└── step0_metadata.json      # Step0元数据
```

### 输出文件说明

#### 1. 清洗后的日线数据 (`features/*.csv`)

**格式**: 标准化OHLCV数据

**列名**:
- `open`, `high`, `low`, `close`, `volume`, `amount`
- `industry` - 行业ID（整数）
- `total_mv`, `float_mv` - 市值
- `factor` - 复权因子

**索引**: `date` (DatetimeIndex)

**编码**: UTF-8-sig

#### 2. 成分股文件 (`instruments/*.txt`)

**格式**: `股票代码\t起始日期\t结束日期`

**示例**:
```
SH600000	2008-01-01	2025-01-01
SZ000001	2008-01-01	2025-01-01
```

**文件命名**:
- `csi300.txt` - 沪深300成分股
- `sse50.txt` - 上证50成分股
- `csi500.txt` - 中证500成分股
- `csi1000.txt` - 中证1000成分股
- `csi2000.txt` - 中证2000成分股
- `csi_gem.txt` - 创业板指成分股

#### 3. 财务数据 (`financial/*.csv`)

**格式**: PIT (Point-in-Time) 长表

**列名**:
- `stock_code` - 股票代码（如 "SH600000"）
- `date` - 公告日 (YYYY-MM-DD)
- `period` - 报告期 (YYYYQ，如 20231 = 2023Q1)
- `field` - 指标名称
- `value` - 指标值

**示例**:
```
stock_code,date,period,field,value
SH600000,2018-03-30,20171,净利润,123.45
SH600000,2018-03-30,20171,营业收入,678.90
```

**编码**: UTF-8-sig

#### 4. 行业映射 (`industry_mapping.json`)

**格式**: 行业名称 → 行业ID映射

```json
{
  "银行": 1,
  "非银金融": 2,
  "医药生物": 3,
  ...
}
```

#### 5. Step0元数据 (`step0_metadata.json`)

```json
{
  "start_date": "2008-01-01",
  "end_date": "2025-01-01",
  "raw_data_dir": "raw_data",
  "stock_dir": "raw_data/stock",
  "index_dir": "raw_data/index",
  "finance_dir": "raw_data/finance",
  "qlib_dir": "cache/qlib_data",
  "qlib_src_dir": "qlib_src",
  "generated_at": "2025-01-10T12:00:00Z",
  "stage": "step0"
}
```

## 🚀 使用示例

### 基本用法

```bash
# 使用默认参数
bash step0/step0.sh

# 指定日期范围
bash step0/step0.sh --start-date 2020-01-01 --end-date 2024-01-01

# 详细输出
bash step0/step0.sh --verbose
```

### 自定义数据目录

```bash
# 指定原始数据目录
bash step0/step0.sh --raw-data-dir /path/to/data

# 指定输出目录
bash step0/step0.sh --cache-dir /path/to/cache
```

### 模拟运行

```bash
# 查看将要执行的操作，不实际运行
bash step0/step0.sh --dry-run
```

### 组合参数

```bash
# 完整参数示例
bash step0/step0.sh \
  --start-date 2010-01-01 \
  --end-date 2023-12-31 \
  --raw-data-dir my_data \
  --cache-dir my_cache \
  --verbose
```

## 🔄 执行流程

Step0按以下顺序执行6个子步骤:

1. **Step0.1**: 清洗日线CSV数据 (`日线数据清洗.py`)
   - 读取股票和指数CSV
   - 规范化股票代码（交易所前缀大写）
   - 填充缺失值
   - 计算复权因子
   - 提取成分股信息
   - 输出到 `step0_temp/features/`

2. **Step0.2**: 转换为Qlib二进制格式 (`qlib_src/scripts/dump_bin.py`)
   - 调用Qlib内置脚本
   - 生成二进制数据文件
   - 输出到 `qlib_data/`

3. **Step0.3**: 复制成分股文件
   - 复制 `step0_temp/instruments/` 到 `qlib_data/instruments/`

4. **Step0.4**: 透视财务数据 (`财务数据透视.py`)
   - 读取宽表财务CSV
   - 透视成长表格式
   - 输出到 `step0_temp/financial/`

5. **Step0.5**: 转换财务数据为Qlib格式 (`qlib_src/scripts/dump_pit.py`)
   - 调用Qlib内置脚本
   - 生成PIT二进制数据

6. **Step0.6**: 校验数据完整性
   - `qlib_src/scripts/check_dump_bin.py` - 检查二进制数据
   - `qlib_src/scripts/check_data_health.py` - 检查数据健康度

## ⚠️ 注意事项

### 数据要求

1. **日期格式**: 必须为 YYYYMMDD 或 YYYY-MM-DD
2. **股票代码**: 必须包含交易所前缀（SZ/SH/BJ）
3. **编码**: 股票和财务CSV必须为GBK编码
4. **跳过行**: 第一行通常是标题，需要跳过

### 常见问题

1. **内存占用**: 处理大量股票时内存占用较高，建议分批处理
2. **复权因子**: 使用前收盘价计算后复权因子
3. **行业映射**: 自动生成行业ID映射，保存在 `industry_mapping.json`
4. **成分股缺失**: 如果某个指数成分股文件为空，会被删除

### 性能优化

- 使用 `--verbose` 查看详细进度
- 大数据集建议分年度处理
- 财务数据透视可能耗时较长

## 🔗 后续步骤

Step0完成后，可以继续执行:

```bash
# Step1: 因子提取与预处理
python step1/因子提取与预处理.py \
  --market csi300 \
  --factor-formulas "Ref($close,60)/$close,MOM($close,20)" \
  --periods 1d,1w,1m \
  --provider-uri cache/qlib_data
```

## 🤖 Agent Skill转换指南

### Skill定义

**Skill名称**: `data_preprocessing`

**功能描述**: 将原始CSV金融数据转换为Qlib可用格式

### 输入参数

```python
{
    "start_date": "2008-01-01",      # str: 起始日期
    "end_date": "2025-01-01",        # str: 结束日期
    "raw_data_dir": "raw_data",      # str: 原始数据目录
    "cache_dir": "cache",            # str: 输出目录
    "qlib_src_dir": "qlib_src",      # str: Qlib脚本目录
    "verbose": false                 # bool: 详细输出
}
```

### 输出数据

```python
{
    "qlib_dir": "cache/qlib_data",          # str: Qlib数据目录
    "step0_metadata": "cache/step0_metadata.json",  # str: 元数据文件
    "status": "success",                    # str: 执行状态
    "message": "数据预处理完成"              # str: 执行信息
}
```

### 依赖关系

- **前置**: 无（作为第一步）
- **后置**: Step1依赖Step0的 `qlib_data`

### 执行方式

```bash
# 调用step0.sh
bash step0/step0.sh [参数]

# 或使用Python执行子脚本
python step0/日线数据清洗.py [参数]
python step0/财务数据透视.py [参数]
```

### 验证方法

```python
# 检查输出文件
from pathlib import Path

def verify_step0(cache_dir="cache"):
    """验证Step0输出"""
    qlib_dir = Path(cache_dir) / "qlib_data"
    metadata_file = Path(cache_dir) / "step0_metadata.json"

    assert qlib_dir.exists(), "Qlib数据目录不存在"
    assert metadata_file.exists(), "元数据文件不存在"

    # 检查Qlib数据完整性
    instruments_dir = qlib_dir / "instruments"
    assert instruments_dir.exists(), "成分股目录不存在"

    return True
```

## 📚 相关文件

- `step0/step0.sh` - 主执行脚本
- `step0/日线数据清洗.py` - 日线数据清洗
- `step0/财务数据透视.py` - 财务数据透视
- `cli_config.py` - CLI参数解析
- `metadata.py` - 元数据管理
