# Step0: 数据预处理

## CLI参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--start-date` | str | 是 | 起始日期 (YYYY-MM-DD) |
| `--end-date` | str | 是 | 结束日期 (YYYY-MM-DD) |

## 输出文件

```
.cache/
├── step0_temp/              # 临时数据
│   ├── features/            # 清洗后的日线数据 (CSV)
│   ├── instruments/         # 指数成分股文件 (TXT)
│   ├── financial/           # 透视后的财务数据 (CSV)
│   └── industry_mapping.json # 行业名称映射
├── qlib_data/               # Qlib二进制数据 (持久)
│   └── instruments/         # 成分股文件
├── industry_mapping.json    # 行业名称映射 (跨市场复用)
├── all__styles.parquet      # 风格变量 (全市场，固定文件名)
└── all__returns.parquet     # 未来收益率 (全市场，固定文件名)
```

### 文件说明

**features/*.csv** - 清洗后的日线数据
- 列: `open`, `high`, `low`, `close`, `volume`, `amount`, `industry`, `total_mv`, `float_mv`, `factor`
- 索引: `date` (DatetimeIndex)

**instruments/*.txt** - 成分股文件
- 格式: `股票代码\t起始日期\t结束日期`
- 文件: `csi300.txt`, `sse50.txt`, `csi500.txt`, `csi1000.txt`, `csi2000.txt`, `csi_gem.txt`

**financial/*.csv** - 财务数据 (PIT长表格式)
- 列: `stock_code`, `date`, `period`, `field`, `value`

**styles.parquet** - 风格变量
- 索引: MultiIndex (`instrument`, `datetime`)
- 列: `$total_mv`, `$industry`, `$float_mv`

**returns.parquet** - 未来收益率
- 索引: MultiIndex (`instrument`, `datetime`)
- 列: `ret_1d`, `ret_1w`, `ret_1m`

## 依赖关系

**前置依赖**: 无

**被依赖**:
- Step1依赖 `qlib_data/`
- Step2依赖 `styles.parquet`
- Step3依赖 `returns.parquet`, `styles.parquet`
- Step4依赖 `returns.parquet`
- Step5依赖 `returns.parquet`
