# Step1: 因子提取与预处理

## CLI参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识 |
| `--start-date` | str | - | **是** | 起始日期 (YYYY-MM-DD) |
| `--end-date` | str | - | **是** | 结束日期 (YYYY-MM-DD) |
| `--factor-formulas` | str | - | **是** | 因子表达式列表，分号分隔 |

### 可用市场标识
`csi300`, `sse50`, `csi500`, `csi1000`, `csi2000`, `csi_gem`, `all`

## 输出文件

```
.cache/
├── {market}_{startdate}_{enddate}__factor_raw.parquet    # 原始因子 (去极值后)
└── {market}_{startdate}_{enddate}__factor_std.parquet    # 标准化因子
```

### 文件说明

**factor_raw.parquet** - 原始因子 (去极值后)
- 索引: MultiIndex (`instrument`, `datetime`)
- 列: 因子表达式列

**factor_std.parquet** - 标准化因子
- 索引: MultiIndex (`instrument`, `datetime`)
- 列: 因子表达式列

## 依赖关系

**前置依赖**:
- Step0生成的 `qlib_data/`

**被依赖**:
- Step2依赖 `factor_std.parquet`
- Step3依赖 `factor_std.parquet`
