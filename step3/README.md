# Step3: 因子收益率计算

## CLI参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识 (必须与Step1一致) |
| `--start-date` | str | None | **是** | 起始日期 (必须与Step1一致) |
| `--end-date` | str | None | **是** | 结束日期 (必须与Step1一致) |
| `--factor-formulas` | str | None | **是** | 因子表达式列表 (必须是Step1因子的子集) |

### 参数约束
- `market`, `start_date`, `end_date` 必须与Step1完全一致
- `factor_formulas` 必须是Step1中因子的子集

## 输出文件

```
.cache/
├── {market}_{startdate}_{enddate}__return_coef.parquet    # 收益率系数时间序列
├── {market}_{startdate}_{enddate}__return_tval.parquet    # t值时间序列
├── {market}_{startdate}_{enddate}__return_coef_summary.xlsx    # 系数汇总统计
└── {market}_{startdate}_{enddate}__return_tval_summary.xlsx    # t值汇总统计
```

### 文件说明

**return_coef.parquet** - 收益率系数时间序列
- 索引: MultiIndex (`datetime`, `period`)
- 列: 因子表达式列

**return_tval.parquet** - t值时间序列
- 索引: MultiIndex (`datetime`, `period`)
- 列: 因子表达式列

**return_coef_summary.xlsx** - 系数汇总统计
- 列: 因子表达式列
- 行: `因子收益率均值`, `因子收益率序列t检验`

**return_tval_summary.xlsx** - t值汇总统计
- 列: 因子表达式列
- 行: `|t|均值`, `|t|>2占比`, `t均值`, `t均值/标准差`

## 依赖关系

**前置依赖**:
- Step1生成的 `factor_std.parquet`
- Step1生成的 `returns.parquet`
- Step1生成的 `styles.parquet`
