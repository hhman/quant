# Step4: 因子绩效评估

## CLI参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识 (必须与Step2一致) |
| `--start-date` | str | None | **是** | 起始日期 (必须与Step2一致) |
| `--end-date` | str | None | **是** | 结束日期 (必须与Step2一致) |
| `--factor-formulas` | str | None | **是** | 因子表达式列表 (必须是Step2因子的子集) |

### 参数约束
- `market`, `start_date`, `end_date` 必须与Step2完全一致
- `factor_formulas` 必须是Step2中因子的子集

## 输出文件

```
.cache/
├── {market}_{startdate}_{enddate}__ic.parquet             # IC时间序列
├── {market}_{startdate}_{enddate}__rank_ic.parquet        # RankIC时间序列
├── {market}_{startdate}_{enddate}__group_return.parquet   # 分组收益时间序列
├── {market}_{startdate}_{enddate}__autocorr.parquet       # 自相关时间序列
├── {market}_{startdate}_{enddate}__turnover.parquet       # 换手率时间序列
├── {market}_{startdate}_{enddate}__ic_summary.xlsx        # IC汇总统计
├── {market}_{startdate}_{enddate}__rank_ic_summary.xlsx   # RankIC汇总统计
├── {market}_{startdate}_{enddate}__group_return_summary.xlsx  # 分组收益汇总
├── {market}_{startdate}_{enddate}__autocorr_summary.xlsx  # 自相关汇总
├── {market}_{startdate}_{enddate}__turnover_summary.xlsx  # 换手率汇总
└── graphs/                                                # 可视化图表
    ├── group_return*.png
    ├── pred_ic*.png
    ├── pred_autocorr*.png
    └── pred_turnover*.png
```

### 文件说明

**ic.parquet** - IC时间序列
- 索引: MultiIndex (`datetime`, `period`)
- 列: 因子表达式列

**rank_ic.parquet** - RankIC时间序列
- 索引: MultiIndex (`datetime`, `period`)
- 列: 因子表达式列

**group_return.parquet** - 分组收益时间序列
- 索引: MultiIndex (`datetime`, `period`, `group`)
- 列: 因子表达式列

**autocorr.parquet** - 自相关时间序列
- 索引: MultiIndex (`datetime`, `period`, `lag`)
- 列: 因子表达式列

**turnover.parquet** - 换手率时间序列
- 索引: MultiIndex (`datetime`, `period`, `threshold`)
- 列: 因子表达式列

**graphs/** - 可视化图表
- 分组收益图、IC图、自相关图、换手率图

## 依赖关系

**前置依赖**:
- Step2生成的 `neutralized.parquet`
- Step1生成的 `returns.parquet`
