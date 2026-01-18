# Step2: 因子中性化

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
└── {market}_{startdate}_{enddate}__neutralized.parquet    # 中性化后的因子
```

### 文件说明

**neutralized.parquet** - 中性化因子
- 索引: MultiIndex (`instrument`, `datetime`)
- 列: 因子表达式列
- 值: 回归残差

## 依赖关系

**前置依赖**:
- Step1生成的 `factor_std.parquet`
- Step1生成的 `styles.parquet`

**被依赖**: Step4依赖 `neutralized.parquet`
