# Step5: 遗传算法因子挖掘

## CLI参数

| 参数 | 类型 | 默认值 | 必需 | 说明 |
|------|------|--------|------|------|
| `--market` | str | "csi300" | 否 | 市场标识 (必须与Step1一致) |
| `--start-date` | str | None | **是** | 起始日期 (必须与Step1一致) |
| `--end-date` | str | None | **是** | 结束日期 (必须与Step1一致) |
| `--random-state` | int | None | 否 | 随机种子 |

### 参数约束
- `market`, `start_date`, `end_date` 必须与Step1完全一致

### 可用市场标识
`csi300`, `sse50`, `csi500`, `csi1000`, `csi2000`, `csi_gem`, `all`

## 输出文件

```
.cache/
└── {market}_{startdate}_{enddate}__gp_seed{seed}.expression.txt
```

**expression.txt** - 因子表达式结果
- 格式: 纯文本，每行一个表达式
- 文件名包含完整元数据（市场、日期范围、随机种子）

## 依赖关系

**前置依赖**:
- Step1生成的 `returns.parquet`

**被依赖**: 无
