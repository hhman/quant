# 量化因子分析流程 - 测试套件

本目录包含量化因子分析流程的完整测试套件，使用CLI参数方式验证各个step的功能。

## 快速开始

### 1. 激活Conda环境
```bash
conda activate quant
```

### 2. 运行所有测试
```bash
# 主测试脚本会自动激活quant环境
bash tests/run_all_tests.sh
```

### 3. 查看测试报告
```bash
# 测试报告
cat tests/test_report.txt

# 详细日志
cat tests/test_log.txt
```

## 测试结构

```
tests/
├── run_all_tests.sh              # 主测试入口
├── lib/
│   └── test_utils.sh             # 测试工具函数库
├── phase1_basic/                 # Phase 1: 基本运行测试
│   ├── test_step0_basic.sh       # Step0数据预处理
│   ├── test_step1_basic.sh       # Step1因子提取与预处理
│   ├── test_step2_basic.sh       # Step2因子中性化
│   ├── test_step3_basic.sh       # Step3因子收益率计算
│   └── test_step4_basic.sh       # Step4因子绩效评估
├── phase2_cache/                 # Phase 2: Cache机制测试
│   ├── test_cache_subset.sh      # 子集匹配测试
│   ├── test_cache_combination.sh # 组合子集匹配
│   └── test_cache_dependency.sh  # 依赖链测试
└── phase3_errors/                # Phase 3: 错误处理测试
    ├── test_error_cache_mismatch.sh           # Cache不匹配错误
    ├── test_error_missing_prerequisite.sh     # 缺失前置step
    ├── test_error_step1_required_params.sh    # Step1必需参数
    ├── test_dry_run.sh                        # Dry-run模式
    └── test_boundary_data.sh                  # 边界数据测试
```

## 使用方法

### 运行所有测试
```bash
bash tests/run_all_tests.sh
```

### 只运行特定Phase
```bash
# Phase 1: 基本运行测试
bash tests/run_all_tests.sh --phase 1

# Phase 2: Cache机制测试
bash tests/run_all_tests.sh --phase 2

# Phase 3: 错误处理测试
bash tests/run_all_tests.sh --phase 3
```

### 只运行特定Step
```bash
# 只测试Step1
bash tests/run_all_tests.sh --step step1

# 只测试Step2
bash tests/run_all_tests.sh --step step2
```

### 使用不同的Conda环境
```bash
bash tests/run_all_tests.sh --conda-env myenv
```

### 自定义测试参数
```bash
bash tests/run_all_tests.sh \
    --market csi300 \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --verbose
```

### 保留测试结果
```bash
# 不自动清理测试结果
bash tests/run_all_tests.sh --no-cleanup
```

## 单独运行测试脚本

如果你想单独运行某个测试脚本，需要先手动激活conda环境：

```bash
# 激活环境
conda activate quant

# 运行特定测试
bash tests/phase1_basic/test_step1_basic.sh
```

## 测试说明

### Phase 1: 基本运行测试
确保每个step在正常情况下可以运行并生成正确的输出文件。

- **Step0**: 数据预处理（CSV → Qlib格式）
- **Step1**: 因子提取与预处理
- **Step2**: 因子中性化
- **Step3**: 因子收益率计算
- **Step4**: 因子绩效评估

### Phase 2: Cache机制测试
验证cache子集匹配、依赖关系等机制。

- **因子子集匹配**: 从多个因子中选择部分因子
- **周期子集匹配**: 从多个周期中选择部分周期
- **时间范围子集匹配**: 从长时间范围中选择子时间段
- **组合子集匹配**: 三个维度同时进行子集匹配
- **依赖链测试**: 验证Step之间的依赖关系

### Phase 3: 错误处理测试
验证错误处理和边界场景。

- **Cache不匹配**: 市场、日期、因子不匹配
- **缺失前置step**: 直接运行后续step
- **Step1必需参数**: 缺少factor-formulas或periods
- **Dry-run模式**: 模拟运行不实际执行
- **边界数据**: 极短时间、单日数据、无效参数

## 测试输出

### 终端输出
- 实时显示每个测试的执行状态
- 使用颜色标识：✓ (成功)、✗ (失败)、! (警告)

### 文件输出
- `test_report.txt`: 测试报告摘要
- `test_log.txt`: 详细执行日志

### Cache文件
测试会生成以下cache文件：
```
cache/
├── factor_raw.parquet                    # 原始因子
├── factor_standardized.parquet           # 标准化因子
├── factor_行业市值中性化.parquet          # 中性化因子
├── factor_回归收益率.parquet              # 回归系数
├── factor_回归t值.parquet                 # t统计量
├── factor_ic.parquet                     # IC序列
├── factor_rank_ic.parquet                # RankIC序列
├── factor_group_return.parquet           # 分组收益
├── factor_autocorr.parquet               # 自相关
├── factor_turnover.parquet               # 换手率
├── data_returns.parquet                  # 收益率数据
├── data_styles.parquet                   # 风格数据
└── step*_metadata.json                   # 各step元数据
```

## 环境要求

### Conda环境
- 需要名为`quant`的conda环境
- 可通过`--conda-env`参数指定其他环境

### Python依赖
- qlib
- pandas
- numpy
- statsmodels
- openpyxl

### 数据要求
- `raw_data/`: 原始CSV数据目录
  - `stock/`: 股票日线数据
  - `index/`: 指数数据
  - `finance/`: 财务数据
- `qlib_src/`: Qlib脚本目录（用于Step0）

## 故障排查

### Conda环境未激活
```bash
# 检查环境是否存在
conda info --envs

# 如果不存在，创建环境
conda create -n quant python=3.9
conda activate quant
pip install qlib pandas numpy statsmodels openpyxl
```

### Step0执行失败
- 检查`raw_data/`目录是否存在
- 检查`qlib_src/`目录是否存在
- 确保有足够的磁盘空间

### 其他测试失败
- 查看详细日志：`cat tests/test_log.txt`
- 使用`--verbose`选项重新运行
- 检查cache目录权限

## 执行顺序建议

1. **第一步**: 运行Phase 1，确保每个step基本可运行
2. **第二步**: 运行Phase 2，验证cache机制
3. **第三步**: 运行Phase 3，验证错误处理

## 持续集成

测试脚本设计为可在CI环境中运行：
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    conda activate quant
    bash tests/run_all_tests.sh
```

## 贡献指南

添加新测试时：
1. 在相应的phase目录下创建`test_*.sh`文件
2. 加载测试工具函数：`source "$PROJECT_ROOT/tests/lib/test_utils.sh"`
3. 使用`test_pass`和`test_fail`记录结果
4. 更新主测试脚本以包含新测试

## 许可证

与主项目保持一致
