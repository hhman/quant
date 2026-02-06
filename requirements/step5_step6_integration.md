# Step5 → Step6 集成需求

## 一、问题背景

当前架构中，Step5 使用 Gplearn 挖掘因子，但生成的表达式无法被 Step1 识别和计算。

工作流 B 存在断层：
- Step5: Gplearn 挖掘（测试集）
- Step1-4: Qlib 评估（验证集）

**核心问题**: Gplearn 表达式与 Qlib 表达式不兼容。

## 二、解决方案

### 方案选择

**新增 Step6 模块**，专门负责 Gplearn 表达式计算。

架构调整为：
- 工作流 A: Step1 (Qlib) → Step2-4
- 工作流 B: Step5 (Gplearn 挖掘) → Step6 (Gplearn 计算) → Step2-4

### 设计决策

**决策 1: 新增 Step6 vs 修改 Step1**

选择新增 Step6，理由：
- 职责分离：Step1 专注 Qlib，Step6 专注 Gplearn
- 兼容性：不影响现有工作流 A
- 可维护性：模块边界清晰

**决策 2: Pickle 序列化 vs 表达式解析**

选择 pickle 序列化程序对象，理由：
- 避免实现复杂的表达式解析器
- Gplearn 程序对象可直接执行
- 自定义算子可通过 cloudpickle 序列化

**决策 3: 智能合并因子**

支持 Step1 和 Step6 输出合并到同一文件，理由：
- 允许混合使用 Qlib 和 Gplearn 因子
- Step2-4 无需感知因子来源
- 用户可灵活组合因子

## 三、模块设计

### Step5 职责

**输入**: 测试集数据（时间范围 + 市场标识）
**输出**: Pickle 文件

Pickle 文件结构：
- `metadata`: 表达式字符串、依赖特征列表、适应度值
- `program`: Gplearn `_Program` 对象

**关键实现点**：
- 导出时提取 `base_features`（按特征索引排序）
- 使用 `wrap=True` 确保自定义算子可序列化
- 单个文件包含所有程序（支持多程序输出）

### Step6 职责

**输入**:
- 验证集时间范围 + 市场标识
- Step5 输出的 pickle 文件路径

**输出**: factor_raw.parquet, factor_std.parquet

**核心流程**：
1. 加载 pickle 文件，提取程序对象和元数据
2. 根据 `base_features` 加载 Qlib 特征数据
3. 使用 `flatten_features()` 展平为 2D 数组
4. 执行 `program.execute(X_flat)`
5. 恢复为 MultiIndex DataFrame
6. 去极值、标准化（与 Step1 相同逻辑）
7. 保存到缓存目录

**关键约束**：
- 特征加载顺序必须与 Step5 训练时严格一致
- 支持智能合并模式（不覆盖现有因子）

### CacheManager 增强

**新增合并模式**：

方法签名：
```python
def write_dataframe(df, name, mode="merge")
```

模式说明：
- `mode="overwrite"`: 覆盖现有文件
- `mode="merge"`: 与现有文件按列合并

**合并逻辑**：
- 读取现有 parquet 文件
- 按索引对齐合并列
- 处理列名冲突（添加后缀）
- 保存合并后的文件

## 四、数据流设计

### 文件命名约定

**Step5 输出**:
```
{market}_{test_start}_{test_end}__gp_seed{seed}.pkl
```

**Step6 输出**:
```
.cache/{market}_{val_start}_{val_end}/
├── factor_raw.parquet
└── factor_std.parquet
```

### 因子列名策略

**Step1**: 列名 = Qlib 表达式字符串
**Step6**: 列名 = Gplearn 表达式字符串（或哈希值）

冲突处理：
- 检测列名重复
- 自动添加后缀 `_dup`
- 记录日志提醒用户

## 五、边界条件

### 特征顺序一致性

Step5 保存 `base_features` 时必须按特征索引排序，Step6 加载时严格按此顺序。

**为什么重要**：
Gplearn 程序内部使用特征索引（0, 1, 2...），顺序错误会导致计算结果错误。

### 缺失值处理

执行后检查 NaN 比例，超过阈值（如 50%）发出警告。

### 多程序支持

Step5 可输出多个程序到单个 pickle 文件，Step6 循环处理每个程序。

### 时间范围独立性

Step5 和 Step6 使用不同时间范围，程序对象不绑定训练数据。

## 六、实现优先级

### P0: 核心功能
1. Step6 CLI 框架
2. Step6 核心计算逻辑
3. Step5 输出 pickle 文件
4. CacheManager 合并模式

### P1: 增强功能
5. 列名冲突处理
6. NaN 检查与告警
7. 多程序支持

### P2: 优化功能
8. 性能优化
9. 进度反馈
10. 单元测试覆盖

## 七、验收标准

### 功能验收
- Step5 可成功序列化自定义算子
- Step6 可加载并执行程序
- 输出格式与 Step1 兼容
- 智能合并正常工作
- 支持跨时间范围计算

### 性能验收
- 程序加载时间 < 1 秒
- 因子计算时间与 Step1 相当
- 内存占用合理

## 八、不做什么

- 不实现 Gplearn 表达式解析器（直接使用 pickle）
- 不修改 Step1 逻辑（保持工作流 A 独立）
- 不支持跨 market 计算（market 必须一致）
- 不自动处理程序执行异常（由用户决定是否重试）
