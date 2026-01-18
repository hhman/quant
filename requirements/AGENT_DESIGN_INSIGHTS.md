# Agent Skills 架构设计核心洞察

> 本文档记录了从 Step-CLI 到 Agent Skills 架构转型设计过程中的关键思考和领悟

---

## 🎯 核心理念转变

### 从"教 Agent"到"信任 Agent"

**❌ 错误思路**: 过度工程化
```python
# 试图通过代码教 Agent 如何理解
- 写语义解析器 (semantic_parser.py)
- 写表达式构建器 (expression_builder.py)
- 写因子规则库 (FACTOR_PATTERNS.md)
- 注册自定义函数 (factor_library.py)

# 结果:
# - 增加了维护负担
# - 限制了 Agent 的灵活性
# - 违背了 Agent 的设计初衷
```

**✅ 正确思路**: 文档驱动 + 示例学习
```python
# 只需提供清晰的能力说明
skills/__init__.py              # 纯导出,100行
parent_skills/single_factor_analysis.py  # 纯编排,150行
docs/AGENT_GUIDE.md             # 文档+示例,300行

# 结果:
# - 零维护成本
# - Agent 可以举一反三
# - 符合 Agent 的设计理念
```

---

## 💡 三层能力架构

### Skills 的价值 = 能力封装,不是知识灌输

```
┌─────────────────────────────────────────┐
│  Agent (智能层 - 可进化)                  │
│  - 理解用户意图                           │
│  - 推理因子表达式                         │
│  - 决定调用策略                           │
└─────────────────────────────────────────┘
                    ↓ 调用
┌─────────────────────────────────────────┐
│  Skills (能力层 - 稳定)                   │
│  - single_factor_analysis               │
│  - factor_extraction                     │
│  - neutralize_factor                     │
└─────────────────────────────────────────┘
                    ↓ 调用
┌─────────────────────────────────────────┐
│  Step 1-4 (实现层 - 不变)                 │
│  - step1/cli.py                          │
│  - step2/cli.py                          │
│  - step3/cli.py                          │
│  - step4/cli.py                          │
└─────────────────────────────────────────┘
```

---

## 📚 Agent 学习机制

### 自然语言 → Qlib表达式 的转换

**场景**: 用户说"60日动量因子"

#### Agent 的推理过程 (自动完成)

```python
# 1. 理解用户需求
用户: "测试60日动量因子"
↓
Agent 理解:
- 用户想做因子分析
- "60日动量" = 动量因子,周期60天

# 2. 激活金融知识 (LLM 预训练)
Agent 知道:
- 动量 = Momentum = 价格趋势
- 量化中常用: MOM($close, N)

# 3. 查阅文档
docs/AGENT_GUIDE.md:
"""
- MOM($close, N): N日动量
- Ref($close, N): N期前的值
"""

# 4. 构建表达式
Agent 推理: MOM($close, 60)

# 5. 调用 Skill
single_factor_analysis(
    factor_formula="MOM($close,60)",
    market="csi300",
    ...
)
```

---

## 🔑 关键原则

### 1. Skills = 能力接口,不是智能教学

**应该做**:
- ✅ 封装 step1-4 的复杂流程
- ✅ 统一输入输出格式
- ✅ 提供清晰的 API 文档
- ✅ 提供丰富的使用示例

**不应该做**:
- ❌ 教 Agent 如何理解自然语言
- ❌ 教 Agent 如何构建表达式
- ❌ 教 Agent 如何映射概念
- ❌ 维护规则库

---

### 2. Agent 本身就懂,不需要我们教

**Agent 的预训练能力**:
- ✅ 金融知识: "动量"、"反转"、"波动率"
- ✅ 编程能力: 知道怎么调用 Python 函数
- ✅ 推理能力: "60日" = 参数 N=60
- ✅ 学习能力: 从示例中举一反三

**我们只需要**:
- ✅ 告诉 Agent "有什么能力可用"
- ✅ 告诉 Agent "参数格式是什么"
- ✅ 提供示例展示 "怎么用"

---

### 3. 文档 + 示例 = 高准确率

**核心公式**:
```
准确率 = 文档清晰度 × 示例丰富度
```

**文档应该包含**:

#### (1) 能力清单
```markdown
## 可用 Skills

### single_factor_analysis
完整分析一个因子的表现

参数:
- factor_formula (str): Qlib表达式
- market (str): csi300/csi500/csi1000
- start_date (str): YYYY-MM-DD
- end_date (str): YYYY-MM-DD
- neutralize (bool): 是否中性化,默认True
```

#### (2) Qlib 语法说明
```markdown
## Qlib 表达式基础

基础字段:
- $close, $open, $high, $low
- $volume, $total_mv, $float_mv

常用函数:
- MOM($close, N): N日动量
- MA($close, N): N日均线
- Ref($field, N): N期前的值
- Std/Mean($field, N): 标准差/均值
```

#### (3) 自然语言映射示例
```markdown
## 自然语言 → Qlib表达式 示例

| 用户描述 | Qlib表达式 | 说明 |
|---------|-----------|------|
| "60日动量" | MOM($close,60) | 标准动量 |
| "20日反转" | -MOM($close,20) | 负动量 |
| "均线偏离度" | $close/MA($close,20)-1 | 相对位置 |
| "市值加权动量" | MOM($close,20)*Rank($total_mv) | 组合因子 |
```

#### (4) 完整对话示例
```markdown
## 对话示例

**用户**: 测试60日动量因子在沪深300最近3年的表现

**Agent推理**:
1. 因子: "60日动量" → MOM($close,60)
2. 市场: "沪深300" → csi300
3. 时间: "最近3年" → 2022-01-01 ~ 2024-12-31

**Agent调用**:
```python
single_factor_analysis(
    factor_formula="MOM($close,60)",
    market="csi300",
    start_date="2022-01-01",
    end_date="2024-12-31"
)
```

**Agent回复**:
✅ 60日动量因子分析完成...
```

---

## 🚀 三种 Agent 能力层次

| 维度 | 通用 LLM | RAG + 规则库 | 微调模型 |
|------|---------|-------------|---------|
| **实现成本** | ⭐ 低 | ⭐⭐ 中 | ⭐⭐⭐ 高 |
| **维护成本** | ⭐ 低 | ⭐⭐⭐ 高 | ⭐ 低 |
| **准确率** | ⭐⭐ 60-80% | ⭐⭐⭐⭐ 90-95% | ⭐⭐⭐⭐⭐ 95-99% |
| **泛化能力** | ⭐⭐⭐ 中 | ⭐⭐ 弱 | ⭐⭐⭐⭐⭐ 强 |

### 推荐路线: 渐进式进化

#### Phase 1: 信任 Agent (当前)
```python
# 用通用 LLM + 简洁文档
docs/AGENT_GUIDE.md  # 清晰的能力说明 + 丰富示例

# 让 Agent 自己推理
# 如果错了,用户会反馈,Agent 会学习
```

#### Phase 2: 收集反馈 (运行时)
```python
# 记录 Agent 的转换历史
conversion_history = [
    {
        "user_input": "60日动量",
        "agent_output": "MOM($close,60)",
        "user_feedback": "correct"
    },
    {
        "user_input": "市值加权动量",
        "agent_output": "MOM($close,20)*$total_mv",
        "user_feedback": "wrong,应该用Rank",
        "correct_output": "MOM($close,20)*Rank($total_mv)"
    },
]

# 这些数据就是未来的微调金矿
```

#### Phase 3: 按需微调 (未来)
```python
# 当积累了几千条真实数据后
# 或者发现 Agent 经常出错时

# 做一次轻量级 LoRA 微调
model = fine_tune_with_lora(
    base_model="deepseek-ai/DeepSeek-R1",
    training_data=conversion_history,  # 真实用户数据
    epochs=3
)

# 微调后的模型成为"Qlib专家"
```

---

## 📋 最终架构方案

### 极简文件结构

```bash
# 只需 3 个文件,零代码改动 step1-4

skills/
  └── __init__.py                 # 100行: 纯导出

parent_skills/
  └── single_factor_analysis.py   # 150行: 纯编排

docs/
  └── AGENT_GUIDE.md              # 300行: 文档 + 示例
```

### 文件职责

#### skills/__init__.py
```python
"""
Skills 统一导出

纯别名,零新逻辑
"""
from parent_skills.single_factor_analysis import single_factor_analysis

__all__ = ["single_factor_analysis"]
```

#### parent_skills/single_factor_analysis.py
```python
"""
父Skill: 单因子完整分析

纯编排,调用 step1-4
"""
def single_factor_analysis(factor_formula, market, start_date, end_date, neutralize=True):
    # Step 1: 因子提取
    from step1.因子提取与预处理 import extract_and_preprocess_factor
    result1 = extract_and_preprocess_factor(...)

    # Step 2: 因子中性化 (可选)
    if neutralize:
        from step2.因子中性化 import neutralize_factors
        result2 = neutralize_factors(...)

    # Step 3: 因子收益率
    from step3.因子收益率计算 import calculate_factor_returns
    result3 = calculate_factor_returns(...)

    # Step 4: 因子绩效评估
    from step4.因子绩效评估 import evaluate_factor_performance
    result4 = evaluate_factor_performance(...)

    # 汇总报告
    return generate_report(result1, result2, result3, result4)
```

#### docs/AGENT_GUIDE.md
```markdown
# Agent Skills 完全指南

## 1. 可用 Skills
## 2. Qlib 表达式语法
## 3. 自然语言映射示例表
## 4. 完整对话示例
## 5. 错误处理最佳实践
```

---

## 🎯 核心结论

### 终极领悟

> "到最后,其实是通过不断补充完善文档,以及示例,来实现高准确率的目标。"

### Agent 系统的正确打开方式

- **不是教 Agent 怎么思考** (那是徒劳的)
- **而是提供清晰的说明书和示例** (让 Agent 自己学)

### 文档越详细,示例越丰富,Agent 就越聪明!

---

## 📖 相关文档

- [完整架构设计](.claude/plans/linked-hatching-glaze.md)
- [Skills 实现方案](.claude/plans/linked-hatching-glaze.md)
- [Step 功能说明](step1/README.md, step2/README.md, step3/README.md, step4/README.md)

---

## 🤔 关键决策: 是否需要 Skills 改造

### 决策结论: **暂时不要改造**

经过深入讨论,达成共识:**当前阶段不需要进行 Skills 改造**

---

### 为什么现在不需要?

#### **1. 当前阶段: 验证需求优先**

```python
# 现状
✅ 有完整的 Step1-4 流程
✅ 可以通过 CLI 手动执行
❓ 还没有真实的 Agent 使用场景
❓ 还不清楚 Agent 会如何调用这些能力

# 如果现在改造
- 可能过度设计
- 可能不符合实际使用需求
- 维护成本增加,但价值未知
```

#### **2. Agent 未必需要 Skills**

**核心洞察**: Agent 可以直接调用现有的 Step 函数!

```python
# Agent 完全可以直接这样调用

from step1.因子提取与预处理 import extract_and_preprocess_factor
from step2.因子中性化 import neutralize_factors
from step3.因子收益率计算 import calculate_factor_returns
from step4.因子绩效评估 import evaluate_factor_performance

# Agent 自己编排
result1 = extract_and_preprocess_factor(...)
result2 = neutralize_factors(...)
result3 = calculate_factor_returns(...)
result4 = evaluate_factor_performance(...)

# 不需要 Skills 层!
```

#### **3. Skills 的价值尚未体现**

| 场景 | 是否需要 Skills |
|------|---------------|
| 单次分析 | ❌ 直接调用 Step 更简单 |
| 固定流程 | ❌ 写个脚本就好 |
| 灵活编排 | ⚠️ Agent 可以直接调用 |
| 多人协作 | ✅ **这才是 Skills 的价值** |
| API 暴露 | ✅ **这才是 Skills 的价值** |

---

### 分阶段决策框架

#### **Phase 1: 先让 Agent 直接调用 (当前)**

```python
# 不改任何代码
# 让 Agent 直接导入 Step 函数使用

# 如果 Agent 需要封装,它会自己创建包装函数
# 等真实使用后,再根据需求重构
```

**优点**:
- ✅ 零改造成本
- ✅ 快速验证 Agent 场景
- ✅ 保持灵活性

---

#### **Phase 2: 根据使用反馈决定 (未来)**

**什么时候需要改造?**

**情况A: Agent 经常需要固定流程**
```python
# 如果发现 Agent 总是这样调用:
for factor in factors:
    extract_and_preprocess_factor(factor)
    neutralize_factors(factor)
    calculate_factor_returns(factor)
    evaluate_factor_performance(factor)

# → 创建父Skill有价值
```

**情况B: 需要暴露给其他系统**
```python
# 如果需要:
- 提供 REST API
- 提供给其他开发者使用
- 作为标准能力接口

# → 创建 Skills 层有价值
```

**情况C: 团队协作需要**
```python
# 如果团队有多人:
- 有人写 Agent
- 有人维护分析逻辑
- 需要清晰的接口契约

# → 创建 Skills 层有价值
```

---

### 决策树

```
Q1: 你现在有真实的 Agent 在用吗?
   → 没有 → ❌ 暂不改造

Q2: 你是否已经尝试让 Agent 直接调用 Step 函数?
   → 没有 → ✅ 先尝试直接调用
   → 是,但很麻烦 → 进入Q3

Q3: 遇到什么问题?
   → 参数传递繁琐 → 创建父Skill
   → 需要给他人使用 → 创建 Skills 层
   → 需要暴露 API → 创建 Skills 层
   → 其他问题 → 具体分析
```

---

### 对比分析

| 维度 | 不改造 | 改造 |
|------|-------|------|
| **代码量** | 0行 | ~400行 |
| **维护成本** | 低 | 中 |
| **灵活性** | 高 | 中 |
| **易用性** | 低 | 高 |
| **适用场景** | 个人开发 | 团队协作/API |

---

### 最终建议

#### **现在: 做好准备,但不动手**

```bash
# 只做一件事: 写好文档
docs/AGENT_INTEGRATION.md

"""
## Agent 如何使用本项目的分析能力

### 直接调用方式

Agent 可以直接导入 Step 函数:

\`\`\`python
from step1.因子提取与预处理 import extract_and_preprocess_factor
from step2.因子中性化 import neutralize_factors
# ...

# 完整分析流程
result1 = extract_and_preprocess_factor(
    market="csi300",
    start_date="2022-01-01",
    end_date="2024-12-31",
    factor_formulas=["MOM($close,60)"],
    provider_uri="~/data/qlib_data"
)

# ... 继续调用 Step2-4
\`\`\`

### 参数说明
...

### 示例对话
...
"""

# 只需这份文档,不需要任何代码改动!
```

#### **未来: 根据需求决定**

```python
# 如果未来发现:
- "Agent 总是重复调用这4个Step"
  → 创建 single_factor_analysis 父Skill

- "需要提供 REST API 给其他系统"
  → 创建完整的 Skills 层

- "团队成员需要标准接口"
  → 创建 Skills 层 + 文档

# 否则:
# 保持现状,让 Agent 直接调用 Step 函数
```

---

### 核心原则

> **"不要为了 Agent 而 Agent,等真实需求出现再设计"**

---

## 📚 总结: 本文档的核心价值

本文档记录了完整的思考演进过程:

1. **初始想法**: 将 Step1-4 拆分为 Skills
2. **深入讨论**: 发现过度工程化的风险
3. **关键领悟**: Agent 可以直接调用现有函数
4. **最终决策**: 暂不改造,先验证需求

这个过程本身就是一笔宝贵的财富,为未来的架构演进提供了清晰的思路。

---

**文档版本**: v1.0
**更新日期**: 2025-01-13
**维护策略**: 根据 Agent 使用反馈持续更新文档和示例