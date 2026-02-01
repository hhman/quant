# Coding Agent 工作准则

## 强制规则（最高优先级）

**以下规则对 Agent 行为具有强制约束力，优先级高于本页其他内容：**

- [编码规则（强制）](rules/CODING_RULES.md) - Agent 编码与重构行为约束规范
- [文档编写规则（强制）](rules/DOC_RULES.md) - Agent 撰写文档、设计说明、分析报告时的输出约束

---

## 工作方式

1. **语言**: 中文回复
2. **文件管理**: 不要随意生成文件，如需创建，必须先获得用户许可
3. **工作流程**: 先分析、规划、确认，再执行；主动扫描引用关系，不要询问

## 环境配置

1. **依赖环境**: `conda quant` 环境包含了项目的依赖
   - Conda 路径: `/opt/homebrew/anaconda3`
   - 初始化脚本: `/opt/homebrew/anaconda3/etc/profile.d/conda.sh`
   - 激活命令: `source /opt/homebrew/anaconda3/etc/profile.d/conda.sh && conda activate quant`
2. **代码质量**: 项目使用 pre-commit 机制，在代码提交前会自动进行 Ruff 检查

## 项目说明

这是一个量化金融项目。

具体使用方法见各模块 README 文档。
