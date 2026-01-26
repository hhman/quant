# Coding Agent 工作准则

## 工作方式

1. **语言**: 中文回复
2. **文件管理**: 🚫 不要随意生成文件，如需创建，必须先获得用户许可
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
