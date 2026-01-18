"""
因子挖掘 CLI 模块

本模块提供命令行接口用于执行 Gplearn 遗传算法因子挖掘。

使用示例：
    # 基础用法
    python -m factor_mining.gplearn_mining \\
        --market csi300 \\
        --start-date 2023-01-01 \\
        --end-date 2024-12-31 \\
        --features $close $volume $total_mv

    # 自定义参数
    python -m factor_mining.gplearn_mining \\
        --market csi300 \\
        --start-date 2023-01-01 \\
        --end-date 2024-12-31 \\
        --features $close $volume $total_mv \\
        --window-size 20 \\
        --population-size 2000 \\
        --generations 30 \\
        --n-components 30
"""

__all__ = ["main"]

from .gplearn_mining import main
