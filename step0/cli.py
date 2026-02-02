#!/usr/bin/env python3
"""
Step 0 CLI: 数据预处理 - 从原始CSV到Qlib格式
独立的 CLI 入口脚本
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cli_args import parse_date_range


def parse_args() -> argparse.Namespace:
    """
    定义并解析 CLI 参数

    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="Step 0: 数据预处理 - 从原始CSV到Qlib格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python step0/cli.py --start-date 2020-01-01 --end-date 2023-12-31
        """,
    )

    parser.add_argument(
        "--start-date", type=str, required=True, help="起始日期 (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=str, required=True, help="结束日期 (YYYY-MM-DD)"
    )

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> dict:
    """
    校验和标准化参数

    Args:
        args: 解析后的 CLI 参数

    Returns:
        标准化的参数字典

    Raises:
        ValueError: 当参数验证失败时
    """
    # 验证日期范围
    start_date, end_date = parse_date_range(args.start_date, args.end_date)

    return {
        "start_date": start_date,
        "end_date": end_date,
    }


def run_command(cmd: str, description: str = "") -> None:
    """
    运行 shell 命令，出错时退出程序

    Args:
        cmd: 要执行的命令字符串
        description: 命令描述（用于错误提示）

    Raises:
        SystemExit: 当命令执行失败时
    """
    try:
        subprocess.run(
            cmd,
            check=True,
            shell=True,
            capture_output=False,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {description}")
        sys.exit(e.returncode)


def validate_input_directories() -> None:
    """
    验证必要的输入目录是否存在

    Raises:
        SystemExit: 当任何必需目录不存在时
    """
    directories = {
        "stock": "股票CSV目录",
        "index": "指数CSV目录",
        "finance": "财务CSV目录",
    }

    for dir_name, dir_desc in directories.items():
        if not Path(dir_name).exists():
            print(f"{dir_desc}不存在: {dir_name}")
            sys.exit(1)


def main():
    args = parse_args()
    params = normalize_args(args)

    stock_dir = "stock"
    index_dir = "index"
    fin_dir = "finance"
    cache_dir = ".cache"
    output_dir = Path(cache_dir) / "step0_temp"
    qlib_dir = Path(cache_dir) / "qlib_data"
    qlib_src_dir = "qlib_src"

    validate_input_directories()

    from step0.daily_data_cleaning import process_stock_data

    process_stock_data(
        start_date=params["start_date"],
        end_date=params["end_date"],
        stock_dir=stock_dir,
        index_dir=index_dir,
        output_dir=str(output_dir),
    )

    if qlib_dir.exists():
        import shutil

        shutil.rmtree(qlib_dir)
        print(f"删除旧数据: {qlib_dir}")

    run_command(
        f'python "{qlib_src_dir}/scripts/dump_bin.py" dump_all '
        f'--data_path "{output_dir}" '
        f'--qlib_dir "{qlib_dir}" '
        f'--include_fields "open,high,low,close,volume,amount,industry,total_mv,float_mv,factor"',
        description="转换为Qlib二进制格式",
    )

    instruments_src = output_dir / "instruments"
    instruments_dst = qlib_dir / "instruments"

    instruments_dst.mkdir(parents=True, exist_ok=True)

    txt_files = list(instruments_src.glob("*.txt"))
    if not txt_files:
        print(f"未找到成分股文件: {instruments_src}/*.txt")
        sys.exit(1)

    for txt_file in txt_files:
        dst_file = instruments_dst / txt_file.name
        run_command(f'cp "{txt_file}" "{dst_file}"', description="复制membership文件")

    industry_mapping_src = output_dir / "industry_mapping.json"
    industry_mapping_dst = Path(cache_dir) / "industry_mapping.json"
    if industry_mapping_src.exists():
        run_command(
            f'cp "{industry_mapping_src}" "{industry_mapping_dst}"',
            description="复制industry_mapping文件",
        )

    from step0.financial_data_pivot import process_financial_data

    process_financial_data(
        start_date=params["start_date"],
        end_date=params["end_date"],
        finance_dir=fin_dir,
        output_dir=str(output_dir / "financial"),
    )

    run_command(
        f'python "{qlib_src_dir}/scripts/dump_pit.py" '
        f'--csv_path "{output_dir}/financial" '
        f'--qlib_dir "{qlib_dir}"',
        description="转换财务数据为Qlib格式",
    )

    run_command(
        f'python "{qlib_src_dir}/scripts/check_dump_bin.py" check '
        f'--qlib_dir "{qlib_dir}" '
        f'--csv_path "{output_dir}"',
        description="数据质量校验 (check_dump_bin)",
    )

    run_command(
        f'python "{qlib_src_dir}/scripts/check_data_health.py" check_data '
        f'--qlib_dir "{qlib_dir}"',
        description="数据质量校验 (check_data_health)",
    )

    from step0.returns_and_styles import calculate_returns_and_styles

    calculate_returns_and_styles(
        start_date=params["start_date"],
        end_date=params["end_date"],
        provider_uri=str(qlib_dir),
    )

    print("Step0完成!")


if __name__ == "__main__":
    main()
