#!/usr/bin/env python3
"""
Excel 转 CSV 脚本（使用 xlsx2csv Python API）

xlsx2csv 比 openpyxl 快 10-50 倍，适合处理大文件。

用法：
    # 转换所有 Sheet（每个 Sheet 一个 CSV）
    uv run python scripts/pipeline/excel_to_csv.py \
        --input ./data/large.xlsx \
        --output-dir ./data/csv

    # 合并所有 Sheet 到单个 CSV
    uv run python scripts/pipeline/excel_to_csv.py \
        --input ./data/large.xlsx \
        --output ./data/merged.csv \
        --merge-sheets
"""

from __future__ import annotations

import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import List, Optional

# 强制禁用输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'


def log(msg: str = ""):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def get_excel_sheets(excel_path: Path) -> List[str]:
    """获取 Excel 所有 Sheet 名称"""
    import openpyxl
    wb = openpyxl.load_workbook(str(excel_path), read_only=True)
    sheets = wb.sheetnames
    wb.close()
    return sheets


def excel_sheet_to_csv(
    excel_path: Path,
    output_path: Path,
    sheet_name: str,
) -> bool:
    """
    使用 xlsx2csv 转换单个 Sheet

    Args:
        excel_path: Excel 文件路径
        output_path: 输出 CSV 路径
        sheet_name: Sheet 名称

    Returns:
        是否成功
    """
    from xlsx2csv import Xlsx2csv

    try:
        # 使用 StringIO 捕获输出
        output = StringIO()

        converter = Xlsx2csv(str(excel_path), outputfile=output)
        converter.convert(sheet=sheet_name)

        # 写入文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output.getvalue(), encoding='utf-8')

        return True

    except Exception as e:
        log(f"  错误: {e}")
        return False


def excel_to_csv(
    excel_path: Path,
    output_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    merge_sheets: bool = False,
    exclude_patterns: List[str] = None,
) -> List[Path]:
    """
    转换 Excel 到 CSV

    Args:
        excel_path: Excel 文件路径
        output_path: 单文件输出路径
        output_dir: 多文件输出目录
        merge_sheets: 是否合并所有 Sheet
        exclude_patterns: 排除的 Sheet 名称模式

    Returns:
        生成的 CSV 文件列表
    """
    exclude_patterns = exclude_patterns or ["query", "summary"]
    start_time = time.time()

    log("=" * 50)
    log("Excel 转 CSV（xlsx2csv）")
    log("=" * 50)
    log(f"输入文件: {excel_path}")
    log(f"文件大小: {excel_path.stat().st_size / 1024 / 1024:.1f} MB")
    log("")

    # 获取 Sheet 列表
    all_sheets = get_excel_sheets(excel_path)
    log(f"Sheet 列表: {all_sheets}")

    # 过滤 Sheet
    data_sheets = [s for s in all_sheets
                   if not any(p.lower() in s.lower() for p in exclude_patterns)]
    log(f"待处理 Sheet: {data_sheets}")
    log("")

    output_files = []
    temp_dir = excel_path.parent / ".temp_csv"

    if merge_sheets and len(data_sheets) > 1:
        # 合并模式：逐个转换后合并
        log("模式: 合并所有 Sheet 到单个 CSV")
        log("-" * 40)

        temp_dir.mkdir(exist_ok=True)
        temp_files = []

        for i, sheet_name in enumerate(data_sheets):
            log(f"[Sheet {i+1}/{len(data_sheets)}] {sheet_name}")

            temp_csv = temp_dir / f"sheet_{i}.csv"
            sheet_start = time.time()

            success = excel_sheet_to_csv(excel_path, temp_csv, sheet_name)

            if success:
                rows = sum(1 for _ in open(temp_csv))
                size_mb = temp_csv.stat().st_size / 1024 / 1024
                elapsed = time.time() - sheet_start
                log(f"  完成: {rows:,} 行, {size_mb:.1f} MB, {elapsed:.1f}s")
                temp_files.append(temp_csv)
            else:
                log(f"  跳过")

        # 合并 CSV
        if temp_files:
            log("")
            log("合并 CSV 文件...")

            output_path = output_path or excel_path.with_suffix(".csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as outfile:
                header_written = False
                for temp_csv in temp_files:
                    with open(temp_csv, 'r', encoding='utf-8') as infile:
                        header = infile.readline()
                        if not header_written:
                            outfile.write(header)
                            header_written = True
                        for line in infile:
                            outfile.write(line)

                    temp_csv.unlink()

            temp_dir.rmdir()
            output_files.append(output_path)
            log(f"合并完成: {output_path}")

    else:
        # 分离模式：每个 Sheet 一个 CSV
        output_dir = output_dir or excel_path.parent

        for i, sheet_name in enumerate(data_sheets):
            log(f"[Sheet {i+1}/{len(data_sheets)}] {sheet_name}")

            safe_name = sheet_name.replace(" ", "_").replace("/", "_")
            csv_path = output_dir / f"{excel_path.stem}_{safe_name}.csv"

            sheet_start = time.time()
            success = excel_sheet_to_csv(excel_path, csv_path, sheet_name)

            if success:
                rows = sum(1 for _ in open(csv_path))
                size_mb = csv_path.stat().st_size / 1024 / 1024
                elapsed = time.time() - sheet_start
                log(f"  完成: {rows:,} 行, {size_mb:.1f} MB, {elapsed:.1f}s")
                output_files.append(csv_path)
            else:
                log(f"  失败")

    # 摘要
    elapsed = time.time() - start_time
    log("")
    log("=" * 50)
    log("转换完成")
    log("=" * 50)

    total_size = sum(f.stat().st_size for f in output_files) / 1024 / 1024
    log(f"输出文件: {len(output_files)} 个, 共 {total_size:.1f} MB")
    for f in output_files:
        log(f"  - {f.name}")
    log(f"总耗时: {elapsed:.1f} 秒")
    log("=" * 50)

    return output_files


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Excel 转 CSV（使用 xlsx2csv，快速）",
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Excel 输入文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出 CSV 路径（合并模式）"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（分离模式）"
    )
    parser.add_argument(
        "--merge-sheets",
        action="store_true",
        help="合并所有 Sheet 到单个 CSV"
    )
    parser.add_argument(
        "--exclude-sheets",
        nargs="+",
        default=["query", "summary"],
        help="排除的 Sheet 名称模式"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        log(f"❌ 输入文件不存在: {input_path}")
        return 1

    try:
        files = excel_to_csv(
            excel_path=input_path,
            output_path=Path(args.output) if args.output else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            merge_sheets=args.merge_sheets,
            exclude_patterns=args.exclude_sheets,
        )
        return 0 if files else 1

    except Exception as e:
        log(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())