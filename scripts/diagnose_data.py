#!/usr/bin/env python3
"""
数据格式诊断脚本

检查数据文件的实际格式，帮助调试适配问题。
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def diagnose(file_path: str):
    """诊断数据文件格式"""
    print(f"\n{'='*60}")
    print(f"诊断文件: {file_path}")
    print(f"{'='*60}\n")

    # 1. 检查文件基本信息
    path = Path(file_path)
    print(f"文件大小: {path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"文件扩展名: {path.suffix}")

    # 2. 检查分隔符
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    tab_count = first_line.count('\t')
    comma_count = first_line.count(',')

    print(f"\n首行分析:")
    print(f"  - Tab 数量: {tab_count}")
    print(f"  - 逗号数量: {comma_count}")
    print(f"  - 首行长度: {len(first_line)} 字符")

    # 3. 确定分隔符
    if tab_count > comma_count:
        sep = '\t'
        print(f"  - 检测分隔符: Tab (\\t)")
    else:
        sep = ','
        print(f"  - 检测分隔符: 逗号 (,)")

    # 4. 检查列数
    columns = first_line.strip().split(sep)
    print(f"\n列数分析:")
    print(f"  - 实际列数: {len(columns)}")

    # 5. 检查是否有表头
    first_field = columns[0] if columns else ""
    has_header = any('\u4e00' <= c <= '\u9fff' for c in first_field)  # 包含中文
    print(f"\n表头检测:")
    print(f"  - 首个字段: {first_field[:50]}...")
    print(f"  - 是否有中文表头: {has_header}")

    # 6. 如果无表头，显示前几个字段
    if not has_header:
        print(f"\n前10个字段值示例:")
        for i, col in enumerate(columns[:10]):
            print(f"  列{i}: {col[:30]}{'...' if len(col) > 30 else ''}")

    # 7. 尝试用适配器加载
    print(f"\n{'='*60}")
    print("尝试用适配器加载...")
    print(f"{'='*60}\n")

    try:
        from src.data.loader import DataLoader

        loader = DataLoader(file_path, auto_adapt=True)
        df = loader.load()

        print(f"✅ 加载成功!")
        print(f"  - 数据量: {len(df):,} 行")
        print(f"  - 列数: {len(df.columns)}")
        print(f"\n关键列检查:")
        key_columns = ["线索创建时间", "线索唯一ID", "到店时间", "试驾时间", "线索评级结果"]
        for col in key_columns:
            exists = "✅" if col in df.columns else "❌"
            print(f"  {exists} {col}")

        print(f"\n前10列名:")
        for i, col in enumerate(df.columns[:10]):
            print(f"  {i}: {col}")

        if "线索创建时间" in df.columns:
            print(f"\n线索创建时间示例:")
            print(f"  {df['线索创建时间'].head(3).tolist()}")

        # 检查 OHAB 相关列
        print(f"\n{'='*60}")
        print("OHAB 评级相关检查")
        print(f"{'='*60}")

        # 检查线索评级结果
        if "线索评级结果" in df.columns:
            print(f"\n线索评级结果 分布:")
            print(df["线索评级结果"].value_counts(dropna=False))
            print(f"\n线索评级结果 前5个值:")
            print(df["线索评级结果"].head(5).tolist())
        else:
            print("❌ 线索评级结果 列不存在")

        # 检查线索评级_试驾前
        if "线索评级_试驾前" in df.columns:
            print(f"\n线索评级_试驾前 分布:")
            print(df["线索评级_试驾前"].value_counts(dropna=False))
        else:
            print("❌ 线索评级_试驾前 列不存在")

        # 显示所有列名（帮助调试列名映射）
        print(f"\n{'='*60}")
        print("所有列名和样本值（用于调试列名映射）")
        print(f"{'='*60}")
        for i, col in enumerate(df.columns):
            # 显示前3个非空值
            sample_vals = df[col].dropna().head(3).tolist()
            # 截断过长的值
            sample_str = str(sample_vals)[:80]
            print(f"{i:2d}. {col}: {sample_str}")

        # 特别检查第 25-35 列（线索评级结果应该在这个范围）
        print(f"\n{'='*60}")
        print("关键列位置检查（列 25-35）")
        print(f"{'='*60}")
        for i in range(25, min(36, len(df.columns))):
            col = df.columns[i]
            sample_vals = df[col].dropna().head(5).tolist()
            unique_count = df[col].nunique()
            print(f"{i:2d}. {col}")
            print(f"    样本: {sample_vals}")
            print(f"    唯一值数: {unique_count}")
            if unique_count <= 20:
                print(f"    分布: {df[col].value_counts().to_dict()}")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="诊断数据格式")
    parser.add_argument("file", help="数据文件路径")
    args = parser.parse_args()

    diagnose(args.file)