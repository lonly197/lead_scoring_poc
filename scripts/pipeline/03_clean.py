#!/usr/bin/env python3
"""
数据清洗脚本

功能：
- 基础清洗：删除高缺失列、重复行、常量列
- 高级清洗：异常值检测（IQR/Z-Score）、偏斜分布处理、高基数处理
- 生成清洗报告

用法：
    uv run python scripts/pipeline/03_clean.py \\
        --input ./data/merged.parquet \\
        --output ./data/cleaned.parquet \\
        --report ./reports/clean_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.utils import load_data, save_data, print_step


class DataCleaner:
    """数据清洗器"""

    def __init__(
        self,
        high_missing_threshold: float = 0.5,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        skew_threshold: float = 1.0,
        high_cardinality_threshold: int = 100,
    ):
        """
        初始化清洗器

        Args:
            high_missing_threshold: 高缺失阈值（默认 0.5，即 50%）
            outlier_method: 异常值检测方法（iqr/zscore）
            outlier_threshold: 异常值阈值（IQR 乘数或 Z-Score 阈值）
            skew_threshold: 偏斜阈值（默认 1.0）
            high_cardinality_threshold: 高基数阈值（默认 100）
        """
        self.high_missing_threshold = high_missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.skew_threshold = skew_threshold
        self.high_cardinality_threshold = high_cardinality_threshold

        # 清洗日志
        self.clean_log: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "dropped_columns": [],
            "dropped_rows": 0,
            "outliers": {},
            "skewed_columns": [],
            "high_cardinality_columns": [],
        }

    def clean(
        self,
        df: "pl.DataFrame",
        drop_high_missing: bool = True,
        drop_duplicates: bool = True,
        drop_constant: bool = True,
        detect_outliers: bool = True,
        handle_skewed: bool = True,
    ) -> "pl.DataFrame":
        """
        执行数据清洗

        Args:
            df: 输入 DataFrame
            drop_high_missing: 是否删除高缺失列
            drop_duplicates: 是否删除重复行
            drop_constant: 是否删除常量列
            detect_outliers: 是否检测异常值
            handle_skewed: 是否处理偏斜分布

        Returns:
            清洗后的 DataFrame
        """
        import polars as pl

        original_rows = len(df)
        original_cols = len(df.columns)

        # 1. 删除高缺失列
        if drop_high_missing:
            df = self._drop_high_missing(df)

        # 2. 删除常量列
        if drop_constant:
            df = self._drop_constant_columns(df)

        # 3. 删除重复行
        if drop_duplicates:
            df = self._drop_duplicates(df)

        # 4. 检测异常值（仅记录，不删除）
        if detect_outliers:
            self._detect_outliers(df)

        # 5. 检测偏斜分布
        if handle_skewed:
            self._detect_skewed(df)

        # 6. 检测高基数列
        self._detect_high_cardinality(df)

        # 记录摘要
        self.clean_log["summary"] = {
            "original_rows": original_rows,
            "original_cols": original_cols,
            "final_rows": len(df),
            "final_cols": len(df.columns),
            "dropped_rows": original_rows - len(df),
            "dropped_cols": original_cols - len(df.columns),
        }

        return df

    def _drop_high_missing(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """删除高缺失列"""
        import polars as pl

        missing_ratio = df.null_count() / len(df)
        high_missing_cols = [
            col for col in df.columns
            if missing_ratio.select(col).item() > self.high_missing_threshold
        ]

        if high_missing_cols:
            self.clean_log["dropped_columns"].extend([
                {"column": col, "reason": "high_missing"} for col in high_missing_cols
            ])
            df = df.drop(high_missing_cols)
            print_step("删除高缺失列", "success", f"删除 {len(high_missing_cols)} 列")

        return df

    def _drop_constant_columns(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """删除常量列"""
        import polars as pl

        constant_cols = []
        for col in df.columns:
            unique_count = df.select(pl.col(col).n_unique()).item()
            if unique_count <= 1:
                constant_cols.append(col)

        if constant_cols:
            self.clean_log["dropped_columns"].extend([
                {"column": col, "reason": "constant"} for col in constant_cols
            ])
            df = df.drop(constant_cols)
            print_step("删除常量列", "success", f"删除 {len(constant_cols)} 列")

        return df

    def _drop_duplicates(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """删除重复行"""
        import polars as pl

        before = len(df)
        df = df.unique()
        dropped = before - len(df)

        if dropped > 0:
            self.clean_log["dropped_rows"] = dropped
            print_step("删除重复行", "success", f"删除 {dropped:,} 行")

        return df

    def _detect_outliers(self, df: "pl.DataFrame") -> Dict[str, int]:
        """
        检测数值列的异常值

        使用 IQR 或 Z-Score 方法检测异常值。
        注意：仅记录异常值数量，不删除数据。
        """
        import polars as pl
        import numpy as np

        outliers = {}

        for col in df.columns:
            # 只处理数值列
            if df[col].dtype not in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                continue

            data = df[col].drop_nulls().to_numpy()

            if len(data) == 0:
                continue

            if self.outlier_method == "iqr":
                # IQR 方法
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
                outlier_count = np.sum((data < lower) | (data > upper))

            else:  # zscore
                # Z-Score 方法
                mean = np.mean(data)
                std = np.std(data)
                if std == 0:
                    continue
                z_scores = np.abs((data - mean) / std)
                outlier_count = np.sum(z_scores > self.outlier_threshold)

            if outlier_count > 0:
                outliers[col] = int(outlier_count)

        self.clean_log["outliers"] = outliers

        if outliers:
            total = sum(outliers.values())
            print_step("检测异常值", "success", f"发现 {len(outliers)} 列共 {total:,} 个异常值")

        return outliers

    def _detect_skewed(self, df: "pl.DataFrame") -> List[str]:
        """
        检测偏斜分布

        计算数值列的偏度，记录超过阈值的列。
        注意：不进行变换，仅记录建议。
        """
        import polars as pl
        import numpy as np
        from scipy import stats

        skewed_cols = []

        for col in df.columns:
            if df[col].dtype not in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                continue

            data = df[col].drop_nulls().to_numpy()

            if len(data) < 10:
                continue

            # 计算偏度
            skewness = stats.skew(data)

            if abs(skewness) > self.skew_threshold:
                skewed_cols.append({
                    "column": col,
                    "skewness": round(skewness, 2),
                    "suggestion": "log" if skewness > 0 else "power",
                })

        self.clean_log["skewed_columns"] = skewed_cols

        if skewed_cols:
            print_step("检测偏斜分布", "success", f"发现 {len(skewed_cols)} 列偏斜")

        return skewed_cols

    def _detect_high_cardinality(self, df: "pl.DataFrame") -> List[str]:
        """
        检测高基数类别列

        唯一值数量超过阈值的字符串列。
        注意：不进行编码，仅记录建议。
        """
        import polars as pl

        high_card_cols = []

        for col in df.columns:
            if df[col].dtype != pl.Utf8:
                continue

            unique_count = df.select(pl.col(col).n_unique()).item()

            if unique_count > self.high_cardinality_threshold:
                high_card_cols.append({
                    "column": col,
                    "unique_values": unique_count,
                    "suggestion": "frequency_encoding or target_encoding",
                })

        self.clean_log["high_cardinality_columns"] = high_card_cols

        if high_card_cols:
            print_step("检测高基数列", "success", f"发现 {len(high_card_cols)} 列")

        return high_card_cols

    def generate_report(self, output_path: Path) -> None:
        """
        生成清洗报告（Markdown 格式）

        Args:
            output_path: 报告输出路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # 标题
        lines.append("# 数据清洗报告")
        lines.append(f"\n生成时间: {self.clean_log['timestamp']}")

        # 摘要
        summary = self.clean_log.get("summary", {})
        lines.append("\n## 清洗摘要")
        lines.append(f"| 指标 | 清洗前 | 清洗后 | 变化 |")
        lines.append("|------|--------|--------|------|")
        lines.append(f"| 行数 | {summary.get('original_rows', 0):,} | {summary.get('final_rows', 0):,} | -{summary.get('dropped_rows', 0):,} |")
        lines.append(f"| 列数 | {summary.get('original_cols', 0)} | {summary.get('final_cols', 0)} | -{summary.get('dropped_cols', 0)} |")

        # 删除的列
        dropped_cols = self.clean_log.get("dropped_columns", [])
        if dropped_cols:
            lines.append("\n## 删除的列")
            lines.append("| 列名 | 删除原因 |")
            lines.append("|------|----------|")
            for item in dropped_cols:
                lines.append(f"| {item['column']} | {item['reason']} |")

        # 异常值
        outliers = self.clean_log.get("outliers", {})
        if outliers:
            lines.append("\n## 异常值检测")
            lines.append(f"方法: {self.outlier_method.upper()}")
            lines.append("\n| 列名 | 异常值数量 |")
            lines.append("|------|-----------|")
            for col, count in sorted(outliers.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {col} | {count:,} |")

        # 偏斜分布
        skewed = self.clean_log.get("skewed_columns", [])
        if skewed:
            lines.append("\n## 偏斜分布")
            lines.append(f"偏度阈值: {self.skew_threshold}")
            lines.append("\n| 列名 | 偏度 | 建议变换 |")
            lines.append("|------|------|----------|")
            for item in sorted(skewed, key=lambda x: abs(x["skewness"]), reverse=True):
                lines.append(f"| {item['column']} | {item['skewness']} | {item['suggestion']} |")

        # 高基数列
        high_card = self.clean_log.get("high_cardinality_columns", [])
        if high_card:
            lines.append("\n## 高基数列")
            lines.append(f"基数阈值: {self.high_cardinality_threshold}")
            lines.append("\n| 列名 | 唯一值数 | 建议编码 |")
            lines.append("|------|----------|----------|")
            for item in sorted(high_card, key=lambda x: x["unique_values"], reverse=True):
                lines.append(f"| {item['column']} | {item['unique_values']:,} | {item['suggestion']} |")

        # 建议
        lines.append("\n## 后续建议")
        if outliers:
            lines.append("1. 检查异常值是否为真实数据，必要时进行 Winsorize 处理")
        if skewed:
            lines.append("2. 对偏斜分布进行 Log 或 Box-Cox 变换，提升模型性能")
        if high_card:
            lines.append("3. 对高基数列使用频率编码或目标编码，避免维度爆炸")

        lines.append("\n---")
        lines.append("*报告由数据清洗脚本自动生成*")

        # 写入文件
        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")

        print(f"清洗报告已保存: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据清洗脚本 - 基础清洗 + 高级清洗",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    uv run python scripts/pipeline/03_clean.py \\
        --input ./data/merged.parquet \\
        --output ./data/cleaned.parquet

    # 生成清洗报告
    uv run python scripts/pipeline/03_clean.py \\
        --input ./data/merged.parquet \\
        --output ./data/cleaned.parquet \\
        --report ./reports/clean_report.md

    # 自定义阈值
    uv run python scripts/pipeline/03_clean.py \\
        --input ./data/merged.parquet \\
        --output ./data/cleaned.parquet \\
        --high-missing-threshold 0.3 \\
        --outlier-method zscore
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入数据文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--report", "-r",
        default=None,
        help="清洗报告输出路径（Markdown 格式）"
    )

    # 清洗选项
    parser.add_argument(
        "--drop-high-missing",
        action="store_true",
        default=True,
        help="删除高缺失列（默认: True）"
    )
    parser.add_argument(
        "--no-drop-high-missing",
        action="store_false",
        dest="drop_high_missing",
        help="保留高缺失列"
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        default=True,
        help="删除重复行（默认: True）"
    )
    parser.add_argument(
        "--no-drop-duplicates",
        action="store_false",
        dest="drop_duplicates",
        help="保留重复行"
    )
    parser.add_argument(
        "--high-missing-threshold",
        type=float,
        default=0.5,
        help="高缺失阈值（默认: 0.5，即 50%%）"
    )

    # 高级清洗选项
    parser.add_argument(
        "--detect-outliers",
        action="store_true",
        default=True,
        help="检测异常值（默认: True）"
    )
    parser.add_argument(
        "--no-detect-outliers",
        action="store_false",
        dest="detect_outliers",
        help="跳过异常值检测"
    )
    parser.add_argument(
        "--outlier-method",
        choices=["iqr", "zscore"],
        default="iqr",
        help="异常值检测方法（默认: iqr）"
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=1.5,
        help="异常值阈值（默认: 1.5，IQR 乘数）"
    )
    parser.add_argument(
        "--handle-skewed",
        action="store_true",
        default=True,
        help="检测偏斜分布（默认: True）"
    )
    parser.add_argument(
        "--no-handle-skewed",
        action="store_false",
        dest="handle_skewed",
        help="跳过偏斜分布检测"
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=1.0,
        help="偏斜阈值（默认: 1.0）"
    )
    parser.add_argument(
        "--high-cardinality-threshold",
        type=int,
        default=100,
        help="高基数阈值（默认: 100）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else None

    # 检查输入文件
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return 1

    try:
        import polars as pl

        print("=" * 60)
        print("数据清洗脚本")
        print("=" * 60)

        # 加载数据
        print_step("加载数据", "running", str(input_path))
        df = load_data(input_path, engine="polars")
        print_step("加载数据", "success", f"{len(df):,} 行, {len(df.columns)} 列")

        # 创建清洗器
        cleaner = DataCleaner(
            high_missing_threshold=args.high_missing_threshold,
            outlier_method=args.outlier_method,
            outlier_threshold=args.outlier_threshold,
            skew_threshold=args.skew_threshold,
            high_cardinality_threshold=args.high_cardinality_threshold,
        )

        # 执行清洗
        print_step("执行清洗", "running")
        df_cleaned = cleaner.clean(
            df=df,
            drop_high_missing=args.drop_high_missing,
            drop_duplicates=args.drop_duplicates,
            drop_constant=True,
            detect_outliers=args.detect_outliers,
            handle_skewed=args.handle_skewed,
        )
        print_step("执行清洗", "success")

        # 保存结果
        print_step("保存结果", "running", str(output_path))
        save_data(df_cleaned, output_path)
        print_step("保存结果", "success")

        # 生成报告
        if report_path:
            cleaner.generate_report(report_path)

        # 打印摘要
        summary = cleaner.clean_log.get("summary", {})
        print("\n" + "=" * 60)
        print("清洗完成")
        print("=" * 60)
        print(f"输出文件: {output_path}")
        print(f"数据量: {summary.get('final_rows', 0):,} 行, {summary.get('final_cols', 0)} 列")
        print(f"删除行数: {summary.get('dropped_rows', 0):,}")
        print(f"删除列数: {summary.get('dropped_cols', 0)}")

        if report_path:
            print(f"清洗报告: {report_path}")

        print("=" * 60)

        # 释放内存
        del df, df_cleaned, cleaner

        return 0

    except Exception as e:
        print(f"❌ 清洗失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())