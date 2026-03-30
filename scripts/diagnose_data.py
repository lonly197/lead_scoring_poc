#!/usr/bin/env python3
"""
数据探查诊断脚本（增强版）

功能：
1. 文件格式检测（分隔符、编码、表头）
2. 缺失值统计（按列、按行）
3. 数据类型分布
4. 目标变量分布检查
5. 数据清洗建议生成
6. 概览报告输出（控制台/Markdown）

用法：
    # 控制台输出
    uv run python scripts/diagnose_data.py ./data/线索宽表_完整.parquet

    # 生成 Markdown 报告
    uv run python scripts/diagnose_data.py ./data/线索宽表_完整.parquet --report ./reports/data_profile.md

    # 指定目标变量
    uv run python scripts/diagnose_data.py ./data/train.parquet --target 线索评级结果
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== 数据探查类 ====================

class DataProfiler:
    """数据探查器"""

    def __init__(self, file_path: str, target_column: Optional[str] = None):
        """
        初始化数据探查器

        Args:
            file_path: 数据文件路径
            target_column: 目标变量列名
        """
        self.file_path = Path(file_path)
        self.target_column = target_column
        self.df: Optional[pd.DataFrame] = None
        self.profile: Dict[str, Any] = {}

    def load(self) -> bool:
        """加载数据文件"""
        print(f"\n{'='*60}")
        print(f"数据探查: {self.file_path}")
        print(f"{'='*60}\n")

        if not self.file_path.exists():
            print(f"❌ 文件不存在: {self.file_path}")
            return False

        # 文件基本信息
        self.profile["file"] = {
            "path": str(self.file_path),
            "size_mb": self.file_path.stat().st_size / 1024 / 1024,
            "extension": self.file_path.suffix,
        }
        print(f"文件大小: {self.profile['file']['size_mb']:.2f} MB")
        print(f"文件格式: {self.profile['file']['extension']}")

        # 根据格式加载
        try:
            suffix = self.file_path.suffix.lower()
            if suffix == ".parquet":
                self.df = pd.read_parquet(self.file_path)
            elif suffix == ".csv":
                self.df = self._load_csv_with_detect()
            elif suffix == ".tsv":
                self.df = pd.read_csv(self.file_path, sep="\t", low_memory=False)
            else:
                # 尝试自动检测
                self.df = self._load_csv_with_detect()

            print(f"✅ 加载成功: {len(self.df):,} 行, {len(self.df.columns)} 列")
            return True

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

    def _load_csv_with_detect(self) -> pd.DataFrame:
        """自动检测分隔符并加载 CSV"""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()

        tab_count = first_line.count("\t")
        comma_count = first_line.count(",")

        sep = "\t" if tab_count > comma_count else ","
        print(f"检测分隔符: {'Tab' if sep == chr(9) else '逗号'}")

        return pd.read_csv(self.file_path, sep=sep, low_memory=False)

    def profile_basic(self) -> Dict[str, Any]:
        """基础信息探查"""
        if self.df is None:
            return {}

        print("\n--- 基础信息 ---")

        basic = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "memory_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            "duplicate_rows": self.df.duplicated().sum(),
        }

        self.profile["basic"] = basic

        print(f"数据量: {basic['rows']:,} 行 × {basic['columns']} 列")
        print(f"内存占用: {basic['memory_mb']:.2f} MB")
        print(f"重复行: {basic['duplicate_rows']:,} ({basic['duplicate_rows']/basic['rows']*100:.2f}%)")

        return basic

    def profile_missing(self) -> Dict[str, Any]:
        """缺失值统计"""
        if self.df is None:
            return {}

        print("\n--- 缺失值统计 ---")

        # 按列统计
        missing_by_col = self.df.isnull().sum()
        missing_ratio = (missing_by_col / len(self.df) * 100).round(2)

        missing_df = pd.DataFrame({
            "missing_count": missing_by_col,
            "missing_ratio": missing_ratio,
        }).sort_values("missing_ratio", ascending=False)

        # 高缺失列（>50%）
        high_missing = missing_df[missing_df["missing_ratio"] > 50]
        medium_missing = missing_df[(missing_df["missing_ratio"] > 10) & (missing_df["missing_ratio"] <= 50)]

        missing_stats = {
            "total_missing": int(missing_by_col.sum()),
            "total_cells": int(len(self.df) * len(self.df.columns)),
            "overall_missing_ratio": round(missing_by_col.sum() / (len(self.df) * len(self.df.columns)) * 100, 2),
            "high_missing_columns": len(high_missing),
            "medium_missing_columns": len(medium_missing),
            "columns": missing_df.to_dict("index"),
        }

        self.profile["missing"] = missing_stats

        print(f"总缺失值: {missing_stats['total_missing']:,} ({missing_stats['overall_missing_ratio']}%)")
        print(f"高缺失列(>50%): {missing_stats['high_missing_columns']} 个")
        print(f"中缺失列(10-50%): {missing_stats['medium_missing_columns']} 个")

        if len(high_missing) > 0:
            print(f"\n  高缺失列列表:")
            for col, row in high_missing.head(10).iterrows():
                print(f"    - {col}: {row['missing_ratio']:.1f}%")

        return missing_stats

    def profile_columns(self) -> Dict[str, Any]:
        """列类型和分布统计"""
        if self.df is None:
            return {}

        print("\n--- 列类型统计 ---")

        # 类型统计
        type_counts = self.df.dtypes.value_counts().to_dict()
        type_counts_str = {str(k): v for k, v in type_counts.items()}

        # 每列详情
        columns_info = {}
        for col in self.df.columns:
            col_data = self.df[col]
            info = {
                "dtype": str(col_data.dtype),
                "non_null": int(col_data.count()),
                "null": int(col_data.isnull().sum()),
                "unique": int(col_data.nunique()),
            }

            # 数值型统计
            if pd.api.types.is_numeric_dtype(col_data):
                info["type_category"] = "numeric"
                info["min"] = float(col_data.min()) if col_data.notna().any() else None
                info["max"] = float(col_data.max()) if col_data.notna().any() else None
                info["mean"] = float(col_data.mean()) if col_data.notna().any() else None

            # 类别型统计
            elif col_data.dtype == "object" or pd.api.types.is_string_dtype(col_data):
                info["type_category"] = "categorical"
                # 取前5个最常见值
                top_values = col_data.value_counts().head(5).to_dict()
                info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

            # 时间型
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                info["type_category"] = "datetime"
                if col_data.notna().any():
                    info["min_date"] = str(col_data.min())
                    info["max_date"] = str(col_data.max())

            else:
                info["type_category"] = "other"

            columns_info[col] = info

        col_stats = {
            "type_counts": type_counts_str,
            "columns": columns_info,
        }

        self.profile["columns"] = col_stats

        print(f"列类型分布: {type_counts_str}")

        # 低唯一值列（可能是常量或ID）
        low_unique = [col for col, info in columns_info.items() if info["unique"] <= 1]
        if low_unique:
            print(f"\n  常量列(唯一值<=1): {len(low_unique)} 个")
            for col in low_unique[:5]:
                print(f"    - {col}")

        return col_stats

    def profile_target(self) -> Dict[str, Any]:
        """目标变量分布检查"""
        if self.df is None:
            return {}

        print("\n--- 目标变量检查 ---")

        # 确定目标列
        target = self.target_column
        if target is None:
            # 自动检测可能的目标列
            target_keywords = ["评级", "标签", "结果", "level", "label", "target"]
            for col in self.df.columns:
                if any(kw in col.lower() for kw in target_keywords):
                    target = col
                    break

        if target is None or target not in self.df.columns:
            print("⚠️ 未找到目标变量列")
            self.profile["target"] = {"found": False}
            return self.profile["target"]

        print(f"目标变量: {target}")

        target_data = self.df[target]
        distribution = target_data.value_counts(dropna=False).to_dict()

        target_stats = {
            "found": True,
            "column": target,
            "dtype": str(target_data.dtype),
            "non_null": int(target_data.count()),
            "null": int(target_data.isnull().sum()),
            "unique": int(target_data.nunique()),
            "distribution": {str(k): int(v) for k, v in distribution.items()},
        }

        # 检查类别不平衡
        if target_data.notna().any():
            value_counts = target_data.value_counts(normalize=True)
            max_ratio = value_counts.max()
            min_ratio = value_counts.min()
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float("inf")

            target_stats["imbalance_ratio"] = round(imbalance_ratio, 2)
            target_stats["is_imbalanced"] = imbalance_ratio > 10

            print(f"分布: {target_stats['distribution']}")
            print(f"不平衡比: {target_stats['imbalance_ratio']:.2f}")
            if target_stats["is_imbalanced"]:
                print("⚠️ 数据严重不平衡，建议处理")

        self.profile["target"] = target_stats
        return target_stats

    def generate_cleaning_suggestions(self) -> List[str]:
        """生成数据清洗建议"""
        if not self.profile:
            return []

        suggestions = []

        # 1. 缺失值建议
        missing = self.profile.get("missing", {})
        if missing.get("high_missing_columns", 0) > 0:
            suggestions.append(
                f"[缺失值] 有 {missing['high_missing_columns']} 个列缺失率>50%，建议删除或检查数据源"
            )
        if missing.get("medium_missing_columns", 0) > 0:
            suggestions.append(
                f"[缺失值] 有 {missing['medium_missing_columns']} 个列缺失率10-50%，建议填充或标记"
            )

        # 2. 重复行建议
        basic = self.profile.get("basic", {})
        if basic.get("duplicate_rows", 0) > 0:
            suggestions.append(
                f"[重复数据] 发现 {basic['duplicate_rows']} 行重复数据，建议去重"
            )

        # 3. 目标变量建议
        target = self.profile.get("target", {})
        if target.get("found") and target.get("is_imbalanced"):
            suggestions.append(
                f"[类别不平衡] 目标变量 '{target.get('column')}' 不平衡比 {target.get('imbalance_ratio')}，建议使用过采样/欠采样"
            )

        # 4. 常量列建议
        columns = self.profile.get("columns", {}).get("columns", {})
        constant_cols = [col for col, info in columns.items() if info.get("unique", 0) <= 1]
        if constant_cols:
            suggestions.append(
                f"[常量列] 发现 {len(constant_cols)} 个常量列，建议删除（无信息量）"
            )

        # 5. 高基数类别列
        high_cardinality = [
            col for col, info in columns.items()
            if info.get("type_category") == "categorical" and info.get("unique", 0) > 100
        ]
        if high_cardinality:
            suggestions.append(
                f"[高基数] 发现 {len(high_cardinality)} 个高基数类别列(>100唯一值)，建议编码处理"
            )

        self.profile["suggestions"] = suggestions
        return suggestions

    def print_suggestions(self):
        """打印清洗建议"""
        suggestions = self.profile.get("suggestions", [])
        if not suggestions:
            return

        print("\n" + "="*60)
        print("数据清洗建议")
        print("="*60)

        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

    def generate_report(self, output_path: str) -> str:
        """
        生成 Markdown 格式的数据概览报告

        Args:
            output_path: 报告输出路径

        Returns:
            报告文件路径
        """
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # 标题
        lines.append(f"# 数据概览报告")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"数据文件: `{self.file_path}`")

        # 1. 基础信息
        basic = self.profile.get("basic", {})
        lines.append("\n## 1. 基础信息")
        lines.append(f"| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 数据量 | {basic.get('rows', 0):,} 行 × {basic.get('columns', 0)} 列 |")
        lines.append(f"| 文件大小 | {self.profile.get('file', {}).get('size_mb', 0):.2f} MB |")
        lines.append(f"| 内存占用 | {basic.get('memory_mb', 0):.2f} MB |")
        lines.append(f"| 重复行 | {basic.get('duplicate_rows', 0):,} |")

        # 2. 缺失值统计
        missing = self.profile.get("missing", {})
        lines.append("\n## 2. 缺失值统计")
        lines.append(f"| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 总缺失值 | {missing.get('total_missing', 0):,} |")
        lines.append(f"| 整体缺失率 | {missing.get('overall_missing_ratio', 0)}% |")
        lines.append(f"| 高缺失列(>50%) | {missing.get('high_missing_columns', 0)} |")
        lines.append(f"| 中缺失列(10-50%) | {missing.get('medium_missing_columns', 0)} |")

        # 高缺失列详情
        columns_missing = missing.get("columns", {})
        high_missing_cols = {k: v for k, v in columns_missing.items() if v.get("missing_ratio", 0) > 10}
        if high_missing_cols:
            lines.append("\n### 高缺失列（>10%）")
            lines.append("| 列名 | 缺失数 | 缺失率 |")
            lines.append("|------|--------|--------|")
            for col, info in sorted(high_missing_cols.items(), key=lambda x: x[1].get("missing_ratio", 0), reverse=True)[:20]:
                lines.append(f"| {col} | {info.get('missing_count', 0):,} | {info.get('missing_ratio', 0)}% |")

        # 3. 列类型分布
        col_stats = self.profile.get("columns", {})
        lines.append("\n## 3. 列类型分布")
        type_counts = col_stats.get("type_counts", {})
        lines.append(f"| 类型 | 数量 |")
        lines.append("|------|------|")
        for dtype, count in type_counts.items():
            lines.append(f"| {dtype} | {count} |")

        # 4. 目标变量
        target = self.profile.get("target", {})
        lines.append("\n## 4. 目标变量")
        if target.get("found"):
            lines.append(f"**目标列**: `{target.get('column')}`")
            lines.append(f"\n| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 数据类型 | {target.get('dtype')} |")
            lines.append(f"| 非空值 | {target.get('non_null', 0):,} |")
            lines.append(f"| 唯一值数 | {target.get('unique', 0)} |")
            lines.append(f"| 不平衡比 | {target.get('imbalance_ratio', 'N/A')} |")

            # 分布表
            distribution = target.get("distribution", {})
            if distribution:
                lines.append("\n### 分布")
                lines.append("| 值 | 数量 |")
                lines.append("|-----|------|")
                for val, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"| {val} | {count:,} |")
        else:
            lines.append("⚠️ 未检测到目标变量")

        # 5. 数据清洗建议
        suggestions = self.profile.get("suggestions", [])
        lines.append("\n## 5. 数据清洗建议")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"{i}. {suggestion}")
        else:
            lines.append("✅ 数据质量良好，无需特殊处理")

        # 写入文件
        content = "\n".join(lines)
        report_path.write_text(content, encoding="utf-8")

        print(f"\n✅ 报告已生成: {report_path}")
        return str(report_path)

    def run(self) -> Dict[str, Any]:
        """执行完整探查流程"""
        if not self.load():
            return self.profile

        self.profile_basic()
        self.profile_missing()
        self.profile_columns()
        self.profile_target()
        self.generate_cleaning_suggestions()
        self.print_suggestions()

        return self.profile


def diagnose(file_path: str, target_column: Optional[str] = None, report_path: Optional[str] = None) -> Dict[str, Any]:
    """
    诊断数据文件

    Args:
        file_path: 数据文件路径
        target_column: 目标变量列名
        report_path: 报告输出路径

    Returns:
        探查结果字典
    """
    profiler = DataProfiler(file_path, target_column)
    profile = profiler.run()

    if report_path:
        profiler.generate_report(report_path)

    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="数据探查诊断脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础探查
    uv run python scripts/diagnose_data.py ./data/线索宽表_完整.parquet

    # 指定目标变量
    uv run python scripts/diagnose_data.py ./data/train.parquet --target 线索评级结果

    # 生成 Markdown 报告
    uv run python scripts/diagnose_data.py ./data/线索宽表.parquet --report ./reports/data_profile.md

    # 探查训练集和测试集
    uv run python scripts/diagnose_data.py ./data/线索宽表_train.parquet --report ./reports/train_profile.md
    uv run python scripts/diagnose_data.py ./data/线索宽表_test.parquet --report ./reports/test_profile.md
        """,
    )
    parser.add_argument("file", help="数据文件路径")
    parser.add_argument("--target", default=None, help="目标变量列名")
    parser.add_argument("--report", default=None, help="报告输出路径（Markdown 格式）")

    args = parser.parse_args()

    diagnose(args.file, args.target, args.report)