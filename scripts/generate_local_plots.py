#!/usr/bin/env python3
"""
本地图表生成脚本

读取服务器同步回来的 CSV/JSON 产出物，在本地环境生成带中文字体的图表。
适用于服务器 CLI 环境无法显示中文的场景。

支持的模型验证输出：
- 到店模型 (validation/arrive_validation): 特征重要性、Lift 图、Top-K 指标、HAB 桶分布
- 试驾模型 (validation/test_drive_validation): 特征重要性、Lift 图、Top-K 指标
- OHAB 模型 (validation/ohab_validation): HAB 桶分布、混淆矩阵、分类报告、单调性检查

使用方式：
    # 生成到店模型验证图表
    uv run python scripts/generate_local_plots.py --input-dir ./outputs/validation/arrive_validation

    # 生成 OHAB 模型验证图表
    uv run python scripts/generate_local_plots.py --input-dir ./outputs/validation/ohab_validation

    # 生成试驾模型验证图表
    uv run python scripts/generate_local_plots.py --input-dir ./outputs/validation/test_drive_validation

    # 指定输出目录
    uv run python scripts/generate_local_plots.py --input-dir ./outputs/validation/ohab_validation --output-dir ./outputs/plots/ohab
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def setup_chinese_fonts():
    """配置中文字体支持"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 按优先级排列的中文字体（优先使用效果好的）
    chinese_fonts = [
        "Songti SC",        # macOS 宋体
        "PingFang SC",      # macOS 苹方
        "Noto Sans CJK SC", # Linux 思源黑体
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",  # Windows 微软雅黑
        "SimHei",           # Windows 黑体
        "Heiti SC",         # macOS 黑体
        "STHeiti",
        "Arial Unicode MS", # 兜底字体（中文支持有限）
    ]

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        logger.info(f"使用中文字体: {selected_font}")
        return True
    else:
        logger.warning("未找到中文字体，图表中的中文可能显示异常")
        logger.info(f"可用字体: {sorted(available_fonts)[:10]}...")
        return False


def plot_feature_importance(importance_df: pd.DataFrame, output_path: str, top_n: int = 20):
    """
    绘制特征重要性图表（支持中文）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    df_plot = importance_df.head(top_n).copy().sort_values("importance", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(df_plot)))

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df_plot["feature"], df_plot["importance"], color=colors)

    ax.set_title("特征重要性 (Feature Importance)", fontsize=15)
    ax.set_xlabel("重要性 (Importance)", fontsize=12)
    ax.set_ylabel("特征 (Feature)", fontsize=12)

    # 添加数值标签
    for bar, val in zip(bars, df_plot["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"特征重要性图表已保存: {output_path}")


def plot_lift_chart(lift_df: pd.DataFrame, output_path: str):
    """
    绘制 Lift 图表
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lift 柱状图
    ax1 = axes[0]
    bars = ax1.bar(lift_df["decile"], lift_df["lift"], color="steelblue")
    ax1.axhline(y=1, color="red", linestyle="--", label="基线 (Baseline)")
    ax1.set_xlabel("十分位 (Decile)", fontsize=11)
    ax1.set_ylabel("提升度 (Lift)", fontsize=11)
    ax1.set_title("各十分位提升度 (Lift by Decile)", fontsize=13)
    ax1.legend()

    # 累积命中率
    ax2 = axes[1]
    ax2.plot(lift_df["decile"], lift_df["cumulative_hit_rate"],
             marker="o", color="green", linewidth=2, markersize=8)
    ax2.axhline(y=lift_df["hit_rate"].mean(), color="red",
                linestyle="--", label="随机基线 (Random)")
    ax2.set_xlabel("十分位 (Decile)", fontsize=11)
    ax2.set_ylabel("累积命中率 (Cumulative Hit Rate)", fontsize=11)
    ax2.set_title("累积命中率曲线 (Cumulative Hit Rate)", fontsize=13)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Lift 图表已保存: {output_path}")


def plot_topk_metrics(topk_metrics: dict, output_path: str):
    """
    绘制 Top-K 指标图表
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k_values = []
    hit_rates = []
    lifts = []

    for key, val in topk_metrics.items():
        if key.startswith("top_"):
            k_values.append(val["k"])
            hit_rates.append(val["hit_rate"] * 100)
            lifts.append(val["lift"])

    x = np.arange(len(k_values))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "steelblue"
    ax1.bar(x - width/2, hit_rates, width, label="命中率 (%)", color=color1)
    ax1.set_xlabel("Top-K", fontsize=11)
    ax1.set_ylabel("命中率 (%)", color=color1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Top-{k}" for k in k_values])

    ax2 = ax1.twinx()
    color2 = "darkorange"
    ax2.bar(x + width/2, lifts, width, label="提升度 (Lift)", color=color2)
    ax2.set_ylabel("提升度 (Lift)", color=color2, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    ax1.set_title("Top-K 指标分析 (Top-K Metrics Analysis)", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Top-K 指标图表已保存: {output_path}")


def plot_hab_bucket_summary(bucket_df: pd.DataFrame, output_path: str):
    """
    绘制 HAB 桶分布图表（支持详细的转化率分层）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 检查是否是 OHAB 格式（有转化率列）
    is_ohab_format = "到店标签_14天_rate" in bucket_df.columns

    if is_ohab_format:
        # OHAB 格式：绘制转化率分层对比
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        buckets = bucket_df["bucket"].tolist()
        x = np.arange(len(buckets))
        width = 0.25

        # 1. 样本分布
        ax1 = axes[0, 0]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        bars = ax1.bar(buckets, bucket_df["sample_count"], color=colors)
        ax1.set_xlabel("预测等级", fontsize=11)
        ax1.set_ylabel("样本数量", fontsize=11)
        ax1.set_title("H/A/B 桶样本分布", fontsize=13)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=9)

        # 2. 到店率分层
        ax2 = axes[0, 1]
        x = np.arange(len(buckets))
        ax2.bar(x - width, bucket_df["到店标签_7天_rate"] * 100, width, label="7天到店率", color="#3498db")
        ax2.bar(x, bucket_df["到店标签_14天_rate"] * 100, width, label="14天到店率", color="#2ecc71")
        ax2.bar(x + width, bucket_df["到店标签_30天_rate"] * 100, width, label="30天到店率", color="#e74c3c")
        ax2.set_xlabel("预测等级", fontsize=11)
        ax2.set_ylabel("到店率 (%)", fontsize=11)
        ax2.set_title("H/A/B 到店率分层", fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(buckets)
        ax2.legend()

        # 3. 试驾率分层
        ax3 = axes[1, 0]
        ax3.bar(x - width, bucket_df["试驾标签_7天_rate"] * 100, width, label="7天试驾率", color="#3498db")
        ax3.bar(x, bucket_df["试驾标签_14天_rate"] * 100, width, label="14天试驾率", color="#2ecc71")
        ax3.bar(x + width, bucket_df["试驾标签_30天_rate"] * 100, width, label="30天试驾率", color="#e74c3c")
        ax3.set_xlabel("预测等级", fontsize=11)
        ax3.set_ylabel("试驾率 (%)", fontsize=11)
        ax3.set_title("H/A/B 试驾率分层", fontsize=13)
        ax3.set_xticks(x)
        ax3.set_xticklabels(buckets)
        ax3.legend()

        # 4. 最终转化率
        ax4 = axes[1, 1]
        bars = ax4.bar(buckets, bucket_df["is_final_ordered_rate"] * 100, color=["#2ecc71", "#f39c12", "#e74c3c"])
        ax4.set_xlabel("预测等级", fontsize=11)
        ax4.set_ylabel("成交率 (%)", fontsize=11)
        ax4.set_title("H/A/B 最终成交率", fontsize=13)
        for bar in bars:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}%", ha="center", fontsize=9)

    else:
        # 简单格式（arrive 模型）
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        if "预测标签" in bucket_df.columns and "数量" in bucket_df.columns:
            bars = ax1.bar(bucket_df["预测标签"], bucket_df["数量"], color=["#2ecc71", "#f39c12", "#e74c3c"])
            ax1.set_xlabel("预测等级", fontsize=11)
            ax1.set_ylabel("数量", fontsize=11)
            ax1.set_title("H/A/B 桶样本分布", fontsize=13)
            for bar in bars:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f"{int(bar.get_height())}", ha="center", fontsize=10)

        ax2 = axes[1]
        rate_col = None
        for col in ["到店率", "转化率", "hit_rate"]:
            if col in bucket_df.columns:
                rate_col = col
                break

        if rate_col:
            bars = ax2.bar(bucket_df["预测标签"], bucket_df[rate_col] * 100,
                           color=["#2ecc71", "#f39c12", "#e74c3c"])
            ax2.set_xlabel("预测等级", fontsize=11)
            ax2.set_ylabel("转化率 (%)", fontsize=11)
            ax2.set_title("H/A/B 桶转化率分层", fontsize=13)
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"HAB 桶分布图表已保存: {output_path}")


def plot_confusion_matrix(confusion_path: Path, output_path: str):
    """
    绘制混淆矩阵热力图
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    df = pd.read_csv(confusion_path, index_col=0)
    matrix = df.values
    labels = df.columns.tolist()

    fig, ax = plt.subplots(figsize=(8, 6))

    # 使用 seaborn 绘制热力图
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel("预测标签", fontsize=11)
    ax.set_ylabel("真实标签", fontsize=11)
    ax.set_title("混淆矩阵 (Confusion Matrix)", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"混淆矩阵图表已保存: {output_path}")


def plot_classification_report(report_path: Path, output_path: str):
    """
    绘制分类报告雷达图/柱状图
    """
    import matplotlib.pyplot as plt
    import numpy as np

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # 提取各类别指标
    classes = []
    precision_list = []
    recall_list = []
    f1_list = []

    for key in ["H", "A", "B"]:
        if key in report:
            classes.append(key)
            precision_list.append(report[key]["precision"])
            recall_list.append(report[key]["recall"])
            f1_list.append(report[key]["f1-score"])

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, precision_list, width, label="精确率 (Precision)", color="#3498db")
    ax.bar(x, recall_list, width, label="召回率 (Recall)", color="#2ecc71")
    ax.bar(x + width, f1_list, width, label="F1分数", color="#e74c3c")

    ax.set_xlabel("预测等级", fontsize=11)
    ax.set_ylabel("分数", fontsize=11)
    ax.set_title("H/A/B 分类指标对比", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # 添加数值标签
    for i, (p, r, f) in enumerate(zip(precision_list, recall_list, f1_list)):
        ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)

    # 添加整体指标
    if "accuracy" in report:
        ax.text(0.98, 0.98, f"准确率: {report['accuracy']:.2%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"分类报告图表已保存: {output_path}")


def plot_monotonicity_check(monotonicity_path: Path, output_path: str):
    """
    绘制单调性检查结果
    """
    import matplotlib.pyplot as plt

    with open(monotonicity_path, encoding="utf-8") as f:
        data = json.load(f)

    passed = data.get("passed", False)
    metric = data.get("metric", "unknown")
    values = data.get("values", {})

    fig, ax = plt.subplots(figsize=(8, 5))

    buckets = list(values.keys())
    rates = [v * 100 for v in values.values()]

    colors = ["#2ecc71" if passed else "#e74c3c"] * len(buckets)
    bars = ax.bar(buckets, rates, color=colors)

    ax.set_xlabel("预测等级", fontsize=11)
    ax.set_ylabel(f"{metric} (%)", fontsize=11)

    status = "通过" if passed else "未通过"
    ax.set_title(f"H/A/B 单调性检查 ({status})", fontsize=13)

    # 添加数值标签
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", fontsize=10)

    # 添加检查信息
    if "message" in data:
        ax.text(0.5, 0.95, data["message"], transform=ax.transAxes,
                ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"单调性检查图表已保存: {output_path}")


def generate_plots(input_dir: Path, output_dir: Path):
    """生成所有图表"""

    # 设置中文字体
    has_chinese = setup_chinese_fonts()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 特征重要性
    importance_path = input_dir / "feature_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        plot_feature_importance(importance_df, str(output_dir / "feature_importance.png"))
    else:
        logger.warning(f"未找到特征重要性文件: {importance_path}")

    # 2. Lift 图
    lift_path = input_dir / "lift_deciles.csv"
    if lift_path.exists():
        lift_df = pd.read_csv(lift_path)
        plot_lift_chart(lift_df, str(output_dir / "lift_chart.png"))
    else:
        logger.warning(f"未找到 Lift 数据文件: {lift_path}")

    # 3. Top-K 指标
    topk_path = input_dir / "topk_metrics.json"
    if topk_path.exists():
        with open(topk_path, encoding="utf-8") as f:
            topk_metrics = json.load(f)
        plot_topk_metrics(topk_metrics, str(output_dir / "topk_metrics.png"))
    else:
        logger.warning(f"未找到 Top-K 指标文件: {topk_path}")

    # 4. HAB 桶分布（如果是 OHAB 模型）
    bucket_path = input_dir / "hab_bucket_summary.csv"
    if bucket_path.exists():
        bucket_df = pd.read_csv(bucket_path)
        plot_hab_bucket_summary(bucket_df, str(output_dir / "hab_bucket_summary.png"))
    else:
        # 也检查模型目录
        model_bucket_path = input_dir.parent / "models" / "ohab_model" / "hab_bucket_summary.csv"
        if model_bucket_path.exists():
            bucket_df = pd.read_csv(model_bucket_path)
            plot_hab_bucket_summary(bucket_df, str(output_dir / "hab_bucket_summary.png"))

    # 5. 混淆矩阵（OHAB 模型）
    confusion_path = input_dir / "confusion_matrix.csv"
    if confusion_path.exists():
        plot_confusion_matrix(confusion_path, str(output_dir / "confusion_matrix.png"))
    else:
        logger.debug(f"未找到混淆矩阵文件: {confusion_path}")

    # 6. 分类报告（OHAB 模型）
    report_path = input_dir / "classification_report.json"
    if report_path.exists():
        plot_classification_report(report_path, str(output_dir / "classification_report.png"))
    else:
        logger.debug(f"未找到分类报告文件: {report_path}")

    # 7. 单调性检查（OHAB 模型）
    monotonicity_path = input_dir / "monotonicity_check.json"
    if monotonicity_path.exists():
        plot_monotonicity_check(monotonicity_path, str(output_dir / "monotonicity_check.png"))
    else:
        logger.debug(f"未找到单调性检查文件: {monotonicity_path}")

    logger.info(f"图表生成完成，输出目录: {output_dir}")
    print(f"\n✅ 图表已生成到: {output_dir}")
    print("生成的图表:")
    for f in output_dir.glob("*.png"):
        print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description="本地图表生成脚本")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入目录（包含 CSV/JSON 产出物的目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为输入目录下的 plots 子目录）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"

    generate_plots(input_dir, output_dir)


if __name__ == "__main__":
    main()
