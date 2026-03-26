#!/usr/bin/env python3
"""
本地图表生成脚本

读取服务器同步回来的 CSV/JSON 产出物，在本地环境生成带中文字体的图表。
适用于服务器 CLI 环境无法显示中文的场景。

使用方式：
    # 生成指定验证目录的图表
    uv run python scripts/generate_local_plots.py --input-dir ./outputs/validation_arrive

    # 指定输出目录
    uv run python scripts/generate_local_plots.py --input-dir ./outputs/validation_arrive --output-dir ./outputs/plots/arrive
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
    绘制 HAB 桶分布图表
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 桶大小分布
    ax1 = axes[0]
    if "预测标签" in bucket_df.columns and "数量" in bucket_df.columns:
        bars = ax1.bar(bucket_df["预测标签"], bucket_df["数量"], color=["#2ecc71", "#f39c12", "#e74c3c"])
        ax1.set_xlabel("预测等级", fontsize=11)
        ax1.set_ylabel("数量", fontsize=11)
        ax1.set_title("H/A/B 桶样本分布", fontsize=13)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{int(bar.get_height())}", ha="center", fontsize=10)

    # 到店率分层（如果有）
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