"""
可视化工具模块。

负责生成模型评估和解释性相关的图表。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

logger = logging.getLogger(__name__)

_FONT_CONFIGURED = False
_HAS_CHINESE_FONT = False
_CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Arial Unicode MS",
]


def _available_font_names() -> set[str]:
    return {font.name for font in font_manager.fontManager.ttflist}


def _configure_plot_fonts() -> bool:
    global _FONT_CONFIGURED, _HAS_CHINESE_FONT
    if _FONT_CONFIGURED:
        return _HAS_CHINESE_FONT

    selected_font = None
    available_fonts = _available_font_names()
    for candidate in _CHINESE_FONT_CANDIDATES:
        if candidate in available_fonts:
            selected_font = candidate
            break

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
        _HAS_CHINESE_FONT = True
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        _HAS_CHINESE_FONT = False
        logger.warning("未检测到可用中文字体，PNG 图表中的中文标签可能显示异常；这不会影响 CSV/JSON 结构化产物。")

    plt.rcParams["axes.unicode_minus"] = False
    _FONT_CONFIGURED = True
    return _HAS_CHINESE_FONT


def _plot_warning_context():
    has_chinese_font = _configure_plot_fonts()
    context = warnings.catch_warnings()
    context.__enter__()
    if not has_chinese_font:
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font", category=UserWarning)
    return context


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str,
    top_n: int = 20,
    title: str = "特征重要性 (Feature Importance)"
):
    """
    绘制特征重要性柱状图

    Args:
        importance_df: 包含 'feature' 和 'importance' 列的 DataFrame
        output_path: 图片保存路径
        top_n: 显示前 N 个特征
        title: 图表标题
    """
    logger.info("生成特征重要性图表: %s", output_path)

    df_plot = importance_df.head(top_n).copy().sort_values("importance", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(df_plot)))

    warning_context = _plot_warning_context()
    try:
        plt.figure(figsize=(12, 8))
        plt.barh(df_plot["feature"], df_plot["importance"], color=colors)
        plt.title(title, fontsize=15)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    finally:
        warning_context.__exit__(None, None, None)

    logger.info("图表已保存")


def plot_dimension_contribution(
    dimension_contribution: dict[str, float],
    output_path: str,
    title: str = "业务维度贡献分析 (Business Dimension Contribution)"
):
    """
    绘制业务维度贡献饼图

    Args:
        dimension_contribution: 业务维度贡献字典
        output_path: 图片保存路径
        title: 图表标题
    """
    logger.info("生成业务维度贡献图表: %s", output_path)

    plot_data = {k: v for k, v in dimension_contribution.items() if v > 0}
    if not plot_data:
        logger.warning("没有可绘制的业务维度数据")
        return

    labels = list(plot_data.keys())
    sizes = list(plot_data.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    warning_context = _plot_warning_context()
    try:
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, shadow=True, colors=colors)
        plt.axis("equal")
        plt.title(title, fontsize=15)
        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    finally:
        warning_context.__exit__(None, None, None)

    logger.info("图表已保存")
