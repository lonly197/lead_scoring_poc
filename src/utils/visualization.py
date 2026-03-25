"""
可视化工具模块

负责生成模型评估和解释性相关的图表。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# 设置中文字体（适配 MacOS 和 Linux）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    logger.info(f"生成特征重要性图表: {output_path}")
    
    # 取前 N 个特征
    df_plot = importance_df.head(top_n).copy()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=df_plot, palette="viridis")
    
    plt.title(title, fontsize=15)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"图表已保存")

def plot_dimension_contribution(
    dimension_contribution: Dict[str, float],
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
    logger.info(f"生成业务维度贡献图表: {output_path}")
    
    # 过滤掉分值为 0 的维度
    plot_data = {k: v for k, v in dimension_contribution.items() if v > 0}
    
    if not plot_data:
        logger.warning("没有可绘制的业务维度数据")
        return
        
    labels = list(plot_data.keys())
    sizes = list(plot_data.values())
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.axis('equal')  # 保证饼图是正圆
    
    plt.title(title, fontsize=15)
    plt.tight_layout()
    
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"图表已保存")
