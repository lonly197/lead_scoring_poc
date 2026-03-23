"""
评估指标模块

实现 Top-K 命中率、分层转化率、基线对比等评估指标。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TopKEvaluator:
    """Top-K 评估器"""

    def __init__(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        ids: Optional[np.ndarray] = None,
    ):
        """
        初始化评估器

        Args:
            y_true: 真实标签
            y_proba: 预测概率（正类概率）
            ids: 样本 ID（可选）
        """
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.ids = ids

        # 创建排序后的索引
        self.sorted_indices = np.argsort(y_proba)[::-1]  # 降序

    def compute_topk_metrics(
        self, k_values: List[int] = [100, 500, 1000, 2000]
    ) -> Dict[str, Dict]:
        """
        计算 Top-K 指标

        Args:
            k_values: K 值列表

        Returns:
            Top-K 指标字典
        """
        results = {}
        total_positives = self.y_true.sum()

        for k in k_values:
            if k > len(self.y_true):
                logger.warning(f"k={k} 超过样本数 {len(self.y_true)}，跳过")
                continue

            # Top-K 索引
            topk_indices = self.sorted_indices[:k]

            # Top-K 的真实标签
            topk_true = self.y_true[topk_indices]

            # 计算指标
            hit_count = topk_true.sum()
            hit_rate = hit_count / k
            precision = hit_count / k
            recall = hit_count / total_positives if total_positives > 0 else 0

            results[f"top_{k}"] = {
                "k": k,
                "hit_count": int(hit_count),
                "hit_rate": float(hit_rate),
                "precision": float(precision),
                "recall": float(recall),
                "expected_rate": float(total_positives / len(self.y_true)),
                "lift": float(hit_rate / (total_positives / len(self.y_true)))
                if total_positives > 0
                else 0,
            }

            logger.info(
                f"Top-{k}: 命中 {hit_count}/{k} ({hit_rate:.2%}), "
                f"Lift: {results[f'top_{k}']['lift']:.2f}x"
            )

        return results

    def get_topk_ids(self, k: int) -> np.ndarray:
        """
        获取 Top-K 样本的 ID

        Args:
            k: K 值

        Returns:
            Top-K 样本的 ID 数组
        """
        if self.ids is None:
            raise ValueError("未提供样本 ID")

        topk_indices = self.sorted_indices[:k]
        return self.ids[topk_indices]

    def compute_lift_by_decile(self) -> pd.DataFrame:
        """
        按十分位计算 Lift

        Returns:
            十分位 Lift DataFrame
        """
        n = len(self.y_true)
        decile_size = n // 10

        results = []

        for i in range(10):
            start = i * decile_size
            end = (i + 1) * decile_size if i < 9 else n

            # 当前十分位的索引（按概率降序）
            decile_indices = self.sorted_indices[start:end]

            # 计算指标
            decile_true = self.y_true[decile_indices]
            hit_rate = decile_true.mean()
            baseline_rate = self.y_true.mean()
            lift = hit_rate / baseline_rate if baseline_rate > 0 else 0

            results.append({
                "decile": i + 1,  # 1-10
                "count": end - start,
                "hit_count": int(decile_true.sum()),
                "hit_rate": float(hit_rate),
                "cumulative_hit_rate": float(self.y_true[self.sorted_indices[:end]].mean()),
                "lift": float(lift),
            })

        return pd.DataFrame(results)

    def compare_with_baseline(
        self,
        baseline_scores: np.ndarray,
        k_values: List[int] = [100, 500, 1000],
    ) -> Dict:
        """
        与基线方法对比

        Args:
            baseline_scores: 基线方法预测概率或得分
            k_values: K 值列表

        Returns:
            对比结果字典
        """
        # 基线排序
        baseline_sorted = np.argsort(baseline_scores)[::-1]

        results = {}

        for k in k_values:
            if k > len(self.y_true):
                continue

            # 模型 Top-K
            model_topk = self.sorted_indices[:k]
            model_hits = self.y_true[model_topk].sum()

            # 基线 Top-K
            baseline_topk = baseline_sorted[:k]
            baseline_hits = self.y_true[baseline_topk].sum()

            # 对比
            improvement = (model_hits - baseline_hits) / baseline_hits if baseline_hits > 0 else 0

            results[f"top_{k}"] = {
                "model_hits": int(model_hits),
                "baseline_hits": int(baseline_hits),
                "improvement": float(improvement),
                "overlap_count": int(len(set(model_topk) & set(baseline_topk))),
            }

        return results


class StratifiedEvaluator:
    """分层评估器"""

    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray, scores: np.ndarray):
        """
        初始化分层评估器

        Args:
            y_true: 真实标签
            y_proba: 预测概率
            scores: 预测得分（用于分层）
        """
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.scores = np.array(scores)

    def compute_stratified_metrics(
        self, n_bins: int = 10
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        计算分层指标

        Args:
            n_bins: 分层数量

        Returns:
            分层统计 DataFrame 和汇总指标
        """
        # 按得分分层
        bins = pd.qcut(self.scores, q=n_bins, labels=False, duplicates="drop")

        results = []

        for bin_id in range(n_bins):
            mask = bins == bin_id
            if mask.sum() == 0:
                continue

            bin_true = self.y_true[mask]
            bin_proba = self.y_proba[mask]

            results.append({
                "bin": bin_id + 1,
                "count": mask.sum(),
                "actual_rate": float(bin_true.mean()),
                "predicted_rate": float(bin_proba.mean()),
                "positive_count": int(bin_true.sum()),
            })

        df = pd.DataFrame(results)

        # 汇总指标
        summary = {
            "total_samples": len(self.y_true),
            "total_positives": int(self.y_true.sum()),
            "overall_rate": float(self.y_true.mean()),
            "high_bin_rate": float(df[df["bin"] == df["bin"].max()]["actual_rate"].iloc[0])
            if len(df) > 0
            else 0,
            "low_bin_rate": float(df[df["bin"] == df["bin"].min()]["actual_rate"].iloc[0])
            if len(df) > 0
            else 0,
        }

        return df, summary


class ModelReport:
    """模型评估报告生成器"""

    def __init__(self, output_dir: str):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model_metrics: Dict,
        topk_metrics: Dict,
        feature_importance: Optional[pd.DataFrame] = None,
        stratified_metrics: Optional[pd.DataFrame] = None,
        model_name: str = "model",
    ) -> Dict:
        """
        生成评估报告

        Args:
            model_metrics: 模型评估指标
            topk_metrics: Top-K 指标
            feature_importance: 特征重要性
            stratified_metrics: 分层指标
            model_name: 模型名称

        Returns:
            完整报告字典
        """
        report = {
            "model_name": model_name,
            "model_metrics": model_metrics,
            "topk_metrics": topk_metrics,
        }

        if feature_importance is not None:
            report["feature_importance"] = feature_importance.head(20).to_dict("records")

        if stratified_metrics is not None:
            report["stratified_metrics"] = stratified_metrics.to_dict("records")

        # 保存报告
        report_path = self.output_dir / f"{model_name}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"报告已保存: {report_path}")

        return report

    def generate_topk_list(
        self,
        ids: np.ndarray,
        y_proba: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        k: int = 1000,
        model_name: str = "model",
        id_column: str = "门店线索编号",
    ) -> pd.DataFrame:
        """
        生成 Top-K 名单

        Args:
            ids: 样本 ID
            y_proba: 预测概率
            y_true: 真实标签（可选）
            k: K 值
            model_name: 模型名称
            id_column: ID 列名

        Returns:
            Top-K 名单 DataFrame
        """
        # 排序
        sorted_indices = np.argsort(y_proba)[::-1][:k]

        # 创建名单
        topk_df = pd.DataFrame({
            "rank": range(1, k + 1),
            id_column: ids[sorted_indices],
            "predicted_proba": y_proba[sorted_indices],
        })

        if y_true is not None:
            topk_df["actual_label"] = y_true[sorted_indices]

        # 保存名单
        output_path = self.output_dir / f"{model_name}_top{k}.csv"
        topk_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info(f"Top-{k} 名单已保存: {output_path}")

        return topk_df


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ids: Optional[np.ndarray] = None,
    k_values: List[int] = [100, 500, 1000],
) -> Dict:
    """
    计算所有评估指标

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        ids: 样本 ID
        k_values: K 值列表

    Returns:
        完整指标字典
    """
    evaluator = TopKEvaluator(y_true, y_proba, ids)

    metrics = {
        "topk_metrics": evaluator.compute_topk_metrics(k_values),
        "decile_lift": evaluator.compute_lift_by_decile().to_dict("records"),
        "summary": {
            "total_samples": len(y_true),
            "total_positives": int(y_true.sum()),
            "positive_rate": float(y_true.mean()),
        },
    }

    return metrics


def plot_lift_chart(
    decile_data: pd.DataFrame, output_path: Optional[str] = None
) -> None:
    """
    绘制 Lift 图表

    Args:
        decile_data: 十分位数据
        output_path: 输出路径
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Lift 柱状图
        ax1 = axes[0]
        bars = ax1.bar(decile_data["decile"], decile_data["lift"], color="steelblue")
        ax1.axhline(y=1, color="red", linestyle="--", label="Baseline")
        ax1.set_xlabel("Decile")
        ax1.set_ylabel("Lift")
        ax1.set_title("Lift by Decile")
        ax1.legend()

        # 累积命中率
        ax2 = axes[1]
        ax2.plot(
            decile_data["decile"],
            decile_data["cumulative_hit_rate"],
            marker="o",
            color="green",
        )
        ax2.axhline(
            y=decile_data["hit_rate"].mean(),
            color="red",
            linestyle="--",
            label="Random",
        )
        ax2.set_xlabel("Decile")
        ax2.set_ylabel("Cumulative Hit Rate")
        ax2.set_title("Cumulative Hit Rate by Decile")
        ax2.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Lift 图表已保存: {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.warning("matplotlib 或 seaborn 未安装，跳过图表生成")


def plot_feature_importance(
    importance_df: pd.DataFrame, top_n: int = 20, output_path: Optional[str] = None
) -> None:
    """
    绘制特征重要性图表

    Args:
        importance_df: 特征重要性 DataFrame
        top_n: 显示前 N 个特征
        output_path: 输出路径
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 取前 N 个特征
        plot_df = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.barh(
            range(len(plot_df)), plot_df["importance"], color="steelblue"
        )
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"特征重要性图表已保存: {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.warning("matplotlib 或 seaborn 未安装，跳过图表生成")