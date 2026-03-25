"""OHAB 多分类评估与排序分析工具。"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.label_policy import ordered_ohab_labels


def classification_report_dict(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    labels = ordered_ohab_labels(list(y_true) + list(y_pred))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_frame(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    labels = ordered_ohab_labels(list(y_true) + list(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def compute_class_ranking_report(
    y_true: Sequence[str],
    proba_df: pd.DataFrame,
    top_ratios: Iterable[float] = (0.05, 0.10, 0.20),
) -> list[Dict[str, float]]:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    total_samples = len(y_true_series)
    report: list[Dict[str, float]] = []

    if total_samples == 0:
        return report

    for label in ordered_ohab_labels(proba_df.columns):
        if label not in proba_df.columns:
            continue

        baseline_rate = float((y_true_series == label).mean())
        label_scores = proba_df[label].reset_index(drop=True)

        for ratio in top_ratios:
            top_n = max(1, int(total_samples * ratio))
            top_indices = label_scores.sort_values(ascending=False).head(top_n).index
            hit_rate = float((y_true_series.loc[top_indices] == label).mean()) if top_n else 0.0
            lift = hit_rate / baseline_rate if baseline_rate > 0 else 0.0
            report.append(
                {
                    "class": label,
                    "top_ratio": float(ratio),
                    "top_n": int(top_n),
                    "baseline_rate": baseline_rate,
                    "hit_rate": hit_rate,
                    "lift": lift,
                }
            )

    return report


def compute_threshold_report(
    y_true: Sequence[str],
    positive_scores: Sequence[float],
    positive_label: str = "B",
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    score_series = pd.Series(positive_scores).reset_index(drop=True)
    thresholds = list(thresholds or [round(step, 2) for step in [0.1, 0.2, 0.3, 0.4, 0.5]])

    rows = []
    truth = (y_true_series == positive_label).astype(int)

    for threshold in thresholds:
        prediction = (score_series >= threshold).astype(int)
        predicted_positive = int(prediction.sum())
        true_positive = int(((prediction == 1) & (truth == 1)).sum())
        actual_positive = int(truth.sum())
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
        recall = true_positive / actual_positive if actual_positive > 0 else 0.0
        f1 = f1_score(truth, prediction, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_positive": predicted_positive,
                "actual_positive": actual_positive,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": float(f1),
            }
        )

    return pd.DataFrame(rows)
