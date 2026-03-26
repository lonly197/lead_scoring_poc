from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

def build_split_group_key(df: pd.DataFrame) -> pd.Series:
    """构造随机切分分组键，优先手机号，其次线索唯一ID。"""
    if "手机号_脱敏" in df.columns:
        phone = df["手机号_脱敏"].astype("string").fillna("").str.strip()
    elif "手机号" in df.columns:
        phone = df["手机号"].astype("string").fillna("").str.strip()
    else:
        phone = pd.Series("", index=df.index, dtype="string")

    invalid_phone = phone.isin({"", "nan", "None", "null"})

    if "线索唯一ID" in df.columns:
        lead_id = df["线索唯一ID"].astype("string").fillna("").str.strip()
    else:
        lead_id = pd.Series(df.index.astype(str), index=df.index, dtype="string")

    fallback = lead_id.where(~lead_id.isin({"", "nan", "None", "null"}), pd.Series(df.index.astype(str), index=df.index, dtype="string"))
    group_key = pd.Series(
        np.where(~invalid_phone, "phone::" + phone.astype(str), "lead::" + fallback.astype(str)),
        index=df.index,
        name="split_group_key",
    )
    return group_key.astype(str)


def _macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    labels = sorted(set(y_true.astype(str)).union(set(y_pred.astype(str))))
    scores = []
    true_values = y_true.astype(str)
    pred_values = y_pred.astype(str)
    for label in labels:
        tp = int(((true_values == label) & (pred_values == label)).sum())
        fp = int(((true_values != label) & (pred_values == label)).sum())
        fn = int(((true_values == label) & (pred_values != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        score = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def _balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    labels = sorted(set(y_true.astype(str)))
    recalls = []
    true_values = y_true.astype(str)
    pred_values = y_pred.astype(str)
    for label in labels:
        positives = true_values == label
        tp = int((positives & (pred_values == label)).sum())
        fn = int((positives & (pred_values != label)).sum())
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def _weighted_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """计算 weighted F1 分数，按各类别样本数加权平均。"""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true.astype(str), y_pred.astype(str), average="weighted", zero_division=0))


def combine_stage_predictions(
    stage1_h_proba: pd.Series,
    stage2_ab_proba: pd.DataFrame,
    h_threshold: float,
) -> tuple[pd.Series, pd.DataFrame]:
    h_score = pd.to_numeric(stage1_h_proba, errors="coerce").fillna(0.0)
    ab = stage2_ab_proba.reindex(columns=["A", "B"]).fillna(0.0)

    h_mask = h_score >= h_threshold
    y_pred = pd.Series(
        ["H" if is_h else ("A" if a_score >= b_score else "B") for is_h, a_score, b_score in zip(h_mask, ab["A"], ab["B"])],
        index=h_score.index,
        name="预测标签",
    )

    final_proba = pd.DataFrame(index=h_score.index)
    final_proba["H"] = h_score
    remaining = 1.0 - h_score
    final_proba["A"] = remaining * ab["A"]
    final_proba["B"] = remaining * ab["B"]
    return y_pred, final_proba


def tune_h_threshold(
    y_true: pd.Series,
    stage1_h_proba: pd.Series,
    stage2_ab_proba: pd.DataFrame,
    thresholds: Iterable[float],
) -> dict:
    best_result = None
    for threshold in thresholds:
        y_pred, _ = combine_stage_predictions(stage1_h_proba, stage2_ab_proba, h_threshold=float(threshold))
        metrics = {
            "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
            "macro_f1": _macro_f1(y_true, y_pred),
        }
        candidate = {
            "strategy": "two_stage_threshold",
            "h_threshold": float(threshold),
            "metrics": metrics,
        }
        if best_result is None:
            best_result = candidate
            continue
        if metrics["balanced_accuracy"] > best_result["metrics"]["balanced_accuracy"]:
            best_result = candidate
        elif (
            metrics["balanced_accuracy"] == best_result["metrics"]["balanced_accuracy"]
            and metrics["macro_f1"] > best_result["metrics"]["macro_f1"]
        ):
            best_result = candidate
    return best_result or {"strategy": "two_stage_threshold", "h_threshold": 0.5, "metrics": {}}


__all__ = [
    "build_split_group_key",
    "combine_stage_predictions",
    "tune_h_threshold",
]


def prepare_stage1_labels(df: pd.DataFrame, target_label: str) -> pd.Series:
    return df[target_label].astype(str).apply(lambda value: "H" if value == "H" else "非H")


def prepare_stage2_frame(df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    subset = df[df[target_label].astype(str) != "H"].copy()
    subset["stage2_label"] = subset[target_label].astype(str)
    return subset


def compute_pipeline_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    true_values = y_true.astype(str)
    pred_values = y_pred.astype(str)
    accuracy = float((true_values == pred_values).mean()) if len(true_values) else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": _balanced_accuracy(true_values, pred_values),
        "macro_f1": _macro_f1(true_values, pred_values),
        "weighted_f1": _weighted_f1(true_values, pred_values),
        "mcc": 0.0,
    }


__all__.extend(
    [
        "prepare_stage1_labels",
        "prepare_stage2_frame",
        "compute_pipeline_metrics",
    ]
)
