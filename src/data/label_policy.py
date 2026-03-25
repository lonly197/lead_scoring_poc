"""OHAB 标签策略工具。"""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


OHAB_LABEL_ORDER = ["O", "H", "A", "B"]


def ordered_ohab_labels(labels: Iterable[str]) -> List[str]:
    """按业务顺序返回标签列表。"""
    label_set = {str(label) for label in labels if pd.notna(label)}
    ordered = [label for label in OHAB_LABEL_ORDER if label in label_set]
    extras = sorted(label_set.difference(OHAB_LABEL_ORDER))
    return ordered + extras


def build_ohab_label_policy(
    train_df: pd.DataFrame,
    target_label: str,
    o_merge_threshold: int = 50,
    merge_target: str = "H",
) -> Dict[str, object]:
    """基于训练集分布构建 OHAB 标签策略。"""
    raw_labels = ordered_ohab_labels(train_df[target_label].dropna().unique())
    o_count = int((train_df[target_label] == "O").sum())
    merged = 0 < o_count < o_merge_threshold
    mapping = {"O": merge_target} if merged else {}

    effective_series = train_df[target_label].replace(mapping) if mapping else train_df[target_label]
    effective_labels = ordered_ohab_labels(effective_series.dropna().unique())

    return {
        "target_label": target_label,
        "raw_classes": raw_labels,
        "effective_classes": effective_labels,
        "o_count_train": o_count,
        "o_merge_threshold": o_merge_threshold,
        "merge_target": merge_target,
        "merged": merged,
        "mapping": mapping,
    }


def apply_ohab_label_policy(
    df: pd.DataFrame,
    target_label: str,
    label_policy: Dict[str, object],
) -> pd.DataFrame:
    """将 OHAB 标签策略应用到数据集。"""
    df = df.copy()
    mapping = label_policy.get("mapping", {}) or {}
    if mapping and target_label in df.columns:
        df[target_label] = df[target_label].replace(mapping)
    return df
