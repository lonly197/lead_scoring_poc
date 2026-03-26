from __future__ import annotations

from typing import Any

import pandas as pd


WEAK_SEMANTIC_COLUMNS = {
    "跟进时间_备用",
    "跟进内容_备用",
    "跟进结果_备用",
    "跟进备注_备用",
    "备用字段_43",
}

MISSING_INDICATOR_COLUMNS = {
    "客户是否主动询问交车时间",
    "客户是否主动询问购车权益",
    "客户是否主动询问金融政策",
    "客户是否同意加微信",
    "客户是否表示门店距离太远拒绝到店",
}


def screen_features(
    df: pd.DataFrame,
    target_label: str,
    high_missing_threshold: float = 0.95,
    indicator_missing_threshold: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """根据当前数据集质量自动筛选特征字段。"""
    screened_df = df.copy()
    dropped_high_missing: list[str] = []
    dropped_constant: list[str] = []
    dropped_weak_semantic: list[str] = []
    added_missing_indicators: list[str] = []

    for column in list(screened_df.columns):
        if column == target_label:
            continue
        if column in WEAK_SEMANTIC_COLUMNS:
            screened_df = screened_df.drop(columns=[column])
            dropped_weak_semantic.append(column)
            continue

        missing_ratio = float(screened_df[column].isna().mean())
        nunique = int(screened_df[column].nunique(dropna=True))

        if column in MISSING_INDICATOR_COLUMNS and missing_ratio >= indicator_missing_threshold:
            indicator_column = f"{column}_缺失"
            screened_df[indicator_column] = screened_df[column].isna().astype(int)
            added_missing_indicators.append(indicator_column)

        if missing_ratio > high_missing_threshold:
            screened_df = screened_df.drop(columns=[column])
            dropped_high_missing.append(column)
            continue

        if nunique <= 1:
            screened_df = screened_df.drop(columns=[column])
            dropped_constant.append(column)

    report = {
        "dropped_high_missing": dropped_high_missing,
        "dropped_constant": dropped_constant,
        "dropped_weak_semantic": dropped_weak_semantic,
        "added_missing_indicators": added_missing_indicators,
    }
    return screened_df, report


def apply_screening_report(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    screened_df = df.copy()
    for column in report.get("dropped_high_missing", []):
        if column in screened_df.columns:
            screened_df = screened_df.drop(columns=[column])
    for column in report.get("dropped_constant", []):
        if column in screened_df.columns:
            screened_df = screened_df.drop(columns=[column])
    for column in report.get("dropped_weak_semantic", []):
        if column in screened_df.columns:
            screened_df = screened_df.drop(columns=[column])
    for indicator_column in report.get("added_missing_indicators", []):
        source_column = indicator_column.removesuffix("_缺失")
        if source_column in screened_df.columns and indicator_column not in screened_df.columns:
            screened_df[indicator_column] = screened_df[source_column].isna().astype(int)
    return screened_df
