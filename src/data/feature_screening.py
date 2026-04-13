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


def _drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return df
    return df.drop(columns=existing)


def clean_raw_schema(
    df: pd.DataFrame,
    target_label: str,
    high_missing_threshold: float = 0.95,
    indicator_missing_threshold: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """原始字段级清洗，仅处理明显无效的原始列。

    注意：high_missing_threshold 默认 0.95（95%），比 03_clean.py 的 50% 更宽松。
    这是设计意图不同：screening 阶段只删除极端高缺失列，避免过度清洗；
    而 03_clean.py 是独立的管道清洗脚本，采用更激进的阈值。
    """
    cleaned_df = df.copy()
    dropped_high_missing: list[str] = []
    dropped_weak_semantic: list[str] = []
    added_missing_indicators: list[str] = []

    for column in list(cleaned_df.columns):
        if column == target_label:
            continue
        if column in WEAK_SEMANTIC_COLUMNS:
            cleaned_df = cleaned_df.drop(columns=[column])
            dropped_weak_semantic.append(column)
            continue

        missing_ratio = float(cleaned_df[column].isna().mean())
        if column in MISSING_INDICATOR_COLUMNS and missing_ratio >= indicator_missing_threshold:
            indicator_column = f"{column}_缺失"
            if indicator_column not in cleaned_df.columns:
                cleaned_df[indicator_column] = cleaned_df[column].isna().astype(int)
                added_missing_indicators.append(indicator_column)

        if missing_ratio > high_missing_threshold:
            cleaned_df = cleaned_df.drop(columns=[column])
            dropped_high_missing.append(column)

    report = {
        "dropped_high_missing": dropped_high_missing,
        "dropped_weak_semantic": dropped_weak_semantic,
        "added_missing_indicators": added_missing_indicators,
    }
    return cleaned_df, report


def screen_post_feature_candidates(
    df: pd.DataFrame,
    target_label: str,
    high_missing_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """派生特征生成后，再剔除常数列和高缺失列。"""
    screened_df = df.copy()
    dropped_high_missing: list[str] = []
    dropped_constant: list[str] = []

    for column in list(screened_df.columns):
        if column == target_label:
            continue

        missing_ratio = float(screened_df[column].isna().mean())
        nunique = int(screened_df[column].nunique(dropna=True))

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
    }
    return screened_df, report


def apply_raw_schema_report(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    cleaned_df = _drop_columns(df.copy(), report.get("dropped_high_missing", []))
    cleaned_df = _drop_columns(cleaned_df, report.get("dropped_weak_semantic", []))
    for indicator_column in report.get("added_missing_indicators", []):
        source_column = indicator_column.removesuffix("_缺失")
        if source_column in cleaned_df.columns and indicator_column not in cleaned_df.columns:
            cleaned_df[indicator_column] = cleaned_df[source_column].isna().astype(int)
    return cleaned_df


def apply_post_feature_screening_report(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    screened_df = _drop_columns(df.copy(), report.get("dropped_high_missing", []))
    screened_df = _drop_columns(screened_df, report.get("dropped_constant", []))
    return screened_df


def screen_features(
    df: pd.DataFrame,
    target_label: str,
    high_missing_threshold: float = 0.95,
    indicator_missing_threshold: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """兼容旧接口：先做原始清洗，再做派生后筛选。"""
    cleaned_df, raw_schema_report = clean_raw_schema(
        df,
        target_label=target_label,
        high_missing_threshold=high_missing_threshold,
        indicator_missing_threshold=indicator_missing_threshold,
    )
    screened_df, post_feature_screening_report = screen_post_feature_candidates(
        cleaned_df,
        target_label=target_label,
        high_missing_threshold=high_missing_threshold,
    )
    report = {
        "raw_schema_report": raw_schema_report,
        "post_feature_screening_report": post_feature_screening_report,
        "dropped_high_missing": sorted(
            set(raw_schema_report.get("dropped_high_missing", []))
            | set(post_feature_screening_report.get("dropped_high_missing", []))
        ),
        "dropped_constant": post_feature_screening_report.get("dropped_constant", []),
        "dropped_weak_semantic": raw_schema_report.get("dropped_weak_semantic", []),
        "added_missing_indicators": raw_schema_report.get("added_missing_indicators", []),
    }
    return screened_df, report


def apply_screening_report(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    """兼容旧接口：按新分层报告顺序应用。"""
    raw_schema_report = report.get("raw_schema_report")
    post_feature_screening_report = report.get("post_feature_screening_report")
    if raw_schema_report or post_feature_screening_report:
        screened_df = apply_raw_schema_report(df, raw_schema_report or {})
        return apply_post_feature_screening_report(screened_df, post_feature_screening_report or {})

    # 兼容旧扁平结构
    screened_df = _drop_columns(df.copy(), report.get("dropped_high_missing", []))
    screened_df = _drop_columns(screened_df, report.get("dropped_constant", []))
    screened_df = _drop_columns(screened_df, report.get("dropped_weak_semantic", []))
    for indicator_column in report.get("added_missing_indicators", []):
        source_column = indicator_column.removesuffix("_缺失")
        if source_column in screened_df.columns and indicator_column not in screened_df.columns:
            screened_df[indicator_column] = screened_df[source_column].isna().astype(int)
    return screened_df
