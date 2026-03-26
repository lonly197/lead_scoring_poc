from __future__ import annotations

from typing import Iterable

import pandas as pd


SCORECARD_DIMENSIONS = {
    "基础特征": {
        "weight": 0.20,
        "fields": ("一级渠道名称", "二级渠道名称", "三级渠道名称", "四级渠道名称", "所在城市", "首触意向车型", "预算区间", "线索类型"),
    },
    "画像特征": {
        "weight": 0.10,
        "fields": ("客户性别", "历史订单次数", "历史到店次数", "历史试驾次数"),
    },
    "行为特征": {
        "weight": 0.25,
        "fields": ("通话次数", "通话总时长", "平均通话时长_派生", "是否接通", "有效通话", "跟进总次数", "接通次数"),
    },
    "时序特征": {
        "weight": 0.20,
        "fields": ("首触响应时长_小时", "首触线索是否及时外呼", "响应及时性", "首触线索当天是否联通实体卡外呼"),
    },
    "意图特征": {
        "weight": 0.25,
        "fields": ("客户是否主动询问交车时间", "客户是否主动询问购车权益", "客户是否主动询问金融政策", "客户是否同意加微信", "提及价格", "提及试驾", "提及到店"),
    },
}


def _to_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _truthy(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    values = df[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).gt(0).astype(float)
    normalized = values.astype("string").fillna("").str.strip()
    return normalized.isin({"1", "true", "True", "是", "有", "已支付", "支付成功", "已转定金"}).astype(float)


def _present(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    values = df[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).ne(0).astype(float)
    return values.astype("string").fillna("").str.strip().ne("").astype(float)


def _clip_ratio(series: pd.Series, denominator: float) -> pd.Series:
    if denominator <= 0:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return (series / denominator).clip(lower=0.0, upper=1.0)


def _build_dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    channel_present = pd.concat([_present(df, column) for column in ("一级渠道名称", "二级渠道名称", "三级渠道名称", "四级渠道名称")], axis=1).max(axis=1)
    basic = (
        0.30 * channel_present
        + 0.25 * _present(df, "首触意向车型")
        + 0.20 * _present(df, "所在城市")
        + 0.15 * _present(df, "预算区间")
        + 0.10 * _present(df, "线索类型")
    ) * 100

    profile_signal = (
        0.30 * _present(df, "客户性别")
        + 0.35 * _clip_ratio(_to_numeric(df, "历史到店次数") + _to_numeric(df, "历史试驾次数"), 2)
        + 0.35 * _clip_ratio(_to_numeric(df, "历史订单次数"), 1)
    ) * 100

    behavior = (
        0.15 * _clip_ratio(_to_numeric(df, "通话次数"), 4)
        + 0.20 * _clip_ratio(_to_numeric(df, "通话总时长"), 900)
        + 0.15 * _clip_ratio(_to_numeric(df, "平均通话时长_派生"), 300)
        + 0.15 * _truthy(df, "是否接通")
        + 0.15 * _truthy(df, "有效通话")
        + 0.20 * _clip_ratio(_to_numeric(df, "跟进总次数") + _to_numeric(df, "接通次数"), 5)
    ) * 100

    response_hours = _to_numeric(df, "首触响应时长_小时")
    response_score = (1.0 - (response_hours / 72.0)).clip(lower=0.0, upper=1.0)
    timing = (
        0.45 * response_score
        + 0.25 * _truthy(df, "首触线索是否及时外呼")
        + 0.15 * _truthy(df, "响应及时性")
        + 0.15 * _truthy(df, "首触线索当天是否联通实体卡外呼")
    ) * 100

    intent_columns = (
        "客户是否主动询问交车时间",
        "客户是否主动询问购车权益",
        "客户是否主动询问金融政策",
        "客户是否同意加微信",
        "提及价格",
        "提及试驾",
        "提及到店",
    )
    intent_hits = pd.concat([_truthy(df, column) for column in intent_columns], axis=1).sum(axis=1)
    intent = _clip_ratio(intent_hits, len(intent_columns)) * 100

    return pd.DataFrame(
        {
            "基础特征得分": basic.round(4),
            "画像特征得分": profile_signal.round(4),
            "行为特征得分": behavior.round(4),
            "时序特征得分": timing.round(4),
            "意图特征得分": intent.round(4),
        },
        index=df.index,
    )


def _score_to_label(total_score: pd.Series) -> pd.Series:
    return pd.Series(
        ["H" if score >= 75 else "A" if score >= 60 else "B" for score in total_score],
        index=total_score.index,
        name="预测标签",
    )


def build_trimmed_scorecard_probability_frame(total_score: Iterable[float]) -> pd.DataFrame:
    score_series = pd.Series(total_score, dtype="float64").clip(lower=0.0, upper=100.0)
    h_score = ((score_series - 60.0) / 40.0).clip(lower=0.0, upper=1.0)
    b_score = ((75.0 - score_series) / 35.0).clip(lower=0.0, upper=1.0)
    a_centered = 1.0 - ((score_series - 67.5).abs() / 22.5)
    a_score = a_centered.clip(lower=0.0, upper=1.0) + 0.05

    proba = pd.DataFrame({"H": h_score, "A": a_score, "B": b_score}, index=score_series.index)
    row_sums = proba.sum(axis=1).replace(0.0, 1.0)
    return proba.div(row_sums, axis=0)


def score_trimmed_hab_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    dimension_scores = _build_dimension_scores(df)
    total_score = (
        dimension_scores["基础特征得分"] * SCORECARD_DIMENSIONS["基础特征"]["weight"]
        + dimension_scores["画像特征得分"] * SCORECARD_DIMENSIONS["画像特征"]["weight"]
        + dimension_scores["行为特征得分"] * SCORECARD_DIMENSIONS["行为特征"]["weight"]
        + dimension_scores["时序特征得分"] * SCORECARD_DIMENSIONS["时序特征"]["weight"]
        + dimension_scores["意图特征得分"] * SCORECARD_DIMENSIONS["意图特征"]["weight"]
    ).round(4)

    available_field_count = pd.Series(0.0, index=df.index, dtype="float64")
    expected_field_count = 0
    for config in SCORECARD_DIMENSIONS.values():
        expected_field_count += len(config["fields"])
        for field_name in config["fields"]:
            available_field_count += _present(df, field_name)

    coverage = (available_field_count / expected_field_count).clip(lower=0.0, upper=1.0)
    scored = dimension_scores.copy()
    scored["总分"] = total_score
    scored["字段覆盖率"] = coverage.round(4)
    scored["预测标签"] = _score_to_label(total_score)
    return scored
