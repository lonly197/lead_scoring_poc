from __future__ import annotations

import pandas as pd

from src.training.hab_pipeline import (
    build_split_group_key,
    combine_stage_predictions,
    tune_h_threshold,
)


def test_build_split_group_key_prefers_phone_and_falls_back_to_lead_id():
    df = pd.DataFrame(
        {
            "手机号_脱敏": ["13800000000", "", None, " 13900000000 "],
            "线索唯一ID": ["L1", "L2", "L3", "L4"],
        }
    )

    result = build_split_group_key(df)

    assert result.tolist() == [
        "phone::13800000000",
        "lead::L2",
        "lead::L3",
        "phone::13900000000",
    ]


def test_combine_stage_predictions_uses_h_threshold_and_routes_non_h_to_ab():
    stage1_h_proba = pd.Series([0.82, 0.49, 0.12], name="H")
    stage2_ab_proba = pd.DataFrame(
        {
            "A": [0.10, 0.72, 0.35],
            "B": [0.90, 0.28, 0.65],
        }
    )

    y_pred, y_proba = combine_stage_predictions(
        stage1_h_proba=stage1_h_proba,
        stage2_ab_proba=stage2_ab_proba,
        h_threshold=0.6,
    )

    assert y_pred.tolist() == ["H", "A", "B"]
    assert list(y_proba.columns) == ["H", "A", "B"]
    assert round(float(y_proba.iloc[0]["H"]), 2) == 0.82
    assert round(float(y_proba.iloc[1]["A"]), 2) == 0.37
    assert round(float(y_proba.iloc[2]["B"]), 2) == 0.57


def test_tune_h_threshold_prefers_balanced_accuracy_then_macro_f1():
    y_true = pd.Series(["H", "A", "B", "H", "A", "B"])
    stage1_h_proba = pd.Series([0.90, 0.55, 0.20, 0.85, 0.40, 0.10])
    stage2_ab_proba = pd.DataFrame(
        {
            "A": [0.20, 0.70, 0.30, 0.10, 0.60, 0.20],
            "B": [0.80, 0.30, 0.70, 0.90, 0.40, 0.80],
        }
    )

    result = tune_h_threshold(
        y_true=y_true,
        stage1_h_proba=stage1_h_proba,
        stage2_ab_proba=stage2_ab_proba,
        thresholds=(0.4, 0.6, 0.8),
    )

    assert result["strategy"] == "two_stage_threshold"
    assert result["h_threshold"] == 0.6
    assert result["metrics"]["balanced_accuracy"] > 0.8
