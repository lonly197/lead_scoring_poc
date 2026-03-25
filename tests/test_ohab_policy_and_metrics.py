import pandas as pd

from src.data.label_policy import (
    apply_ohab_label_policy,
    build_ohab_label_policy,
    filter_to_effective_ohab_labels,
)
from src.evaluation.ohab_metrics import (
    apply_hab_decision_policy,
    compute_class_ranking_report,
    compute_hab_bucket_summary,
    compute_threshold_report,
    optimize_hab_decision_policy,
)


def test_build_ohab_label_policy_merges_o_based_on_training_distribution():
    train_df = pd.DataFrame({"线索评级_试驾前": ["O", "H", "A", "B"]})
    score_df = pd.DataFrame({"线索评级_试驾前": ["O", "B"]})

    policy = build_ohab_label_policy(
        train_df,
        target_label="线索评级_试驾前",
        o_merge_threshold=2,
        merge_target="H",
    )
    transformed = apply_ohab_label_policy(score_df, "线索评级_试驾前", policy)

    assert policy["merged"] is True
    assert policy["mapping"]["O"] == "H"
    assert policy["o_count_train"] == 1
    assert transformed["线索评级_试驾前"].tolist() == ["H", "B"]


def test_compute_class_ranking_and_b_threshold_reports():
    y_true = pd.Series(["H", "A", "B", "B"])
    proba = pd.DataFrame(
        {
            "H": [0.90, 0.10, 0.10, 0.20],
            "A": [0.05, 0.80, 0.20, 0.10],
            "B": [0.05, 0.10, 0.70, 0.70],
        }
    )

    ranking_report = compute_class_ranking_report(y_true, proba, top_ratios=(0.5,))
    threshold_report = compute_threshold_report(
        y_true,
        proba["B"],
        positive_label="B",
        thresholds=[0.5],
    )

    b_ranking = next(item for item in ranking_report if item["class"] == "B")
    assert b_ranking["lift"] == 2.0
    assert b_ranking["hit_rate"] == 1.0
    assert threshold_report.loc[0, "precision"] == 1.0
    assert threshold_report.loc[0, "recall"] == 1.0


def test_build_hab_label_policy_filters_out_o_class():
    train_df = pd.DataFrame({"线索评级_试驾前": ["O", "H", "A", "B", "H"]})
    score_df = pd.DataFrame({"线索评级_试驾前": ["O", "H", "B"]})

    policy = build_ohab_label_policy(
        train_df,
        target_label="线索评级_试驾前",
        label_mode="hab",
    )
    transformed = apply_ohab_label_policy(score_df, "线索评级_试驾前", policy)
    filtered = filter_to_effective_ohab_labels(transformed, "线索评级_试驾前", policy)

    assert policy["label_mode"] == "hab"
    assert policy["effective_classes"] == ["H", "A", "B"]
    assert policy["excluded_classes"] == ["O"]
    assert filtered["线索评级_试驾前"].tolist() == ["H", "B"]


def test_hab_decision_policy_and_bucket_summary():
    y_true = pd.Series(["H", "A", "B", "B", "A", "H"])
    proba = pd.DataFrame(
        {
            "H": [0.80, 0.20, 0.10, 0.15, 0.45, 0.70],
            "A": [0.15, 0.50, 0.25, 0.20, 0.40, 0.20],
            "B": [0.05, 0.30, 0.65, 0.70, 0.15, 0.10],
        }
    )

    policy = optimize_hab_decision_policy(y_true, proba, h_thresholds=[0.5], b_thresholds=[0.3], min_predicted_b_rate=0.2)
    y_pred = apply_hab_decision_policy(proba, policy)
    summary = compute_hab_bucket_summary(
        pd.DataFrame(
            {
                "预测标签": y_pred,
                "到店标签_14天": [1, 1, 0, 0, 1, 1],
                "试驾标签_14天": [1, 0, 0, 0, 0, 1],
            }
        )
    )

    assert policy["strategy"] == "hab_threshold"
    assert set(y_pred.unique()) == {"H", "A", "B"}
    assert "试驾标签_14天_rate" in summary.columns
