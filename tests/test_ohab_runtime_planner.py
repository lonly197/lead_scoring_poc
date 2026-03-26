from src.training.ohab_runtime import build_resource_plan


def test_build_resource_plan_flags_large_text_heavy_two_stage_job_for_degrade():
    runtime_config = {
        "training_profile": "server_16g_compare",
        "preset": "good_quality",
        "time_limit": 5400,
        "eval_metric": "balanced_accuracy",
        "num_bag_folds": 3,
        "label_mode": "hab",
        "enable_model_comparison": True,
        "baseline_family": "gbm",
        "memory_limit_gb": 6.0,
        "fit_strategy": "sequential",
        "excluded_model_types": ["RF", "XT", "KNN", "FASTAI", "NN_TORCH"],
        "num_folds_parallel": 1,
        "max_memory_ratio": 0.7,
        "pipeline_mode": "two_stage",
        "detected_resources": {"available_memory_gb": 10.93},
    }

    plan = build_resource_plan(
        runtime_config,
        dataset_profile={
            "train_rows": 333826,
            "feature_count": 57,
            "train_memory_mb": 520.0,
            "text_feature_count": 1,
        },
    )

    assert plan["should_degrade"] is True
    assert plan["effective_num_bag_folds"] == 0
    assert plan["effective_preset"] == "medium_quality"
    assert plan["effective_enable_model_comparison"] is False
    assert "memory_risk_high" in plan["reasons"]


def test_build_resource_plan_keeps_safer_single_stage_job_unchanged():
    runtime_config = {
        "training_profile": "server_16g_fast",
        "preset": "medium_quality",
        "time_limit": 1800,
        "eval_metric": "log_loss",
        "num_bag_folds": 0,
        "label_mode": "hab",
        "enable_model_comparison": False,
        "baseline_family": "gbm",
        "memory_limit_gb": 8.5,
        "fit_strategy": "sequential",
        "excluded_model_types": ["RF", "XT", "KNN", "FASTAI", "NN_TORCH"],
        "num_folds_parallel": 1,
        "max_memory_ratio": 0.7,
        "pipeline_mode": "single_stage",
        "detected_resources": {"available_memory_gb": 14.0},
    }

    plan = build_resource_plan(
        runtime_config,
        dataset_profile={
            "train_rows": 80000,
            "feature_count": 45,
            "train_memory_mb": 120.0,
            "text_feature_count": 0,
        },
    )

    assert plan["should_degrade"] is False
    assert plan["effective_num_bag_folds"] == 0
    assert plan["effective_preset"] == "medium_quality"
    assert plan["effective_enable_model_comparison"] is False
