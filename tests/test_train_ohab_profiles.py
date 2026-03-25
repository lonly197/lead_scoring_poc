import types

from src.training.ohab_runtime import resolve_training_config


def _make_args(**overrides):
    defaults = dict(
        training_profile=None,
        preset=None,
        time_limit=None,
        num_bag_folds=None,
        label_mode=None,
        enable_model_comparison=None,
        baseline_family=None,
        memory_limit_gb=None,
        fit_strategy=None,
        excluded_model_types=None,
        exclude_memory_heavy_models=None,
        num_folds_parallel=None,
        max_memory_ratio=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_resolve_training_config_uses_server_16g_compare_defaults(monkeypatch):
    monkeypatch.delenv("OHAB_TRAINING_PROFILE", raising=False)
    monkeypatch.delenv("OHAB_MODEL_PRESET", raising=False)
    monkeypatch.delenv("OHAB_TIME_LIMIT", raising=False)
    monkeypatch.delenv("OHAB_NUM_BAG_FOLDS", raising=False)
    monkeypatch.delenv("OHAB_ENABLE_MODEL_COMPARISON", raising=False)
    monkeypatch.delenv("OHAB_BASELINE_FAMILY", raising=False)
    monkeypatch.delenv("OHAB_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("OHAB_FIT_STRATEGY", raising=False)
    monkeypatch.delenv("OHAB_EXCLUDED_MODEL_TYPES", raising=False)
    monkeypatch.delenv("OHAB_NUM_FOLDS_PARALLEL", raising=False)

    resolved = resolve_training_config(_make_args())

    assert resolved["training_profile"] == "server_16g_compare"
    assert resolved["preset"] == "good_quality"
    assert resolved["time_limit"] == 5400
    assert resolved["num_bag_folds"] == 3
    assert resolved["enable_model_comparison"] is True
    assert resolved["baseline_family"] == "gbm"
    assert resolved["memory_limit_gb"] == 12.0
    assert resolved["fit_strategy"] == "sequential"
    assert resolved["num_folds_parallel"] == 1
    assert resolved["excluded_model_types"] == ["RF", "XT", "KNN", "FASTAI", "NN_TORCH"]


def test_resolve_training_config_allows_env_and_cli_override(monkeypatch):
    monkeypatch.setenv("OHAB_MODEL_PRESET", "medium_quality")
    monkeypatch.setenv("OHAB_NUM_BAG_FOLDS", "2")
    monkeypatch.setenv("OHAB_MEMORY_LIMIT_GB", "10")

    resolved = resolve_training_config(
        _make_args(
            preset="high_quality",
            num_bag_folds=4,
            memory_limit_gb=14,
            fit_strategy="parallel",
            excluded_model_types="RF,XT",
            enable_model_comparison=True,
            baseline_family="cat",
        )
    )

    assert resolved["preset"] == "high_quality"
    assert resolved["num_bag_folds"] == 4
    assert resolved["memory_limit_gb"] == 14.0
    assert resolved["fit_strategy"] == "parallel"
    assert resolved["excluded_model_types"] == ["RF", "XT"]
    assert resolved["enable_model_comparison"] is True
    assert resolved["baseline_family"] == "cat"
