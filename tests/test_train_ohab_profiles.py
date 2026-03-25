import types

from src.training import ohab_runtime
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

    monkeypatch.setattr(
        ohab_runtime,
        "detect_system_resources",
        lambda: {"cpu_count": 8, "total_memory_gb": 15.45, "available_memory_gb": 14.2},
    )

    resolved = resolve_training_config(_make_args())

    assert resolved["training_profile"] == "server_16g_compare"
    assert resolved["preset"] == "good_quality"
    assert resolved["time_limit"] == 5400
    assert resolved["num_bag_folds"] == 3
    assert resolved["enable_model_comparison"] is True
    assert resolved["baseline_family"] == "gbm"
    assert resolved["memory_limit_gb"] == 8.0
    assert resolved["fit_strategy"] == "sequential"
    assert resolved["num_folds_parallel"] == 1
    assert resolved["excluded_model_types"] == ["RF", "XT", "KNN", "FASTAI", "NN_TORCH"]
    assert resolved["resource_tuning"]["memory_limit_source"] == "auto"
    assert resolved["resource_tuning"]["num_folds_parallel_source"] == "auto"


def test_resolve_training_config_allows_env_and_cli_override(monkeypatch):
    monkeypatch.setenv("OHAB_MODEL_PRESET", "medium_quality")
    monkeypatch.setenv("OHAB_NUM_BAG_FOLDS", "2")
    monkeypatch.setenv("OHAB_MEMORY_LIMIT_GB", "10")

    monkeypatch.setattr(
        ohab_runtime,
        "detect_system_resources",
        lambda: {"cpu_count": 32, "total_memory_gb": 64.0, "available_memory_gb": 52.0},
    )

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
    assert resolved["resource_tuning"]["memory_limit_source"] == "manual"
    assert resolved["resource_tuning"]["num_folds_parallel_source"] == "auto"


def test_resolve_training_config_auto_tunes_memory_limit_from_available_memory(monkeypatch):
    monkeypatch.delenv("OHAB_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("OHAB_NUM_FOLDS_PARALLEL", raising=False)
    monkeypatch.setattr(
        ohab_runtime,
        "detect_system_resources",
        lambda: {"cpu_count": 8, "total_memory_gb": 15.45, "available_memory_gb": 11.3},
    )

    resolved = resolve_training_config(_make_args())

    assert resolved["memory_limit_gb"] == 6.5
    assert resolved["num_folds_parallel"] == 1
    assert resolved["resource_tuning"]["derived_memory_limit_gb"] == 6.5


def test_resolve_training_config_probe_profile_restores_only_nn_torch(monkeypatch):
    monkeypatch.delenv("OHAB_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("OHAB_NUM_FOLDS_PARALLEL", raising=False)
    monkeypatch.setattr(
        ohab_runtime,
        "detect_system_resources",
        lambda: {"cpu_count": 8, "total_memory_gb": 15.45, "available_memory_gb": 10.96},
    )

    resolved = resolve_training_config(_make_args(training_profile="server_16g_probe_nn_torch"))

    assert resolved["training_profile"] == "server_16g_probe_nn_torch"
    assert resolved["preset"] == "good_quality"
    assert resolved["time_limit"] == 7200
    assert resolved["num_bag_folds"] == 0
    assert resolved["enable_model_comparison"] is False
    assert resolved["fit_strategy"] == "sequential"
    assert resolved["num_folds_parallel"] == 1
    assert resolved["excluded_model_types"] == ["RF", "XT", "KNN", "FASTAI"]
