"""
OHAB/HAB 训练运行时配置解析。
"""

from __future__ import annotations

import os
from typing import Any


DEFAULT_MEMORY_HEAVY_MODELS = ["RF", "XT", "KNN", "FASTAI", "NN_TORCH"]
LEGACY_MEMORY_HEAVY_MODELS = ["RF", "XT", "KNN"]

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "server_16g_compare": {
        "preset": "good_quality",
        "time_limit": 5400,
        "num_bag_folds": 3,
        "label_mode": "hab",
        "enable_model_comparison": True,
        "baseline_family": "gbm",
        "memory_limit_gb": 12.0,
        "fit_strategy": "sequential",
        "excluded_model_types": DEFAULT_MEMORY_HEAVY_MODELS,
        "num_folds_parallel": 1,
        "max_memory_ratio": None,
    },
    "server_16g_fast": {
        "preset": "medium_quality",
        "time_limit": 1800,
        "num_bag_folds": 0,
        "label_mode": "hab",
        "enable_model_comparison": False,
        "baseline_family": "gbm",
        "memory_limit_gb": 10.0,
        "fit_strategy": "sequential",
        "excluded_model_types": DEFAULT_MEMORY_HEAVY_MODELS,
        "num_folds_parallel": 1,
        "max_memory_ratio": None,
    },
    "lab_full_quality": {
        "preset": "high_quality",
        "time_limit": 3600,
        "num_bag_folds": 5,
        "label_mode": "hab",
        "enable_model_comparison": True,
        "baseline_family": "gbm",
        "memory_limit_gb": None,
        "fit_strategy": "sequential",
        "excluded_model_types": None,
        "num_folds_parallel": None,
        "max_memory_ratio": None,
    },
}


def _env(key: str) -> str | None:
    value = os.getenv(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(key: str) -> bool | None:
    value = _env(key)
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"环境变量 {key} 不是合法布尔值: {value}")


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _merge_model_types(base: list[str] | None, extra: list[str] | None) -> list[str] | None:
    merged: list[str] = []
    for source in (base or [], extra or []):
        if source not in merged:
            merged.append(source)
    return merged or None


def resolve_training_config(args) -> dict[str, Any]:
    profile_name = _coalesce(
        getattr(args, "training_profile", None),
        _env("OHAB_TRAINING_PROFILE"),
        "server_16g_compare",
    )
    if profile_name not in TRAINING_PROFILES:
        raise ValueError(f"未知训练档位: {profile_name}")
    profile = TRAINING_PROFILES[profile_name]

    excluded_model_types = _coalesce(
        _parse_csv_list(getattr(args, "excluded_model_types", None)),
        _parse_csv_list(_env("OHAB_EXCLUDED_MODEL_TYPES")),
        profile.get("excluded_model_types"),
    )
    if getattr(args, "exclude_memory_heavy_models", None):
        excluded_model_types = _merge_model_types(excluded_model_types, LEGACY_MEMORY_HEAVY_MODELS)

    return {
        "training_profile": profile_name,
        "preset": _coalesce(
            getattr(args, "preset", None),
            _env("OHAB_MODEL_PRESET"),
            profile.get("preset"),
            _env("MODEL_PRESET"),
            "good_quality",
        ),
        "time_limit": int(
            _coalesce(
                getattr(args, "time_limit", None),
                _env("OHAB_TIME_LIMIT"),
                profile.get("time_limit"),
                _env("TIME_LIMIT"),
                3600,
            )
        ),
        "num_bag_folds": int(
            _coalesce(
                getattr(args, "num_bag_folds", None),
                _env("OHAB_NUM_BAG_FOLDS"),
                profile.get("num_bag_folds"),
                3,
            )
        ),
        "label_mode": _coalesce(
            getattr(args, "label_mode", None),
            _env("OHAB_LABEL_MODE"),
            profile.get("label_mode"),
            "hab",
        ),
        "enable_model_comparison": bool(
            _coalesce(
                getattr(args, "enable_model_comparison", None),
                _env_bool("OHAB_ENABLE_MODEL_COMPARISON"),
                profile.get("enable_model_comparison"),
                False,
            )
        ),
        "baseline_family": _coalesce(
            getattr(args, "baseline_family", None),
            _env("OHAB_BASELINE_FAMILY"),
            profile.get("baseline_family"),
            "gbm",
        ),
        "memory_limit_gb": (
            float(
                _coalesce(
                    getattr(args, "memory_limit_gb", None),
                    _env("OHAB_MEMORY_LIMIT_GB"),
                    profile.get("memory_limit_gb"),
                )
            )
            if _coalesce(
                getattr(args, "memory_limit_gb", None),
                _env("OHAB_MEMORY_LIMIT_GB"),
                profile.get("memory_limit_gb"),
            )
            is not None
            else None
        ),
        "fit_strategy": _coalesce(
            getattr(args, "fit_strategy", None),
            _env("OHAB_FIT_STRATEGY"),
            profile.get("fit_strategy"),
            "sequential",
        ),
        "excluded_model_types": excluded_model_types,
        "num_folds_parallel": (
            int(
                _coalesce(
                    getattr(args, "num_folds_parallel", None),
                    _env("OHAB_NUM_FOLDS_PARALLEL"),
                    profile.get("num_folds_parallel"),
                )
            )
            if _coalesce(
                getattr(args, "num_folds_parallel", None),
                _env("OHAB_NUM_FOLDS_PARALLEL"),
                profile.get("num_folds_parallel"),
            )
            is not None
            else None
        ),
        "max_memory_ratio": (
            float(
                _coalesce(
                    getattr(args, "max_memory_ratio", None),
                    _env("OHAB_MAX_MEMORY_RATIO"),
                    profile.get("max_memory_ratio"),
                )
            )
            if _coalesce(
                getattr(args, "max_memory_ratio", None),
                _env("OHAB_MAX_MEMORY_RATIO"),
                profile.get("max_memory_ratio"),
            )
            is not None
            else None
        ),
    }
