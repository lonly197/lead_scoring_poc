"""
OHAB/HAB 训练运行时配置解析。
"""

from __future__ import annotations

import math
import os
from typing import Any


DEFAULT_MEMORY_HEAVY_MODELS = ["RF", "XT", "KNN", "FASTAI", "NN_TORCH"]
LEGACY_MEMORY_HEAVY_MODELS = ["RF", "XT", "KNN"]
BASELINE_FAMILY_ALIASES = {
    "gbm": "gbm",
    "gbdt": "gbm",
    "lightgbm": "gbm",
    "cat": "cat",
    "xgb": "xgb",
    "auto": "auto",
}

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "server_16g_compare": {
        "preset": "good_quality",
        "time_limit": 5400,
        "num_bag_folds": 3,
        "label_mode": "hab",
        "enable_model_comparison": True,
        "baseline_family": "gbm",
        "memory_limit_gb": None,  # 自动根据可用内存计算
        "fit_strategy": "sequential",
        "excluded_model_types": DEFAULT_MEMORY_HEAVY_MODELS,
        "num_folds_parallel": 1,
        "max_memory_ratio": 0.7,  # 单模型内存上限
    },
    "server_16g_fast": {
        "preset": "medium_quality",
        "time_limit": 1800,
        "num_bag_folds": 0,
        "label_mode": "hab",
        "enable_model_comparison": False,
        "baseline_family": "gbm",
        "memory_limit_gb": None,
        "fit_strategy": "sequential",
        "excluded_model_types": DEFAULT_MEMORY_HEAVY_MODELS,
        "num_folds_parallel": 1,
        "max_memory_ratio": 0.7,
    },
    "server_16g_probe_nn_torch": {
        "preset": "good_quality",
        "time_limit": 7200,
        "num_bag_folds": 0,
        "label_mode": "hab",
        "enable_model_comparison": False,
        "baseline_family": "gbm",
        "memory_limit_gb": None,
        "fit_strategy": "sequential",
        "excluded_model_types": ["RF", "XT", "KNN", "FASTAI"],
        "num_folds_parallel": 1,
        "max_memory_ratio": 0.7,
    },
    "lab_full_quality": {
        "preset": "high_quality",
        "time_limit": 3600,
        "num_bag_folds": 5,
        "label_mode": "hab",
        "enable_model_comparison": True,
        "baseline_family": "gbm",
        "memory_limit_gb": None,
        "fit_strategy": "parallel",
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


def normalize_baseline_family(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = BASELINE_FAMILY_ALIASES.get(value.strip().lower())
    if normalized is None:
        valid_values = ", ".join(sorted(BASELINE_FAMILY_ALIASES))
        raise ValueError(f"未知 baseline_family: {value}，可选值: {valid_values}")
    return normalized


def detect_system_resources() -> dict[str, Any]:
    """探测当前机器的 CPU 与内存可用情况。"""
    cpu_count = os.cpu_count() or 1

    total_bytes = None
    available_bytes = None

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_bytes = int(vm.total)
        available_bytes = int(vm.available)
    except Exception:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total_bytes = int(page_size * total_pages)
            available_bytes = int(page_size * avail_pages)
        except (ValueError, OSError, AttributeError):
            total_bytes = None
            available_bytes = None

    total_gb = round(total_bytes / (1024**3), 2) if total_bytes is not None else None
    available_gb = round(available_bytes / (1024**3), 2) if available_bytes is not None else None

    return {
        "cpu_count": cpu_count,
        "total_memory_gb": total_gb,
        "available_memory_gb": available_gb,
    }


def _round_down_half(value: float) -> float:
    return math.floor(value * 2) / 2


def _derive_memory_limit_gb(
    profile_limit_gb: float | None,
    available_memory_gb: float | None,
) -> float | None:
    """
    推导 AutoGluon 内存软限制。

    策略：
    - 预留 2-3GB 给系统和其他进程
    - 使用可用内存的 70% 作为软限制
    - 最小 4GB
    """
    if available_memory_gb is None:
        return profile_limit_gb

    # 预留内存：小机器预留 2GB，大机器预留 3GB
    reserve_gb = 2.5 if available_memory_gb >= 12 else 2.0

    # 使用可用内存的 70%，但确保至少 4GB
    derived_limit_gb = max(4.0, _round_down_half((available_memory_gb - reserve_gb) * 0.7))

    if profile_limit_gb is None:
        return derived_limit_gb
    return max(4.0, min(profile_limit_gb, derived_limit_gb))


def _derive_num_folds_parallel(
    requested_num_bag_folds: int,
    cpu_count: int,
    available_memory_gb: float | None,
) -> int | None:
    """
    根据系统资源推导合理的并行 fold 数量。

    策略：
    - CPU 核心数 >= 4 且可用内存 >= 6GB：启用 2 并行
    - CPU 核心数 >= 8 且可用内存 >= 10GB：启用 3 并行
    - 其他情况：保持串行（1）
    """
    if requested_num_bag_folds <= 1:
        return 1
    if cpu_count <= 2:
        return 1
    if available_memory_gb is None:
        return 1

    # 更激进的并行策略
    if cpu_count >= 8 and available_memory_gb >= 10:
        return min(3, requested_num_bag_folds)
    elif cpu_count >= 4 and available_memory_gb >= 6:
        return min(2, requested_num_bag_folds)
    else:
        return 1


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

    detected_resources = detect_system_resources()
    cpu_count = detected_resources["cpu_count"]
    available_memory_gb = detected_resources["available_memory_gb"]

    explicit_memory_limit = _coalesce(
        getattr(args, "memory_limit_gb", None),
        _env("OHAB_MEMORY_LIMIT_GB"),
    )
    explicit_num_folds_parallel = _coalesce(
        getattr(args, "num_folds_parallel", None),
        _env("OHAB_NUM_FOLDS_PARALLEL"),
    )

    num_bag_folds = int(
        _coalesce(
            getattr(args, "num_bag_folds", None),
            _env("OHAB_NUM_BAG_FOLDS"),
            profile.get("num_bag_folds"),
            3,
        )
    )
    derived_memory_limit_gb = _derive_memory_limit_gb(profile.get("memory_limit_gb"), available_memory_gb)
    derived_num_folds_parallel = _derive_num_folds_parallel(num_bag_folds, cpu_count, available_memory_gb)

    memory_limit_gb = (
        float(explicit_memory_limit)
        if explicit_memory_limit is not None
        else derived_memory_limit_gb
    )
    num_folds_parallel = (
        int(explicit_num_folds_parallel)
        if explicit_num_folds_parallel is not None
        else (
            derived_num_folds_parallel
            if profile.get("num_folds_parallel") is None
            else min(profile.get("num_folds_parallel"), derived_num_folds_parallel)
            if derived_num_folds_parallel is not None
            else profile.get("num_folds_parallel")
        )
    )

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
        "num_bag_folds": num_bag_folds,
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
            normalize_baseline_family(getattr(args, "baseline_family", None)),
            normalize_baseline_family(_env("OHAB_BASELINE_FAMILY")),
            normalize_baseline_family(profile.get("baseline_family")),
            normalize_baseline_family("gbm"),
        ),
        "memory_limit_gb": memory_limit_gb,
        "fit_strategy": _coalesce(
            getattr(args, "fit_strategy", None),
            _env("OHAB_FIT_STRATEGY"),
            profile.get("fit_strategy"),
            "sequential",
        ),
        "excluded_model_types": excluded_model_types,
        "num_folds_parallel": num_folds_parallel,
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
        "detected_resources": detected_resources,
        "resource_tuning": {
            "memory_limit_source": "manual" if explicit_memory_limit is not None else "auto",
            "num_folds_parallel_source": "manual" if explicit_num_folds_parallel is not None else "auto",
            "derived_memory_limit_gb": derived_memory_limit_gb,
            "derived_num_folds_parallel": derived_num_folds_parallel,
        },
    }
