from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_prep_cache_key(
    *,
    data_path: Path,
    target_label: str,
    schema_version: str,
    split_mode: str,
    split_group_mode: str,
    label_mode: str,
    feature_profile: str,
    random_seed: int,
    excluded_columns_version: str,
    feature_pipeline_version: str,
) -> str:
    stat = data_path.stat()
    payload = {
        "data_path": str(data_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "target_label": target_label,
        "schema_version": schema_version,
        "split_mode": split_mode,
        "split_group_mode": split_group_mode,
        "label_mode": label_mode,
        "feature_profile": feature_profile,
        "random_seed": random_seed,
        "excluded_columns_version": excluded_columns_version,
        "feature_pipeline_version": feature_pipeline_version,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


class PrepCacheManager:
    """训练前数据准备缓存。"""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root

    def _cache_dir(self, cache_key: str) -> Path:
        return self.cache_root / cache_key

    def load(self, cache_key: str) -> dict[str, Any] | None:
        cache_dir = self._cache_dir(cache_key)
        required_files = {
            "train_df": cache_dir / "train_df.parquet",
            "valid_df": cache_dir / "valid_df.parquet",
            "test_df": cache_dir / "test_df.parquet",
            "metadata": cache_dir / "cache_metadata.json",
        }
        if not all(path.exists() for path in required_files.values()):
            return None

        metadata = json.loads(required_files["metadata"].read_text(encoding="utf-8"))
        return {
            "train_df": pd.read_parquet(required_files["train_df"]),
            "valid_df": pd.read_parquet(required_files["valid_df"]),
            "test_df": pd.read_parquet(required_files["test_df"]),
            "metadata": metadata,
        }

    def save(self, cache_key: str, payload: dict[str, Any]) -> Path:
        cache_dir = self._cache_dir(cache_key)
        cache_dir.mkdir(parents=True, exist_ok=True)

        payload["train_df"].to_parquet(cache_dir / "train_df.parquet", index=False)
        payload["valid_df"].to_parquet(cache_dir / "valid_df.parquet", index=False)
        payload["test_df"].to_parquet(cache_dir / "test_df.parquet", index=False)
        with open(cache_dir / "cache_metadata.json", "w", encoding="utf-8") as f:
            json.dump(payload["metadata"], f, ensure_ascii=False, indent=2, default=str)
        return cache_dir
