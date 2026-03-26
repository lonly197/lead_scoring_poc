import json
from pathlib import Path

import pandas as pd

from src.training.prep_cache import PrepCacheManager, build_prep_cache_key


def test_build_prep_cache_key_changes_when_inputs_change(tmp_path: Path):
    data_path = tmp_path / "sample.tsv"
    data_path.write_text("a\tb\n", encoding="utf-8")

    key1 = build_prep_cache_key(
        data_path=data_path,
        target_label="线索评级结果",
        schema_version="v2",
        split_mode="random",
        split_group_mode="phone_or_lead",
        label_mode="hab",
        feature_profile="auto_scorecard",
        random_seed=42,
        excluded_columns_version="v1",
        feature_pipeline_version="v3",
    )
    key2 = build_prep_cache_key(
        data_path=data_path,
        target_label="线索评级结果",
        schema_version="v2",
        split_mode="random",
        split_group_mode="phone_or_lead",
        label_mode="hab",
        feature_profile="auto_scorecard",
        random_seed=43,
        excluded_columns_version="v1",
        feature_pipeline_version="v3",
    )

    assert key1 != key2


def test_build_prep_cache_key_changes_when_feature_pipeline_changes(tmp_path: Path):
    data_path = tmp_path / "sample.tsv"
    data_path.write_text("a\tb\n", encoding="utf-8")

    key1 = build_prep_cache_key(
        data_path=data_path,
        target_label="线索评级结果",
        schema_version="v2",
        split_mode="random",
        split_group_mode="phone_or_lead",
        label_mode="hab",
        feature_profile="auto_scorecard",
        random_seed=42,
        excluded_columns_version="v1",
        feature_pipeline_version="v2",
    )
    key2 = build_prep_cache_key(
        data_path=data_path,
        target_label="线索评级结果",
        schema_version="v2",
        split_mode="random",
        split_group_mode="phone_or_lead",
        label_mode="hab",
        feature_profile="auto_scorecard",
        random_seed=42,
        excluded_columns_version="v1",
        feature_pipeline_version="v3",
    )

    assert key1 != key2


def test_prep_cache_manager_roundtrip(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    manager = PrepCacheManager(cache_root=cache_dir)
    cache_key = "demo-cache-key"
    payload = {
        "train_df": pd.DataFrame({"线索唯一ID": [1], "线索评级结果": ["H"]}),
        "valid_df": pd.DataFrame({"线索唯一ID": [2], "线索评级结果": ["A"]}),
        "test_df": pd.DataFrame({"线索唯一ID": [3], "线索评级结果": ["B"]}),
        "metadata": {"schema_contract": {"version": "v2"}, "target_label": "线索评级结果"},
    }

    manager.save(cache_key, payload)
    loaded = manager.load(cache_key)

    assert loaded is not None
    assert loaded["train_df"].to_dict(orient="records") == payload["train_df"].to_dict(orient="records")
    assert loaded["valid_df"].to_dict(orient="records") == payload["valid_df"].to_dict(orient="records")
    assert loaded["test_df"].to_dict(orient="records") == payload["test_df"].to_dict(orient="records")
    assert loaded["metadata"]["target_label"] == "线索评级结果"
    metadata_path = cache_dir / cache_key / "cache_metadata.json"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["target_label"] == "线索评级结果"
