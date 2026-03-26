from __future__ import annotations

import pandas as pd

from src.data.loader import smart_split_data


def test_smart_split_data_random_group_mode_keeps_phone_groups_disjoint():
    df = pd.DataFrame(
        {
            "线索唯一ID": [f"L{i}" for i in range(12)],
            "手机号_脱敏": [
                "13800000000",
                "13800000000",
                "13900000000",
                "13900000000",
                "",
                "",
                None,
                None,
                "13700000000",
                "13700000000",
                "13600000000",
                "13600000000",
            ],
            "线索创建时间": pd.date_range("2026-02-01", periods=12, freq="D"),
            "线索评级_试驾前": ["H", "A", "B", "H", "A", "B", "H", "A", "B", "H", "A", "B"],
        }
    )

    train_df, valid_df, test_df, split_info = smart_split_data(
        df,
        target_label="线索评级_试驾前",
        split_mode="random",
        group_by="phone_or_lead",
        random_seed=7,
    )

    train_groups = set(train_df["split_group_key"])
    valid_groups = set(valid_df["split_group_key"])
    test_groups = set(test_df["split_group_key"])

    assert split_info["mode"] == "random"
    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)
    assert split_info["split_group_mode"] == "phone_or_lead"
    assert split_info["group_counts"]["test"] > 0
    assert "train_group_keys" not in split_info
    assert "valid_group_keys" not in split_info
    assert "test_group_keys" not in split_info
    assert split_info["group_key_fingerprint"]
