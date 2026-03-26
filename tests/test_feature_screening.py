from __future__ import annotations

import pandas as pd

from src.data.feature_screening import screen_features


def test_screen_features_drops_high_missing_constant_and_weak_alias_fields():
    df = pd.DataFrame(
        {
            "线索唯一ID": ["L1", "L2", "L3", "L4"],
            "线索评级_试驾前": ["H", "A", "B", "H"],
            "一级渠道名称": ["官网", "官网", "抖音", "抖音"],
            "所在城市": ["上海市", "上海市", "北京市", "北京市"],
            "预算区间": [None, None, None, None],
            "跟进内容_备用": [1, 2, 3, 4],
            "SOP开口标签": [None, None, None, None],
            "客户是否主动询问购车权益": [None, None, None, "否"],
            "常数字段": [1, 1, 1, 1],
        }
    )

    screened_df, report = screen_features(df, target_label="线索评级_试驾前")

    assert "一级渠道名称" in screened_df.columns
    assert "所在城市" in screened_df.columns
    assert "预算区间" not in screened_df.columns
    assert "SOP开口标签" not in screened_df.columns
    assert "跟进内容_备用" not in screened_df.columns
    assert "常数字段" not in screened_df.columns
    assert "客户是否主动询问购车权益_缺失" in screened_df.columns
    assert "预算区间" in report["dropped_high_missing"]
    assert "跟进内容_备用" in report["dropped_weak_semantic"]
