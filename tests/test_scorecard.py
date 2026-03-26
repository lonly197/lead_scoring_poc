import pandas as pd

from src.evaluation.scorecard import (
    build_trimmed_scorecard_probability_frame,
    score_trimmed_hab_scorecard,
)


def test_trimmed_scorecard_separates_strong_and_weak_leads():
    df = pd.DataFrame(
        {
            "一级渠道名称": ["官网", None],
            "所在城市": ["上海", None],
            "首触意向车型": ["Model X", None],
            "预算区间": ["30-40万", None],
            "客户性别": ["男", None],
            "历史到店次数": [1, 0],
            "历史试驾次数": [1, 0],
            "通话次数": [4, 0],
            "通话总时长": [900, 0],
            "平均通话时长_派生": [300, 0],
            "是否接通": [1, 0],
            "有效通话": [1, 0],
            "跟进总次数": [3, 0],
            "首触响应时长_小时": [0.5, 72.0],
            "首触线索是否及时外呼": [1, 0],
            "客户是否主动询问金融政策": [1, 0],
            "客户是否主动询问交车时间": [1, 0],
            "客户是否同意加微信": [1, 0],
            "提及价格": [1, 0],
            "提及试驾": [1, 0],
            "提及到店": [1, 0],
        }
    )

    scored = score_trimmed_hab_scorecard(df)

    assert scored.loc[0, "总分"] > scored.loc[1, "总分"]
    assert scored.loc[0, "预测标签"] == "H"
    assert scored.loc[1, "预测标签"] == "B"
    assert 0 < scored.loc[0, "字段覆盖率"] <= 1


def test_trimmed_scorecard_probability_frame_is_normalized():
    scores = pd.Series([82.0, 66.0, 42.0])

    proba = build_trimmed_scorecard_probability_frame(scores)

    assert list(proba.columns) == ["H", "A", "B"]
    assert proba.sum(axis=1).round(6).tolist() == [1.0, 1.0, 1.0]
    assert proba.iloc[0]["H"] > proba.iloc[0]["A"] > proba.iloc[0]["B"]
    assert proba.iloc[2]["B"] > proba.iloc[2]["A"] > proba.iloc[2]["H"]
