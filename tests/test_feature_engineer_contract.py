import pandas as pd

from src.data.loader import FeatureEngineer


def test_feature_engineer_reuses_training_city_car_context():
    train_df = pd.DataFrame(
        {
            "线索创建时间": ["2026-03-01 10:00:00", "2026-03-01 11:00:00", "2026-03-02 09:00:00"],
            "首触时间": ["2026-03-01 10:30:00", "2026-03-01 11:30:00", "2026-03-02 12:00:00"],
            "一级渠道名称": ["直营", "直营", "垂媒"],
            "二级渠道名称": ["官网", "官网", "汽车之家"],
            "所在城市": ["上海市", "上海市", "北京市"],
            "首触意向车型": ["车型A", "车型A", "车型B"],
            "通话次数": [2, 1, 1],
            "通话总时长": [120, 60, 45],
            "平均通话时长": [999.0, 888.0, 777.0],
        }
    )
    score_df = pd.DataFrame(
        {
            "线索创建时间": ["2026-03-03 10:00:00", "2026-03-03 11:00:00"],
            "首触时间": ["2026-03-03 10:10:00", "2026-03-04 11:00:00"],
            "一级渠道名称": ["直营", "垂媒"],
            "二级渠道名称": ["官网", "汽车之家"],
            "所在城市": ["上海市", "广州市"],
            "首触意向车型": ["车型A", "车型C"],
            "通话次数": [1, 0],
            "通话总时长": [80, 0],
            "平均通话时长": [123.0, 456.0],
        }
    )

    engineer = FeatureEngineer(
        time_columns=["线索创建时间", "首触时间"],
        numeric_columns=["通话次数", "通话总时长", "平均通话时长"],
    )

    train_processed, metadata = engineer.fit_transform(train_df)
    score_processed, _ = engineer.transform(
        score_df,
        interaction_context=metadata["interaction_context"],
    )

    assert train_processed["城市车型热度"].tolist() == [2, 2, 1]
    assert score_processed["城市车型热度"].tolist() == [2, 0]
    assert metadata["interaction_context"]["city_car_heat"]["上海市|||车型A"] == 2


def test_feature_engineer_keeps_raw_average_duration_and_writes_derived_column():
    df = pd.DataFrame(
        {
            "线索创建时间": ["2026-03-01 10:00:00"],
            "首触时间": ["2026-03-01 10:20:00"],
            "一级渠道名称": ["直营"],
            "二级渠道名称": ["官网"],
            "所在城市": ["上海市"],
            "首触意向车型": ["车型A"],
            "通话次数": [2],
            "通话总时长": [120],
            "平均通话时长": [999.0],
        }
    )

    engineer = FeatureEngineer(
        time_columns=["线索创建时间", "首触时间"],
        numeric_columns=["通话次数", "通话总时长", "平均通话时长"],
    )
    processed, _ = engineer.fit_transform(df)

    assert processed.loc[0, "平均通话时长"] == 999.0
    assert processed.loc[0, "平均通话时长_派生"] == 60.0


def test_feature_engineer_generates_json_availability_from_raw_followup_column():
    df = pd.DataFrame(
        {
            "线索创建时间": ["2026-03-01 10:00:00", "2026-03-01 10:00:00"],
            "首触时间": ["2026-03-01 10:20:00", "2026-03-01 10:20:00"],
            "一级渠道名称": ["直营", "直营"],
            "二级渠道名称": ["官网", "官网"],
            "所在城市": ["上海市", "上海市"],
            "首触意向车型": ["车型A", "车型A"],
            "通话次数": [1, 1],
            "通话总时长": [80, 80],
            "平均通话时长": [80.0, 80.0],
            "非首触跟进记录": ['{"items":[1]}', ""],
        }
    )

    engineer = FeatureEngineer(
        time_columns=["线索创建时间", "首触时间"],
        numeric_columns=["通话次数", "通话总时长", "平均通话时长"],
    )
    processed, _ = engineer.fit_transform(df)

    assert "JSON跟进明细可用" in processed.columns
    assert processed["JSON跟进明细可用"].tolist() == [1, 0]
