from pathlib import Path

import pandas as pd

from src.data.adapter import normalize_schema_contract
from src.data.loader import DataLoader


def _new_format_row(
    lead_id: str,
    customer_id: str,
    *,
    level: str = "H",
    line_type: str = "",
    budget: str = "",
    sop_tag: str = "",
    payment_status: str = "",
    history_orders: str = "0",
    history_arrives: str = "0",
) -> list[str]:
    return [
        lead_id,
        customer_id,
        "STORE_ID",
        "13800000000",
        "2026-03-01 00:10:03",
        "渠道1",
        "渠道2",
        "渠道3",
        "渠道4",
        line_type,
        "男",
        "上海市",
        "车型A",
        budget,
        "2026-03-01 00:20:00",
        "2026-03-01 00:30:00",
        "2",
        "120",
        "60",
        "1",
        "2026-03-01 00:40:00",
        "1",
        "1",
        "1",
        "1",
        "",
        level,
        "2026-03-01 00:50:00",
        "{}",
        "2026-03-01 01:00:00",
        level,
        "否",
        "否",
        "否",
        "否",
        "否",
        "2026-03-05 10:00:00",
        "D1",
        "2026-03-06 10:00:00",
        "2026-03-07 10:00:00",
        "",
        sop_tag,
        payment_status,
        history_orders,
        history_arrives,
        "1",
    ]


def test_auto_adapt_loads_headerless_tsv_and_derives_targets(tmp_path):
    file_path = tmp_path / "202603.tsv"
    rows = [
        "\t".join(_new_format_row("DIS1", "CUST1")),
        "\t".join(_new_format_row("DIS2", "CUST2")),
    ]
    file_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    loader = DataLoader(str(file_path), auto_adapt=True)
    df = loader.load()

    assert loader.get_data_format() == "new"
    assert len(df) == 2
    assert df.iloc[0]["线索唯一ID"] == "DIS1"
    assert "到店标签_14天" in df.columns
    assert "试驾标签_14天" in df.columns
    assert "线索评级_试驾前" in df.columns
    assert "线索创建星期几" in df.columns
    assert "线索创建小时" in df.columns


def test_auto_adapt_normalizes_schema_contract_and_preserves_o_level(tmp_path):
    file_path = tmp_path / "202603.tsv"
    row = _new_format_row(
        "DISO",
        "CUSTO",
        level="O",
        line_type="试驾线索",
        budget="20-25万",
        sop_tag="标准开场",
        payment_status="已支付",
        history_orders="3",
        history_arrives="4",
    )
    file_path.write_text("\t".join(row) + "\n", encoding="utf-8")

    loader = DataLoader(str(file_path), auto_adapt=True)
    df = loader.load()
    metadata = loader.get_adaptation_metadata()

    assert df.iloc[0]["线索评级_试驾前"] == "O"
    assert df.iloc[0]["线索类型"] == "试驾线索"
    assert df.iloc[0]["预算区间"] == "20-25万"
    assert df.iloc[0]["SOP开口标签"] == "标准开场"
    assert df.iloc[0]["意向金支付状态"] == "已支付"
    assert df.iloc[0]["历史订单次数"] == 3
    assert df.iloc[0]["历史到店次数"] == 4
    assert "预算区间_备用" not in df.columns
    assert "SOP开口标签_备用" not in df.columns
    assert "意向金支付状态_备用" not in df.columns
    assert "历史订单次数_备用" not in df.columns
    assert "历史到店次数_备用" not in df.columns
    assert metadata["schema_contract"]["applied_aliases"] == {}


def test_auto_adapt_headerless_tsv_uses_sql_style_canonical_names(tmp_path):
    file_path = tmp_path / "202602~03.tsv"
    row = _new_format_row(
        "DIS_SQL",
        "CUST_SQL",
        line_type="留资线索",
        budget="15-20万",
        sop_tag="邀约SOP",
        payment_status="支付成功",
        history_orders="1",
        history_arrives="2",
    )
    file_path.write_text("\t".join(row) + "\n", encoding="utf-8")

    loader = DataLoader(str(file_path), auto_adapt=True)
    df = loader.load()

    assert df.iloc[0]["客户ID_店端"] == "STORE_ID"
    assert df.iloc[0]["预算区间"] == "15-20万"
    assert pd.isna(df.iloc[0]["首触跟进记录"])
    assert df.iloc[0]["非首触跟进时间"] == "2026-03-01 00:50:00"
    assert df.iloc[0]["线索评级变化时间"] == "2026-03-01 01:00:00"
    assert df.iloc[0]["到店经销商ID"] == "D1"
    assert df.iloc[0]["SOP开口标签"] == "邀约SOP"
    assert df.iloc[0]["意向金支付状态"] == "支付成功"
    assert df.iloc[0]["历史订单次数"] == 1
    assert df.iloc[0]["历史到店次数"] == 2


def test_normalize_schema_contract_handles_sql_export_aliases():
    df = pd.DataFrame(
        {
            "客户ID(店端)": ["STORE_1"],
            "手机号（脱敏）": ["13800000000"],
            "首触意向车型/意向车型": ["铂智3X"],
            "预算区间(购车预算)": ["15-20万"],
            "客户是否主动询问购车权益（优惠）": ["是"],
            "通话时长是否>=45秒": [1],
        }
    )

    normalized_df, metadata = normalize_schema_contract(df)

    assert normalized_df.loc[0, "客户ID_店端"] == "STORE_1"
    assert normalized_df.loc[0, "手机号_脱敏"] == "13800000000"
    assert normalized_df.loc[0, "首触意向车型"] == "铂智3X"
    assert normalized_df.loc[0, "预算区间"] == "15-20万"
    assert normalized_df.loc[0, "客户是否主动询问购车权益"] == "是"
    assert normalized_df.loc[0, "通话时长是否大于等于45秒"] == 1
    assert metadata["applied_aliases"]["客户ID(店端)"] == "客户ID_店端"
