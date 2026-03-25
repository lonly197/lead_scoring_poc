from pathlib import Path

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
        "STORE_ID",
        customer_id,
        "13800000000",
        "2026-03-01 00:10:03",
        "渠道1",
        "渠道2",
        "渠道3",
        "渠道4",
        line_type,
        "1",
        "上海市",
        "车型A",
        budget,
        "2026-03-01 00:20:00",
        "2026-03-01 00:30:00",
        "",
        "",
        "",
        "",
        "2026-03-01 00:40:00",
        "2",
        "120",
        "",
        "",
        "{}",
        level,
        "2026-03-01 01:00:00",
        "{}",
        "",
        level,
        "否",
        "否",
        "否",
        "否",
        "否",
        "2026-03-05 10:00:00",
        "D1",
        "2026-03-05",
        "2026-03-06 10:00:00",
        "",
        sop_tag,
        payment_status,
        "",
        history_orders,
        history_arrives,
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
    assert df.iloc[0]["历史订单次数"] == "3"
    assert df.iloc[0]["历史到店次数"] == "4"
    assert "预算区间_备用" not in df.columns
    assert metadata["schema_contract"]["applied_aliases"]["预算区间_备用"] == "预算区间"
