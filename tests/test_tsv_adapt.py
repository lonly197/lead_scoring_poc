from pathlib import Path

from src.data.loader import DataLoader


def _new_format_row(lead_id: str, customer_id: str) -> list[str]:
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
        "",
        "1",
        "上海市",
        "车型A",
        "",
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
        "H",
        "2026-03-01 01:00:00",
        "{}",
        "",
        "H",
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
        "",
        "",
        "",
        "0",
        "0",
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
