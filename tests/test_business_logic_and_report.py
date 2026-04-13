import json
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.evaluation.business_logic import build_lead_action_record, get_sop_for_label


def _load_generate_business_report():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "generate_business_report.py"
    spec = importlib.util.spec_from_file_location("scripts.generate_business_report", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.generate_business_report"] = module
    spec.loader.exec_module(module)
    return module


generate_business_report = _load_generate_business_report()


def test_build_lead_action_record_returns_sop_and_reasons():
    row = pd.Series(
        {
            "线索唯一ID": "DIS1001",
            "提及试驾": 1,
            "客户是否主动询问金融政策": "是",
            "有效通话": 1,
        }
    )

    record = build_lead_action_record(row, predicted_label="H", probability_map={"H": 0.82, "A": 0.12, "B": 0.06})

    assert record["线索唯一ID"] == "DIS1001"
    assert record["预测HAB"] == "H"
    assert record["建议SOP"] == get_sop_for_label("H")
    assert record["原因1"]


def test_generate_business_report_reads_structured_outputs(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "ohab_model"
    validation_dir = tmp_path / "validation"
    output_path = tmp_path / "reports" / "hab_poc_report.md"
    model_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    (model_dir / "business_dimension_contribution.json").write_text(
        json.dumps({"意图特征": 0.42, "行为特征": 0.31}, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"feature": "提及试驾", "importance": 0.4},
            {"feature": "有效通话", "importance": 0.3},
        ]
    ).to_csv(model_dir / "feature_importance.csv", index=False)
    (validation_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "technical_best_model": "WeightedEnsemble_L2",
                "business_recommended_model": "LightGBM",
                "report_primary_role": "baseline",
                "primary_metrics": {
                    "balanced_accuracy": 0.58,
                    "macro_f1": 0.57,
                },
                "business_kpis": {
                    "h_arrive_lift": 1.8,
                    "h_drive_lift": 2.1,
                    "ha_arrive_capture": 0.82,
                    "ha_drive_capture": 0.79,
                    "b_bucket_share": 0.30,
                    "client_layering_message": "已形成初步分层效果，H/A 边界仍需二期优化",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (validation_dir / "hab_bucket_summary.json").write_text(
        json.dumps(
            [
                {"bucket": "H", "sample_ratio": 0.2, "到店标签_14天_rate": 0.4, "试驾标签_14天_rate": 0.3},
                {"bucket": "A", "sample_ratio": 0.5, "到店标签_14天_rate": 0.2, "试驾标签_14天_rate": 0.1},
                {"bucket": "B", "sample_ratio": 0.3, "到店标签_14天_rate": 0.05, "试驾标签_14天_rate": 0.03},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (validation_dir / "monotonicity_check.json").write_text(
        json.dumps({"message": "H/A/B 分层未形成单调"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (validation_dir / "model_comparison.json").write_text(
        json.dumps(
            [
                {
                    "role": "baseline",
                    "model_name": "LightGBM",
                    "balanced_accuracy": 0.51,
                    "macro_f1": 0.49,
                    "b_recall": 0.08,
                    "h_arrive_lift": 1.8,
                    "h_drive_lift": 2.1,
                    "ha_arrive_capture": 0.82,
                    "ha_drive_capture": 0.79,
                    "b_bucket_share": 0.30,
                },
                {
                    "role": "best",
                    "model_name": "WeightedEnsemble_L2",
                    "balanced_accuracy": 0.58,
                    "macro_f1": 0.57,
                    "b_recall": 0.14,
                    "h_arrive_lift": 1.4,
                    "h_drive_lift": 1.5,
                    "ha_arrive_capture": 0.74,
                    "ha_drive_capture": 0.70,
                    "b_bucket_share": 0.18,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"线索唯一ID": "DIS1", "预测HAB": "H", "建议SOP": "24小时内优先跟进", "原因1": "客户提及试驾"},
        ]
    ).to_csv(validation_dir / "lead_actions.csv", index=False)

    monkeypatch.setattr(
        generate_business_report,
        "parse_args",
        lambda: Namespace(model_dir=str(model_dir), validation_dir=str(validation_dir), output_path=str(output_path)),
    )

    generate_business_report.generate_single_report(model_dir, validation_dir, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "HAB 线索评级 POC 业务报告" in content
    assert "业务推荐模型" in content
    assert "自动寻优候选模型" in content
    assert "H+A 转化覆盖率" in content
    assert "B 桶降频空间" in content
    assert "24小时内优先跟进" in content
    assert "基线模型 vs 最优模型" not in content
    assert "分层未形成单调" not in content
    assert "classification_report" not in content
