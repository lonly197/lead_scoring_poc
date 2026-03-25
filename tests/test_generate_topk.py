import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import generate_topk


class MulticlassPredictor:
    label = "线索评级_试驾前"

    def get_model_info(self):
        return {"best_model": "WeightedEnsemble_L3_FULL"}

    def get_class_proba(self, data, target_class, model=None):
        assert target_class == "H"
        return np.array([0.2, 0.9, 0.6])

    def predict_proba(self, data, model=None):
        return pd.DataFrame(
            {
                "H": [0.2, 0.9, 0.6],
                "A": [0.6, 0.05, 0.3],
                "B": [0.2, 0.05, 0.1],
            }
        )


class BinaryPredictor:
    label = "到店标签_14天"

    def get_model_info(self):
        return {"best_model": "LightGBM_BAG_L1"}

    def predict_proba(self, data, model=None):
        return pd.DataFrame({"neg": [0.8, 0.2], "pos": [0.2, 0.8]})

    def get_positive_proba(self, data, model=None):
        return np.array([0.2, 0.8])


def test_generate_topk_from_predictor_uses_requested_h_class(tmp_path):
    df = pd.DataFrame(
        {
            "线索唯一ID": [101, 102, 103],
            "线索评级_试驾前": ["A", "H", "B"],
            "feat_a": [1, 2, 3],
        }
    )

    output_prefix = tmp_path / "topk_ohab"
    generated_files = generate_topk.generate_topk_from_predictor(
        predictor=MulticlassPredictor(),
        data=df,
        output_path=str(output_prefix),
        target_class="H",
        k_values=(2,),
    )

    assert generated_files == [tmp_path / "topk_ohab_top2.csv"]
    topk_df = pd.read_csv(generated_files[0])
    assert topk_df["线索唯一ID"].tolist() == [102, 103]
    assert topk_df["actual_label"].tolist() == ["H", "B"]


def test_generate_topk_from_predictor_keeps_binary_default_behavior(tmp_path):
    df = pd.DataFrame(
        {
            "线索唯一ID": [201, 202],
            "到店标签_14天": [0, 1],
            "feat_a": [1, 2],
        }
    )

    generated_files = generate_topk.generate_topk_from_predictor(
        predictor=BinaryPredictor(),
        data=df,
        output_path=str(tmp_path / "topk_arrive"),
        k_values=(2,),
    )

    topk_df = pd.read_csv(generated_files[0])
    assert topk_df["线索唯一ID"].tolist() == [202, 201]
    assert topk_df["actual_label"].tolist() == [1, 0]


def test_generate_topk_cli_requires_target_class_for_multiclass(monkeypatch, tmp_path):
    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return pd.DataFrame(
                {
                    "线索唯一ID": [1, 2],
                    "线索评级_试驾前": ["H", "A"],
                    "feat_a": [1, 2],
                }
            )

    class DummyFeatureEngineer:
        def __init__(self, *args, **kwargs):
            pass

        def transform(self, df, interaction_context=None):
            return df.copy(), {}

    monkeypatch.setattr(generate_topk, "DataLoader", DummyLoader)
    monkeypatch.setattr(generate_topk, "FeatureEngineer", DummyFeatureEngineer)
    monkeypatch.setattr(generate_topk, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(generate_topk, "print_separator", lambda *args, **kwargs: None)
    monkeypatch.setattr(generate_topk.LeadScoringPredictor, "load", classmethod(lambda cls, path: MulticlassPredictor()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_topk.py",
            "--model-path",
            str(tmp_path / "model"),
            "--data-path",
            str(tmp_path / "data.tsv"),
        ],
    )

    with pytest.raises(ValueError, match="多分类 Top-K 必须显式指定 --target-class"):
        generate_topk.main()
