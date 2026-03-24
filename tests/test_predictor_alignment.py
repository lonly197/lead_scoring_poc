import json
import sys
import types

import pandas as pd
import pytest

from src.models.predictor import LeadScoringPredictor


class FakeTabularPredictor:
    last_instance = None
    load_return = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.label = kwargs.get("label", "label")
        self.eval_metric = kwargs.get("eval_metric", "roc_auc")
        self.problem_type = kwargs.get("problem_type", "binary")
        self.model_best = "FakeModel"
        self.last_predict_columns = None
        self.last_predict_proba_columns = None
        self.last_evaluate_columns = None
        self.last_feature_importance_columns = None
        self.fit_train_columns = None
        self.fit_tuning_columns = None
        FakeTabularPredictor.last_instance = self

    def fit(self, train_data, **kwargs):
        self.fit_train_columns = list(train_data.columns)
        tuning_data = kwargs.get("tuning_data")
        self.fit_tuning_columns = list(tuning_data.columns) if tuning_data is not None else None
        return self

    def predict(self, data):
        self.last_predict_columns = list(data.columns)
        return pd.Series([0] * len(data))

    def predict_proba(self, data):
        self.last_predict_proba_columns = list(data.columns)
        return pd.DataFrame({"neg": [0.4] * len(data), "pos": [0.6] * len(data)})

    def evaluate(self, data, silent=True):
        self.last_evaluate_columns = list(data.columns)
        return {"accuracy": 1.0}

    def feature_importance(self, data):
        self.last_feature_importance_columns = list(data.columns)
        feature_columns = [col for col in data.columns if col != self.label]
        return pd.Series([1.0] * len(feature_columns), index=feature_columns)

    def model_names(self):
        return ["FakeModel"]

    @classmethod
    def load(cls, path, require_version_match=False):
        return cls.load_return


def install_fake_autogluon(monkeypatch):
    autogluon_module = types.ModuleType("autogluon")
    tabular_module = types.ModuleType("autogluon.tabular")
    tabular_module.TabularPredictor = FakeTabularPredictor
    autogluon_module.tabular = tabular_module

    monkeypatch.setitem(sys.modules, "autogluon", autogluon_module)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_module)


def test_train_aligns_tuning_data_to_training_feature_set(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    train_data = pd.DataFrame(
        {
            "feat_a": [1, 2],
            "feat_b": [3, 4],
            "leak_col": [10, 20],
            "label": [0, 1],
        }
    )
    tuning_data = pd.DataFrame(
        {
            "feat_a": [5, 6],
            "feat_b": [7, 8],
            "leak_col": [30, 40],
            "extra_col": [50, 60],
            "label": [1, 0],
        }
    )

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor.train(
        train_data=train_data,
        tuning_data=tuning_data,
        excluded_columns=["leak_col"],
        presets="medium_quality",
        time_limit=1,
    )

    fake_predictor = FakeTabularPredictor.last_instance
    assert predictor._feature_columns == ["feat_a", "feat_b"]
    assert fake_predictor.fit_train_columns == ["feat_a", "feat_b", "label"]
    assert fake_predictor.fit_tuning_columns == ["feat_a", "feat_b", "label"]


def test_predict_proba_raises_clear_error_when_feature_missing(tmp_path):
    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor._predictor = object()
    predictor._feature_columns = ["feat_a", "feat_b"]

    with pytest.raises(ValueError, match="predict_data 缺少 1 个训练特征列: feat_b"):
        predictor.predict_proba(pd.DataFrame({"feat_a": [1]}))


def test_load_restores_feature_columns_and_aligns_predict_proba(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    predictor = LeadScoringPredictor(label="label", output_path=str(model_dir))
    predictor._feature_columns = ["feat_a", "feat_b"]
    predictor.save(str(model_dir))

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == ["feat_a", "feat_b"]

    FakeTabularPredictor.load_return = FakeTabularPredictor(
        label="label",
        eval_metric="roc_auc",
        problem_type="binary",
        path=str(model_dir),
    )

    loaded_predictor = LeadScoringPredictor.load(str(model_dir))
    loaded_predictor.predict_proba(
        pd.DataFrame(
            {
                "extra_col": [9],
                "feat_b": [2],
                "feat_a": [1],
            }
        )
    )

    assert loaded_predictor._feature_columns == ["feat_a", "feat_b"]
    assert FakeTabularPredictor.load_return.last_predict_proba_columns == ["feat_a", "feat_b"]
