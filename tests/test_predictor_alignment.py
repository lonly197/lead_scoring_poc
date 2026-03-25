import json
import sys
import types

import pandas as pd
import pytest

from src.models.predictor import LeadScoringPredictor


class FakeTabularPredictor:
    last_instance = None
    load_return = None
    feature_importance_return = None

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
        self.last_predict_model = None
        self.last_predict_proba_model = None
        self.last_delete_models = None
        self.fit_train_columns = None
        self.fit_tuning_columns = None
        self.fit_kwargs = None
        FakeTabularPredictor.last_instance = self

    def fit(self, train_data, **kwargs):
        self.fit_train_columns = list(train_data.columns)
        tuning_data = kwargs.get("tuning_data")
        self.fit_tuning_columns = list(tuning_data.columns) if tuning_data is not None else None
        self.fit_kwargs = kwargs
        return self

    def predict(self, data, model=None):
        self.last_predict_columns = list(data.columns)
        self.last_predict_model = model
        return pd.Series([0] * len(data))

    def predict_proba(self, data, model=None):
        self.last_predict_proba_columns = list(data.columns)
        self.last_predict_proba_model = model
        return pd.DataFrame({"neg": [0.4] * len(data), "pos": [0.6] * len(data)})

    def evaluate(self, data, silent=True):
        self.last_evaluate_columns = list(data.columns)
        return {"accuracy": 1.0}

    def feature_importance(self, data):
        self.last_feature_importance_columns = list(data.columns)
        if FakeTabularPredictor.feature_importance_return is not None:
            return FakeTabularPredictor.feature_importance_return
        feature_columns = [col for col in data.columns if col != self.label]
        return pd.Series([1.0] * len(feature_columns), index=feature_columns)

    def model_names(self):
        return ["FakeModel", "BaselineModel"]

    def delete_models(self, models_to_keep, dry_run=False):
        self.last_delete_models = {"models_to_keep": models_to_keep, "dry_run": dry_run}

    def save_space(self):
        return None

    @classmethod
    def load(cls, path, require_version_match=False):
        return cls.load_return


def install_fake_autogluon(monkeypatch):
    FakeTabularPredictor.feature_importance_return = None
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
        num_bag_folds=5,
        presets="medium_quality",
        time_limit=1,
    )

    fake_predictor = FakeTabularPredictor.last_instance
    assert predictor._feature_columns == ["feat_a", "feat_b"]
    assert fake_predictor.fit_train_columns == ["feat_a", "feat_b", "label"]
    assert fake_predictor.fit_tuning_columns == ["feat_a", "feat_b", "label"]
    assert fake_predictor.fit_kwargs["use_bag_holdout"] is True


def test_train_does_not_inject_use_bag_holdout_without_bagging(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    train_data = pd.DataFrame({"feat_a": [1, 2], "label": [0, 1]})
    tuning_data = pd.DataFrame({"feat_a": [3, 4], "label": [1, 0]})

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor.train(
        train_data=train_data,
        tuning_data=tuning_data,
        presets="medium_quality",
        time_limit=1,
        num_bag_folds=None,
    )

    fake_predictor = FakeTabularPredictor.last_instance
    assert "use_bag_holdout" not in fake_predictor.fit_kwargs


def test_train_rejects_invalid_bagging_with_tuning_configuration(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    train_data = pd.DataFrame({"feat_a": [1, 2], "label": [0, 1]})
    tuning_data = pd.DataFrame({"feat_a": [3, 4], "label": [1, 0]})

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))

    with pytest.raises(
        ValueError,
        match="use_bag_holdout=True",
    ):
        predictor.train(
            train_data=train_data,
            tuning_data=tuning_data,
            presets="medium_quality",
            time_limit=1,
            num_bag_folds=5,
            use_bag_holdout=False,
        )


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


def test_predict_proba_passes_explicit_model_name(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor._predictor = FakeTabularPredictor(label="label", path=str(tmp_path))
    predictor._feature_columns = ["feat_a"]

    predictor.predict_proba(pd.DataFrame({"feat_a": [1]}), model="BaselineModel")

    assert predictor._predictor.last_predict_proba_model == "BaselineModel"


def test_cleanup_can_keep_named_models(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor._predictor = FakeTabularPredictor(label="label", path=str(tmp_path))

    predictor.cleanup(keep_best_only=False, keep_model_names=["FakeModel", "BaselineModel"])

    assert predictor._predictor.last_delete_models["models_to_keep"] == ["FakeModel", "BaselineModel"]


def test_get_feature_importance_normalizes_series(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)
    FakeTabularPredictor.feature_importance_return = pd.Series(
        [0.2, 0.9],
        index=["feat_a", "feat_b"],
    )

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor._predictor = FakeTabularPredictor(label="label", path=str(tmp_path))
    predictor._feature_columns = ["feat_a", "feat_b"]

    importance_df = predictor.get_feature_importance(
        pd.DataFrame({"feat_a": [1], "feat_b": [2], "label": [0]})
    )

    assert list(importance_df.columns) == ["feature", "importance"]
    assert importance_df.to_dict(orient="records") == [
        {"feature": "feat_b", "importance": 0.9},
        {"feature": "feat_a", "importance": 0.2},
    ]


def test_get_feature_importance_normalizes_dataframe(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)
    FakeTabularPredictor.feature_importance_return = pd.DataFrame(
        {
            "importance": [0.2, 0.9],
            "stddev": [0.01, 0.03],
        },
        index=pd.Index(["feat_a", "feat_b"], name="feature"),
    )

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor._predictor = FakeTabularPredictor(label="label", path=str(tmp_path))
    predictor._feature_columns = ["feat_a", "feat_b"]

    importance_df = predictor.get_feature_importance(
        pd.DataFrame({"feat_a": [1], "feat_b": [2], "label": [0]})
    )

    assert list(importance_df.columns) == ["feature", "importance", "stddev"]
    assert importance_df.to_dict(orient="records") == [
        {"feature": "feat_b", "importance": 0.9, "stddev": 0.03},
        {"feature": "feat_a", "importance": 0.2, "stddev": 0.01},
    ]


def test_train_passes_memory_and_ensemble_controls_with_autogluon_shape(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    train_data = pd.DataFrame({"feat_a": [1, 2], "label": [0, 1]})

    predictor = LeadScoringPredictor(
        label="label",
        output_path=str(tmp_path),
        memory_limit_gb=12,
        fit_strategy="sequential",
        excluded_model_types=["RF", "XT"],
        num_folds_parallel=1,
        max_memory_usage_ratio=0.9,
    )
    predictor.train(
        train_data=train_data,
        presets="good_quality",
        time_limit=1,
        num_bag_folds=3,
    )

    fake_predictor = FakeTabularPredictor.last_instance
    assert fake_predictor.fit_kwargs["memory_limit"] == 12
    assert fake_predictor.fit_kwargs["fit_strategy"] == "sequential"
    assert fake_predictor.fit_kwargs["excluded_model_types"] == ["RF", "XT"]
    assert fake_predictor.fit_kwargs["ag_args_ensemble"] == {"num_folds_parallel": 1}
    assert fake_predictor.fit_kwargs["ag_args_fit"] == {"max_memory_usage_ratio": 0.9}
