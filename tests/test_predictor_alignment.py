import json
import sys
import types

import pandas as pd
import pytest

import src.models.predictor as predictor_module
from src.models.predictor import LeadScoringPredictor


class FakeTrainer:
    def __init__(self):
        self._callback_early_stop = False
        self.hyperparameters = {"GBM": [{}, {}], "CAT": {}}
        self.num_stack_levels = 1
        self.model_attributes = {
            "FakeModel": {
                "val_score": 0.91,
                "fit_time": 1.5,
                "predict_time": 0.1,
            }
        }

    def get_model_attribute(self, model_name, attribute, default=None):
        return self.model_attributes.get(model_name, {}).get(attribute, default)

    def load_model(self, model_name):
        return types.SimpleNamespace(**self.model_attributes.get(model_name, {}))


class FakeTabularPredictor:
    last_instance = None
    load_return = None
    feature_importance_return = None
    predict_proba_return = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.label = kwargs.get("label", "label")
        self.eval_metric = kwargs.get("eval_metric", "roc_auc")
        self.problem_type = kwargs.get("problem_type", "binary")
        self.positive_class = "pos"
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
        self.features_return = ["feat_a", "feat_b"]
        FakeTabularPredictor.last_instance = self

    def fit(self, train_data, **kwargs):
        self.fit_train_columns = list(train_data.columns)
        tuning_data = kwargs.get("tuning_data")
        self.fit_tuning_columns = list(tuning_data.columns) if tuning_data is not None else None
        self.fit_kwargs = kwargs
        callbacks = kwargs.get("callbacks") or []
        if callbacks:
            trainer = FakeTrainer()
            model = types.SimpleNamespace(name="FakeModel")
            for callback in callbacks:
                if hasattr(callback, "before_trainer_fit"):
                    callback.before_trainer_fit(
                        trainer,
                        hyperparameters=trainer.hyperparameters,
                        level_start=1,
                        level_end=2,
                    )
                if hasattr(callback, "before_model_fit"):
                    callback.before_model_fit(
                        trainer,
                        model,
                        time_limit=kwargs.get("time_limit"),
                        stack_name="core",
                        level=1,
                    )
                elif hasattr(callback, "_before_model_fit"):
                    callback._before_model_fit(
                        trainer=trainer,
                        model=model,
                        time_limit=kwargs.get("time_limit"),
                        stack_name="core",
                        level=1,
                    )
                if hasattr(callback, "after_model_fit"):
                    callback.after_model_fit(
                        trainer,
                        ["FakeModel"],
                        stack_name="core",
                        level=1,
                    )
                elif hasattr(callback, "_after_model_fit"):
                    callback._after_model_fit(
                        trainer=trainer,
                        model_names=["FakeModel"],
                        stack_name="core",
                        level=1,
                    )
                if hasattr(callback, "after_trainer_fit"):
                    callback.after_trainer_fit(trainer)
        return self

    def predict(self, data, model=None):
        self.last_predict_columns = list(data.columns)
        self.last_predict_model = model
        return pd.Series([0] * len(data))

    def predict_proba(self, data, model=None):
        self.last_predict_proba_columns = list(data.columns)
        self.last_predict_proba_model = model
        if FakeTabularPredictor.predict_proba_return is not None:
            return FakeTabularPredictor.predict_proba_return.copy()
        return pd.DataFrame({"neg": [0.4] * len(data), "pos": [0.6] * len(data)})

    def evaluate(self, data, silent=True):
        self.last_evaluate_columns = list(data.columns)
        return {"accuracy": 1.0}

    def feature_importance(self, data, **kwargs):
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

    def features(self, feature_stage="original"):
        return list(self.features_return)

    @classmethod
    def load(
        cls,
        path,
        require_version_match=False,
        require_py_version_match=False,
    ):
        return cls.load_return


def install_fake_autogluon(monkeypatch):
    FakeTabularPredictor.feature_importance_return = None
    FakeTabularPredictor.predict_proba_return = None
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
    assert fake_predictor.fit_train_columns == ["feat_a", "feat_b", "leak_col", "label"]
    assert fake_predictor.fit_tuning_columns == ["feat_a", "feat_b", "leak_col", "label"]
    assert fake_predictor.init_kwargs["learner_kwargs"] == {"ignored_columns": ["leak_col"]}
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

    metadata = json.loads((model_dir / "predictor_metadata.json").read_text(encoding="utf-8"))
    assert "feature_columns" not in metadata
    assert not (model_dir / "metadata.json").exists()

    FakeTabularPredictor.load_return = FakeTabularPredictor(
        label="label",
        eval_metric="roc_auc",
        problem_type="binary",
        path=str(model_dir),
    )
    FakeTabularPredictor.load_return.features_return = ["feat_a", "feat_b"]

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


def test_load_uses_legacy_metadata_json_without_breaking_autogluon(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    model_dir = tmp_path / "legacy_model"
    model_dir.mkdir()
    legacy_metadata = {
        "label": "label",
        "eval_metric": "balanced_accuracy",
        "problem_type": "multiclass",
        "feature_columns": ["feat_a", "feat_b"],
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(legacy_metadata, ensure_ascii=False),
        encoding="utf-8",
    )

    FakeTabularPredictor.load_return = FakeTabularPredictor(
        label="label",
        eval_metric="roc_auc",
        problem_type="binary",
        path=str(model_dir),
    )

    loaded_predictor = LeadScoringPredictor.load(str(model_dir))

    assert loaded_predictor.eval_metric == "balanced_accuracy"
    assert loaded_predictor.problem_type == "multiclass"
    assert loaded_predictor._feature_columns == ["feat_a", "feat_b"]


def test_predict_proba_passes_explicit_model_name(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor._predictor = FakeTabularPredictor(label="label", path=str(tmp_path))
    predictor._feature_columns = ["feat_a"]

    predictor.predict_proba(pd.DataFrame({"feat_a": [1]}), model="BaselineModel")

    assert predictor._predictor.last_predict_proba_model == "BaselineModel"


def test_get_class_proba_returns_requested_multiclass_column(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)
    FakeTabularPredictor.predict_proba_return = pd.DataFrame(
        {
            "H": [0.7, 0.2],
            "A": [0.2, 0.5],
            "B": [0.1, 0.3],
        }
    )

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path), problem_type="multiclass")
    predictor._predictor = FakeTabularPredictor(label="label", path=str(tmp_path), problem_type="multiclass")
    predictor._feature_columns = ["feat_a"]

    proba = predictor.get_class_proba(pd.DataFrame({"feat_a": [1, 2]}), target_class="H")

    assert list(proba) == [0.7, 0.2]


def test_get_class_proba_rejects_unknown_class(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)
    FakeTabularPredictor.predict_proba_return = pd.DataFrame(
        {
            "H": [0.7],
            "A": [0.2],
            "B": [0.1],
        }
    )

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path), problem_type="multiclass")
    predictor._predictor = FakeTabularPredictor(label="label", path=str(tmp_path), problem_type="multiclass")
    predictor._feature_columns = ["feat_a"]

    with pytest.raises(ValueError, match="目标类别 O 不存在于预测概率输出中"):
        predictor.get_class_proba(pd.DataFrame({"feat_a": [1]}), target_class="O")


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
    assert fake_predictor.fit_kwargs["ag_args_fit"] == {"ag.max_memory_usage_ratio": 0.9}


def test_train_injects_compatible_progress_callback(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    events = []

    class FakeProgressCallback:
        def __init__(self, time_limit=None):
            self.time_limit = time_limit
            events.append(("init", time_limit))

        def before_trainer_fit(self, trainer, **kwargs):
            events.append(("before_trainer_fit", sorted(kwargs)))

        def _before_model_fit(self, trainer, model, time_limit=None, stack_name="core", level=1):
            events.append(("before_model_fit", model.name, time_limit, stack_name, level))
            return False, False

        def _after_model_fit(self, trainer, model_names, stack_name="core", level=1):
            events.append(("after_model_fit", tuple(model_names), stack_name, level))
            return False

        def after_trainer_fit(self, trainer):
            events.append(("after_trainer_fit", True))

        def get_summary(self):
            return {
                "total_models_trained": 1,
                "best_model": "FakeModel",
                "best_score": 0.91,
                "total_time_formatted": "1秒",
            }

    progress_module = types.ModuleType("src.training.progress_callback")
    progress_module.TrainingProgressCallback = FakeProgressCallback
    monkeypatch.setitem(sys.modules, "src.training.progress_callback", progress_module)
    monkeypatch.setattr(predictor_module, "_is_progress_callback_compatible", lambda cls: True)

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor.train(
        train_data=pd.DataFrame({"feat_a": [1, 2], "label": [0, 1]}),
        presets="good_quality",
        time_limit=7,
    )

    fake_predictor = FakeTabularPredictor.last_instance
    assert len(fake_predictor.fit_kwargs["callbacks"]) == 1
    assert events == [
        ("init", 7),
        ("before_trainer_fit", ["hyperparameters", "level_end", "level_start"]),
        ("before_model_fit", "FakeModel", 7, "core", 1),
        ("after_model_fit", ("FakeModel",), "core", 1),
        ("after_trainer_fit", True),
    ]


def test_train_skips_incompatible_progress_callback(monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    class ExplodingProgressCallback:
        def __init__(self, time_limit=None):
            raise AssertionError("should not be initialized")

    progress_module = types.ModuleType("src.training.progress_callback")
    progress_module.TrainingProgressCallback = ExplodingProgressCallback
    monkeypatch.setitem(sys.modules, "src.training.progress_callback", progress_module)
    monkeypatch.setattr(predictor_module, "_is_progress_callback_compatible", lambda cls: False)

    predictor = LeadScoringPredictor(label="label", output_path=str(tmp_path))
    predictor.train(
        train_data=pd.DataFrame({"feat_a": [1, 2], "label": [0, 1]}),
        presets="good_quality",
        time_limit=5,
    )

    fake_predictor = FakeTabularPredictor.last_instance
    assert fake_predictor.fit_kwargs["callbacks"] == []
