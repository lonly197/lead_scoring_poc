import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_train_ohab_script(monkeypatch, tmp_path: Path):
    base_output_dir = tmp_path / "outputs"
    model_root = base_output_dir / "models"
    topk_root = base_output_dir / "topk_lists"

    config_module = types.ModuleType("config.config")
    config_module.config = types.SimpleNamespace(
        feature=types.SimpleNamespace(
            time_columns=["线索创建时间"],
            numeric_features=[],
        ),
        output=types.SimpleNamespace(
            models_dir=model_root,
            topk_dir=topk_root,
        ),
    )
    config_module.get_excluded_columns = lambda target: []

    sample_df = pd.DataFrame(
        {
            "线索唯一ID": [1, 2, 3, 4, 5, 6],
            "线索评级_试驾前": ["H", "A", "B", "H", "A", "B"],
            "线索创建时间": pd.date_range("2026-03-01", periods=6, freq="D"),
            "到店标签_14天": [1, 0, 0, 1, 0, 0],
            "试驾标签_14天": [0, 0, 1, 0, 1, 0],
        }
    )

    loader_module = types.ModuleType("src.data.loader")

    class DummyDataLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return sample_df.copy()

        def get_adaptation_metadata(self):
            return {"schema_contract": {}}

    class DummyFeatureEngineer:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, df):
            return df.copy(), {"interaction_context": {}}

        def transform(self, df, interaction_context=None):
            return df.copy(), {}

    def split_data_oot_three_way(df, target_label, time_column, train_end, valid_end):
        return df.iloc[:2].copy(), df.iloc[2:4].copy(), df.iloc[4:6].copy()

    def smart_split_data(df, target_label):
        return (
            df.iloc[:2].copy(),
            df.iloc[2:4].copy(),
            df.iloc[4:6].copy(),
            {"mode": "oot", "train_end": "2026-03-15", "valid_end": "2026-03-20"},
        )

    loader_module.DataLoader = DummyDataLoader
    loader_module.FeatureEngineer = DummyFeatureEngineer
    loader_module.smart_split_data = smart_split_data
    loader_module.split_data_oot_three_way = split_data_oot_three_way

    label_policy_module = types.ModuleType("src.data.label_policy")
    label_policy_module.apply_ohab_label_policy = lambda df, target, policy: df.copy()
    label_policy_module.filter_to_effective_ohab_labels = lambda df, target, policy: df.copy()
    label_policy_module.build_ohab_label_policy = lambda *args, **kwargs: {
        "target_label": "线索评级_试驾前",
        "label_mode": "hab",
        "mapping": {},
    }

    ohab_metrics_module = types.ModuleType("src.evaluation.ohab_metrics")
    ohab_metrics_module.apply_hab_decision_policy = lambda y_proba, policy: y_proba.idxmax(axis=1)
    ohab_metrics_module.classification_report_dict = lambda y_true, y_pred: {
        "macro avg": {"f1-score": 0.5},
        "B": {"recall": 0.25},
    }
    ohab_metrics_module.compute_hab_bucket_summary = lambda *args, **kwargs: pd.DataFrame(
        [
            {"bucket": "H", "到店标签_14天_rate": 0.10, "试驾标签_14天_rate": 0.05},
            {"bucket": "A", "到店标签_14天_rate": 0.08, "试驾标签_14天_rate": 0.03},
            {"bucket": "B", "到店标签_14天_rate": 0.04, "试驾标签_14天_rate": 0.01},
        ]
    )
    ohab_metrics_module.compute_class_ranking_report = lambda *args, **kwargs: {"top_10": {}}
    ohab_metrics_module.compute_threshold_report = lambda *args, **kwargs: pd.DataFrame(
        [{"threshold": 0.5, "recall": 0.25}]
    )
    ohab_metrics_module.confusion_matrix_frame = lambda *args, **kwargs: pd.DataFrame(
        {"预测H": [1], "预测A": [0]}
    )
    ohab_metrics_module.check_hab_monotonicity = lambda *args, **kwargs: {
        "passed": True,
        "message": "ok",
    }
    ohab_metrics_module.optimize_hab_decision_policy = lambda *args, **kwargs: {
        "strategy": "argmax"
    }

    business_logic_module = types.ModuleType("src.evaluation.business_logic")
    business_logic_module.calculate_dimension_contribution = lambda feature_importance: {
        "响应管理": 1.0
    }
    business_logic_module.BUSINESS_DIMENSION_MAP = {}

    class DummyPredictor:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.saved_extra_metadata = None
            self.cleanup_kwargs = None
            DummyPredictor.last_instance = self

        def train(self, **kwargs):
            self.train_kwargs = kwargs

        def get_model_info(self):
            return {"best_model": "WeightedEnsemble_L3_FULL"}

        def get_leaderboard(self, test_data=None, silent=True):
            return pd.DataFrame(
                {
                    "model": [
                        "LightGBMXT_BAG_L2_FULL",
                        "WeightedEnsemble_L3_FULL",
                    ],
                    "score_test": [0.8, 0.9],
                }
            )

        def evaluate(self, test_df):
            return {
                "log_loss": -0.75,
                "accuracy": 0.66,
                "balanced_accuracy": 0.37,
                "mcc": 0.36,
                "macro_f1": 0.37,
                "weighted_f1": 0.65,
            }

        def predict_proba(self, test_df, model=None):
            rows = len(test_df)
            return pd.DataFrame(
                {
                    "H": [0.7] * rows,
                    "A": [0.2] * rows,
                    "B": [0.1] * rows,
                    "O": [0.0] * rows,
                }
            )

        def predict(self, test_df, model=None):
            return pd.Series(["H"] * len(test_df))

        def get_feature_importance(self, test_df):
            raise ValueError("boom")

        def save(self, extra_metadata=None):
            self.saved_extra_metadata = extra_metadata

        def cleanup(self, keep_best_only=True, keep_model_names=None):
            self.cleanup_kwargs = {
                "keep_best_only": keep_best_only,
                "keep_model_names": keep_model_names,
            }

    predictor_module = types.ModuleType("src.models.predictor")
    predictor_module.LeadScoringPredictor = DummyPredictor

    runtime_module = types.ModuleType("src.training.ohab_runtime")
    runtime_module.resolve_training_config = lambda args: {
        "training_profile": "server_16g_compare",
        "preset": "good_quality",
        "time_limit": 60,
        "num_bag_folds": 3,
        "label_mode": "hab",
        "enable_model_comparison": True,
        "baseline_family": "gbm",
        "memory_limit_gb": 8.5,
        "fit_strategy": "sequential",
        "excluded_model_types": ["RF", "XT"],
        "num_folds_parallel": 1,
        "max_memory_ratio": 0.7,
        "detected_resources": {"cpu_count": 8, "available_memory_gb": 10.0},
        "resource_tuning": {
            "memory_limit_source": "auto",
            "num_folds_parallel_source": "auto",
            "derived_memory_limit_gb": 8.5,
            "derived_num_folds_parallel": 1,
        },
    }

    visualization_module = types.ModuleType("src.utils.visualization")
    visualization_module.plot_feature_importance = lambda *args, **kwargs: None
    visualization_module.plot_dimension_contribution = lambda *args, **kwargs: None

    helpers_module = types.ModuleType("src.utils.helpers")
    helpers_module.check_disk_space = lambda *args, **kwargs: {"free_gb": 20.0}
    helpers_module.complete_process_if_running = lambda *args, **kwargs: None
    helpers_module.format_training_duration = lambda seconds: f"{int(seconds)}秒"
    helpers_module.get_local_now = lambda: datetime(2026, 3, 25, 18, 8, 0, tzinfo=timezone.utc)
    helpers_module.get_preset_disk_requirement = lambda preset: 2.0
    helpers_module.get_timestamp = lambda: "20260325_180800"
    helpers_module.save_process_info = lambda *args, **kwargs: None
    helpers_module.setup_logging = lambda *args, **kwargs: None
    helpers_module.update_process_status = lambda *args, **kwargs: None

    topk_module = types.ModuleType("scripts.generate_topk")
    topk_module.generate_topk_from_predictor = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "config.config", config_module)
    monkeypatch.setitem(sys.modules, "src.data.loader", loader_module)
    monkeypatch.setitem(sys.modules, "src.data.label_policy", label_policy_module)
    monkeypatch.setitem(sys.modules, "src.evaluation.ohab_metrics", ohab_metrics_module)
    monkeypatch.setitem(sys.modules, "src.evaluation.business_logic", business_logic_module)
    monkeypatch.setitem(sys.modules, "src.models.predictor", predictor_module)
    monkeypatch.setitem(sys.modules, "src.training.ohab_runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "src.utils.visualization", visualization_module)
    monkeypatch.setitem(sys.modules, "src.utils.helpers", helpers_module)
    monkeypatch.setitem(sys.modules, "scripts.generate_topk", topk_module)

    module_name = "scripts.train_ohab"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_ohab.py"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, DummyPredictor


def test_train_ohab_persists_core_artifacts_when_feature_importance_fails(monkeypatch, tmp_path):
    train_ohab, dummy_predictor_class = load_train_ohab_script(monkeypatch, tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ohab.py",
            "--data-path",
            "./data/202602~03.tsv",
            "--output-dir",
            str(tmp_path / "outputs" / "models"),
            "--train-end",
            "2026-03-15",
            "--valid-end",
            "2026-03-20",
        ],
    )

    train_ohab.main()

    model_dir = tmp_path / "outputs" / "models" / "ohab_model"
    feature_metadata = json.loads((model_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    evaluation_summary = json.loads((model_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    comparison_config = json.loads((model_dir / "model_comparison_config.json").read_text(encoding="utf-8"))

    assert (model_dir / "predictions_test.csv").exists()
    assert feature_metadata["artifact_status"]["training_complete"] is True
    assert "feature_importance" in feature_metadata["artifact_status"]["supplemental_failures"]
    assert "feature_importance" in evaluation_summary["supplemental_failures"]
    assert comparison_config["models"]["LightGBMXT_BAG_L2_FULL"]["role"] == "baseline"
    assert comparison_config["models"]["WeightedEnsemble_L3_FULL"]["role"] == "best"

    dummy_predictor = dummy_predictor_class.last_instance
    assert dummy_predictor.cleanup_kwargs["keep_model_names"] == [
        "LightGBMXT_BAG_L2_FULL",
        "WeightedEnsemble_L3_FULL",
    ]
