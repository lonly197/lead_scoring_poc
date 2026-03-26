import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_test_drive_validate_script(monkeypatch):
    pandas_module = types.ModuleType("pandas")

    class DummyDataFrame:
        pass

    class DummySeries:
        pass

    pandas_module.DataFrame = DummyDataFrame
    pandas_module.Series = DummySeries

    config_module = types.ModuleType("config.config")
    config_module.config = types.SimpleNamespace(
        data=types.SimpleNamespace(data_path="./data/demo.tsv"),
        feature=types.SimpleNamespace(time_columns=[], numeric_features=[]),
    )
    config_module.get_excluded_columns = lambda target: []

    loader_module = types.ModuleType("src.data.loader")
    loader_module.DataLoader = object
    loader_module.FeatureEngineer = object

    metrics_module = types.ModuleType("src.evaluation.metrics")
    metrics_module.TopKEvaluator = object
    metrics_module.plot_feature_importance = lambda *args, **kwargs: None
    metrics_module.plot_lift_chart = lambda *args, **kwargs: None

    predictor_module = types.ModuleType("src.models.predictor")
    predictor_module.LeadScoringPredictor = object

    helpers_module = types.ModuleType("src.utils.helpers")
    helpers_module.complete_process_if_running = lambda *args, **kwargs: None
    helpers_module.format_training_duration = lambda seconds: f"{seconds}s"
    helpers_module.get_timestamp = lambda: "20260325_120000"
    helpers_module.get_local_now = lambda: None
    helpers_module.format_timestamp = lambda dt: "2026-03-25 12:00:00+0800"
    helpers_module.save_process_info = lambda *args, **kwargs: None
    helpers_module.setup_logging = lambda *args, **kwargs: None
    helpers_module.update_process_status = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "pandas", pandas_module)
    monkeypatch.setitem(sys.modules, "config.config", config_module)
    monkeypatch.setitem(sys.modules, "src.data.loader", loader_module)
    monkeypatch.setitem(sys.modules, "src.evaluation.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "src.models.predictor", predictor_module)
    monkeypatch.setitem(sys.modules, "src.utils.helpers", helpers_module)

    module_name = "scripts.validate_test_drive_model"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_test_drive_model.py"

    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_test_drive_artifacts_accept_minimal_feature_metadata(monkeypatch):
    validate_script = load_test_drive_validate_script(monkeypatch)

    validate_script.validate_test_drive_model_artifacts(
        {
            "split_info": {"mode": "random"},
        },
        Path("outputs/models/test_drive_model"),
    )


def test_test_drive_artifacts_reject_hab_pipeline_metadata(monkeypatch):
    validate_script = load_test_drive_validate_script(monkeypatch)

    with pytest.raises(RuntimeError, match="OHAB"):
        validate_script.validate_test_drive_model_artifacts(
            {
                "pipeline_metadata": {"pipeline_mode": "two_stage"},
            },
            Path("outputs/models/test_drive_model"),
        )


def test_infer_test_drive_model_type_from_path(monkeypatch):
    validate_script = load_test_drive_validate_script(monkeypatch)

    assert validate_script.infer_model_type(Path("outputs/models/test_drive_model")) == "test_drive"


def test_parse_args_uses_test_drive_validation_subdir(monkeypatch):
    validate_script = load_test_drive_validate_script(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["validate_test_drive_model.py"])

    args = validate_script.parse_args()

    assert args.output_dir == "outputs/validation/test_drive_validation"
