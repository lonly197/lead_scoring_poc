import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


def load_validate_script(monkeypatch):
    pandas_module = types.ModuleType("pandas")

    class DummyDataFrame:
        pass

    class DummySeries:
        pass

    pandas_module.DataFrame = DummyDataFrame
    pandas_module.Series = DummySeries
    pandas_module.Timestamp = lambda value: value
    pandas_module.to_datetime = lambda value, errors=None: value

    autogluon_module = types.ModuleType("autogluon")
    tabular_module = types.ModuleType("autogluon.tabular")

    class DummyTabularPredictor:
        @staticmethod
        def load(*args, **kwargs):
            raise AssertionError("TabularPredictor.load should not be called in CLI tests")

    tabular_module.TabularPredictor = DummyTabularPredictor
    autogluon_module.tabular = tabular_module

    config_module = types.ModuleType("config.config")
    config_module.config = types.SimpleNamespace(
        feature=types.SimpleNamespace(
            time_columns=[],
            numeric_features=[],
        )
    )
    config_module.get_excluded_columns = lambda target: []

    label_policy_module = types.ModuleType("src.data.label_policy")
    label_policy_module.apply_ohab_label_policy = lambda df, target, policy: df
    label_policy_module.filter_to_effective_ohab_labels = lambda df, target, policy: df

    loader_module = types.ModuleType("src.data.loader")

    class DummyDataLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            raise AssertionError("DataLoader.load should not be called in CLI tests")

    class DummyFeatureEngineer:
        def __init__(self, *args, **kwargs):
            pass

        def transform(self, df, interaction_context=None):
            return df, {}

    loader_module.DataLoader = DummyDataLoader
    loader_module.FeatureEngineer = DummyFeatureEngineer

    business_logic_module = types.ModuleType("src.evaluation.business_logic")
    business_logic_module.build_bucket_summary_text = lambda records: []
    business_logic_module.build_lead_action_record = lambda **kwargs: {}

    ohab_metrics_module = types.ModuleType("src.evaluation.ohab_metrics")
    ohab_metrics_module.apply_hab_decision_policy = lambda y_proba, policy: y_proba
    ohab_metrics_module.classification_report_dict = lambda y_true, y_pred: {}
    ohab_metrics_module.classification_report_text = lambda y_true, y_pred: ""
    ohab_metrics_module.compute_hab_bucket_summary = lambda *args, **kwargs: DummyDataFrame()
    ohab_metrics_module.compute_class_ranking_report = lambda *args, **kwargs: {}
    ohab_metrics_module.compute_threshold_report = lambda *args, **kwargs: None
    ohab_metrics_module.confusion_matrix_frame = lambda *args, **kwargs: DummyDataFrame()
    ohab_metrics_module.check_hab_monotonicity = lambda *args, **kwargs: {"passed": False}

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
    monkeypatch.setitem(sys.modules, "autogluon", autogluon_module)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_module)
    monkeypatch.setitem(sys.modules, "config.config", config_module)
    monkeypatch.setitem(sys.modules, "src.data.label_policy", label_policy_module)
    monkeypatch.setitem(sys.modules, "src.data.loader", loader_module)
    monkeypatch.setitem(sys.modules, "src.evaluation.business_logic", business_logic_module)
    monkeypatch.setitem(sys.modules, "src.evaluation.ohab_metrics", ohab_metrics_module)
    monkeypatch.setitem(sys.modules, "src.utils.helpers", helpers_module)

    module_name = "scripts.validate_model"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_model.py"

    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_model_run_background_strips_daemon_and_appends_log_file(monkeypatch, tmp_path):
    validate_script = load_validate_script(monkeypatch)

    captured = {}

    class DummyProcess:
        pid = 43210

    def fake_popen(cmd, stdout, stderr, start_new_session, env):
        captured["cmd"] = cmd
        captured["stdout_name"] = stdout.name
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        captured["env_flag"] = env.get("LEAD_SCORING_DISABLE_CONSOLE_LOG")
        return DummyProcess()

    monkeypatch.setattr(validate_script.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(validate_script, "get_timestamp", lambda: "20260325_120000")
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_model.py", "--daemon", "--data-path", "./data/demo.tsv", "--oot-test"],
    )
    monkeypatch.chdir(tmp_path)

    pid = validate_script.run_background(
        Namespace(
            daemon=True,
            log_file=None,
            data_path="./data/demo.tsv",
            oot_test=True,
            model_path="outputs/models/ohab_model",
            target="线索评级_试驾前",
            output_dir="outputs/validation",
            train_end="2026-03-11",
            valid_end="2026-03-16",
            report_topk="5,10,20",
        )
    )

    assert pid == 43210
    assert "--daemon" not in captured["cmd"]
    assert "-d" not in captured["cmd"]
    assert captured["cmd"][-2:] == ["--log-file", "outputs/logs/validate_model_20260325_120000.log"]
    assert captured["stdout_name"].endswith("outputs/logs/validate_model_20260325_120000.log")
    assert captured["start_new_session"] is True
    assert captured["env_flag"] == "1"


def test_validate_model_main_uses_background_branch(monkeypatch):
    validate_script = load_validate_script(monkeypatch)
    captured = {}

    def fake_run_background(args):
        captured["daemon"] = args.daemon
        captured["data_path"] = args.data_path
        return 2468

    monkeypatch.setattr(validate_script, "run_background", fake_run_background)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_model.py", "--daemon", "--data-path", "./data/202602~03.tsv"],
    )

    validate_script.main()

    assert captured == {
        "daemon": True,
        "data_path": "./data/202602~03.tsv",
    }


def test_validate_model_artifacts_rejects_missing_artifact_status(monkeypatch):
    validate_script = load_validate_script(monkeypatch)

    with pytest.raises(RuntimeError, match="缺少 artifact_status"):
        validate_script.validate_model_artifacts({}, Path("outputs/models/ohab_model"))


def test_validate_model_artifacts_rejects_incomplete_training(monkeypatch):
    validate_script = load_validate_script(monkeypatch)

    with pytest.raises(RuntimeError, match="尚未完成训练"):
        validate_script.validate_model_artifacts(
            {
                "artifact_status": {
                    "training_complete": False,
                    "comparison_expected": True,
                    "supplemental_failures": [],
                    "completed_at": None,
                }
            },
            Path("outputs/models/ohab_model"),
        )


def test_validate_model_artifacts_rejects_missing_comparison_models(monkeypatch):
    validate_script = load_validate_script(monkeypatch)

    with pytest.raises(RuntimeError, match="缺少 baseline/best 对比元数据"):
        validate_script.validate_model_artifacts(
            {
                "artifact_status": {
                    "training_complete": True,
                    "comparison_expected": True,
                    "supplemental_failures": [],
                    "completed_at": "2026-03-25T20:00:00+08:00",
                },
                "model_comparison": {"enabled": True, "models": {}},
            },
            Path("outputs/models/ohab_model"),
        )


def test_validate_model_artifacts_accepts_complete_metadata(monkeypatch):
    validate_script = load_validate_script(monkeypatch)

    validate_script.validate_model_artifacts(
        {
            "artifact_status": {
                "training_complete": True,
                "comparison_expected": True,
                "supplemental_failures": ["feature_importance"],
                "completed_at": "2026-03-25T20:00:00+08:00",
            },
            "model_comparison": {
                "enabled": True,
                "models": {
                    "BaselineModel": {"role": "baseline"},
                    "BestModel": {"role": "best"},
                },
            },
        },
        Path("outputs/models/ohab_model"),
    )
