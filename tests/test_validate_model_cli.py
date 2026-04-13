import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path


def load_validate_entry(monkeypatch):
    helpers_module = types.ModuleType("src.utils.helpers")
    helpers_module.get_timestamp = lambda: "20260325_120000"
    helpers_module.get_local_now = lambda: None
    helpers_module.format_timestamp = lambda dt: "2026-03-25 12:00:00+0800"

    monkeypatch.setitem(sys.modules, "src.utils.helpers", helpers_module)

    module_name = "scripts.validate_model"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validation" / "validate_model.py"

    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_model_run_background_strips_daemon_and_appends_log_file(monkeypatch, tmp_path):
    validate_script = load_validate_entry(monkeypatch)

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
    monkeypatch.chdir(tmp_path)

    pid = validate_script.run_background(
        script_path="scripts/validate_arrive_model.py",
        args=[
            "--daemon",
            "--data-path",
            "./data/demo.tsv",
            "--model-type",
            "arrive",
        ],
    )

    assert pid == 43210
    assert "--daemon" not in captured["cmd"]
    assert "-d" not in captured["cmd"]
    assert "--model-type" not in captured["cmd"]
    assert captured["cmd"][-2:] == ["--log-file", "outputs/logs/validate_arrive_model_20260325_120000.log"]
    assert captured["stdout_name"].endswith("outputs/logs/validate_arrive_model_20260325_120000.log")
    assert captured["start_new_session"] is True
    assert captured["env_flag"] == "1"


def test_validate_model_main_dispatches_arrive_to_arrive_validator(monkeypatch):
    validate_script = load_validate_entry(monkeypatch)
    captured = {}

    monkeypatch.setattr(validate_script, "resolve_validator_script", lambda args: "scripts/validate_arrive_model.py")
    monkeypatch.setattr(
        validate_script,
        "run_foreground",
        lambda script_path, pass_args: captured.update({"script_path": script_path, "pass_args": pass_args}) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["validate_model.py", "--model-type", "arrive", "--data-path", "./arrive.tsv"])

    validate_script.main()

    assert captured["script_path"].endswith("scripts/validate_arrive_model.py")
    assert captured["pass_args"] == ["--data-path", "./arrive.tsv"]


def test_validate_model_main_dispatches_ohab_to_ohab_validator(monkeypatch):
    validate_script = load_validate_entry(monkeypatch)
    captured = {}

    monkeypatch.setattr(validate_script, "resolve_validator_script", lambda args: "scripts/validate_ohab_model.py")
    monkeypatch.setattr(
        validate_script,
        "run_foreground",
        lambda script_path, pass_args: captured.update({"script_path": script_path, "pass_args": pass_args}) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["validate_model.py", "--model-type", "ohab", "--data-path", "./ohab.tsv"])

    validate_script.main()

    assert captured["script_path"].endswith("scripts/validate_ohab_model.py")
    assert captured["pass_args"] == ["--data-path", "./ohab.tsv"]


def test_validate_model_main_dispatches_test_drive_to_test_drive_validator(monkeypatch):
    validate_script = load_validate_entry(monkeypatch)
    captured = {}

    monkeypatch.setattr(validate_script, "resolve_validator_script", lambda args: "scripts/validate_test_drive_model.py")
    monkeypatch.setattr(
        validate_script,
        "run_foreground",
        lambda script_path, pass_args: captured.update({"script_path": script_path, "pass_args": pass_args}) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["validate_model.py", "--model-type", "test_drive", "--data-path", "./test_drive.tsv"])

    validate_script.main()

    assert captured["script_path"].endswith("scripts/validate_test_drive_model.py")
    assert captured["pass_args"] == ["--data-path", "./test_drive.tsv"]


def test_validate_model_main_uses_background_branch(monkeypatch):
    validate_script = load_validate_entry(monkeypatch)
    captured = {}

    monkeypatch.setattr(validate_script, "resolve_validator_script", lambda args: "scripts/validate_ohab_model.py")
    monkeypatch.setattr(
        validate_script,
        "run_background",
        lambda script_path, args: captured.update({"script_path": script_path, "args": args}) or 2468,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_model.py", "--daemon", "--model-type", "ohab", "--data-path", "./data/202602~03.tsv"],
    )

    validate_script.main()

    assert captured["script_path"].endswith("scripts/validate_ohab_model.py")
    assert captured["args"] == ["--data-path", "./data/202602~03.tsv"]


def test_resolve_validator_script_uses_explicit_model_type(monkeypatch):
    validate_script = load_validate_entry(monkeypatch)

    script_path = validate_script.resolve_validator_script(
        Namespace(model_type="arrive", model_path="outputs/models/arrive_model")
    )

    assert script_path.endswith("scripts/validation/validate_arrive_model.py")


def test_resolve_validator_script_uses_test_drive_model_type(monkeypatch):
    validate_script = load_validate_entry(monkeypatch)

    script_path = validate_script.resolve_validator_script(
        Namespace(model_type="test_drive", model_path="outputs/models/test_drive_model")
    )

    assert script_path.endswith("scripts/validation/validate_test_drive_model.py")


def test_validate_ohab_parse_args_uses_ohab_validation_subdir(monkeypatch):
    helpers_module = types.ModuleType("src.utils.helpers")
    helpers_module.complete_process_if_running = lambda *args, **kwargs: None
    helpers_module.format_training_duration = lambda seconds: f"{seconds}s"
    helpers_module.get_timestamp = lambda: "20260325_120000"
    helpers_module.get_local_now = lambda: None
    helpers_module.format_timestamp = lambda dt: "2026-03-25 12:00:00+0800"
    helpers_module.save_process_info = lambda *args, **kwargs: None
    helpers_module.setup_logging = lambda *args, **kwargs: None
    helpers_module.update_process_status = lambda *args, **kwargs: None

    config_module = types.ModuleType("config.config")
    config_module.config = types.SimpleNamespace(data=types.SimpleNamespace(data_path="./data/demo.tsv"))
    config_module.get_excluded_columns = lambda target: []

    dummy_module_names = [
        "pandas",
        "autogluon",
        "autogluon.tabular",
        "src.data.feature_screening",
        "src.data.label_policy",
        "src.data.loader",
        "src.evaluation.comparison",
        "src.models.predictor",
        "src.evaluation.scorecard",
        "src.training.hab_pipeline",
        "src.evaluation.business_logic",
        "src.evaluation.ohab_metrics",
    ]

    monkeypatch.setitem(sys.modules, "src.utils.helpers", helpers_module)
    monkeypatch.setitem(sys.modules, "config.config", config_module)
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    monkeypatch.setitem(sys.modules, "autogluon", types.ModuleType("autogluon"))
    tabular_module = types.ModuleType("autogluon.tabular")
    tabular_module.TabularPredictor = object
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_module)

    for name in dummy_module_names[3:]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    screen_module = sys.modules["src.data.feature_screening"]
    screen_module.apply_post_feature_screening_report = lambda *args, **kwargs: None
    screen_module.apply_raw_schema_report = lambda *args, **kwargs: None
    screen_module.apply_screening_report = lambda *args, **kwargs: None

    label_module = sys.modules["src.data.label_policy"]
    label_module.apply_ohab_label_policy = lambda *args, **kwargs: None
    label_module.filter_to_effective_ohab_labels = lambda *args, **kwargs: None

    loader_module = sys.modules["src.data.loader"]
    loader_module.DataLoader = object
    loader_module.FeatureEngineer = object

    comparison_module = sys.modules["src.evaluation.comparison"]
    comparison_module.build_comparator_bundle = lambda *args, **kwargs: None

    predictor_module = sys.modules["src.models.predictor"]
    predictor_module.LeadScoringPredictor = object

    scorecard_module = sys.modules["src.evaluation.scorecard"]
    scorecard_module.build_trimmed_scorecard_probability_frame = lambda *args, **kwargs: None
    scorecard_module.score_trimmed_hab_scorecard = lambda *args, **kwargs: None

    pipeline_module = sys.modules["src.training.hab_pipeline"]
    pipeline_module.combine_stage_predictions = lambda *args, **kwargs: None

    business_module = sys.modules["src.evaluation.business_logic"]
    business_module.build_bucket_summary_text = lambda *args, **kwargs: None
    business_module.build_lead_action_record = lambda *args, **kwargs: None

    metrics_module = sys.modules["src.evaluation.ohab_metrics"]
    metrics_module.apply_hab_decision_policy = lambda *args, **kwargs: None
    metrics_module.classification_report_dict = lambda *args, **kwargs: None
    metrics_module.classification_report_text = lambda *args, **kwargs: None
    metrics_module.compute_hab_bucket_summary = lambda *args, **kwargs: None
    metrics_module.compute_class_ranking_report = lambda *args, **kwargs: None
    metrics_module.compute_threshold_report = lambda *args, **kwargs: None
    metrics_module.confusion_matrix_frame = lambda *args, **kwargs: None
    metrics_module.check_hab_monotonicity = lambda *args, **kwargs: None

    module_name = "scripts.validate_ohab_model"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validation" / "validate_ohab_model.py"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", ["validate_ohab_model.py"])

    args = module.parse_args()

    assert args.output_dir == "outputs/validation/ohab_validation"
