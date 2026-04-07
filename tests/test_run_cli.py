import sys
from pathlib import Path

import pytest

from scripts import run as run_script


def test_run_py_passes_test_size_for_test_drive(monkeypatch):
    captured = {}

    def fake_run_foreground(script_path: str, args: list[str]) -> int:
        captured["script_path"] = script_path
        captured["args"] = args
        return 0

    monkeypatch.setattr(run_script, "run_foreground", fake_run_foreground)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "train_test_drive", "--test-size", "0.3"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_script.main()

    assert exc_info.value.code == 0
    assert captured["script_path"].endswith("train_test_drive.py")
    assert captured["args"] == ["--test-size", "0.3"]

def test_run_py_passes_ohab_specific_args(monkeypatch):
    captured = {}

    def fake_run_foreground(script_path: str, args: list[str]) -> int:
        captured["script_path"] = script_path
        captured["args"] = args
        return 0

    monkeypatch.setattr(run_script, "run_foreground", fake_run_foreground)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "train_ohab",
            "--label-mode",
            "hab",
            "--enable-model-comparison",
            "--baseline-family",
            "gbm",
            "--train-end",
            "2026-03-15",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_script.main()

    assert exc_info.value.code == 0
    assert captured["script_path"].endswith("train_ohab.py")
    assert captured["args"] == [
        "--label-mode",
        "hab",
        "--enable-model-comparison",
        "--baseline-family",
        "gbm",
        "--train-end",
        "2026-03-15",
    ]


def test_run_py_passes_ohab_resource_profile_args(monkeypatch):
    captured = {}

    def fake_run_foreground(script_path: str, args: list[str]) -> int:
        captured["script_path"] = script_path
        captured["args"] = args
        return 0

    monkeypatch.setattr(run_script, "run_foreground", fake_run_foreground)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "train_ohab",
            "--training-profile",
            "server_16g_compare",
            "--memory-limit-gb",
            "12",
            "--fit-strategy",
            "sequential",
            "--excluded-model-types",
            "RF,XT,KNN",
            "--num-folds-parallel",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_script.main()

    assert exc_info.value.code == 0
    assert captured["script_path"].endswith("train_ohab.py")
    assert captured["args"] == [
        "--training-profile",
        "server_16g_compare",
        "--memory-limit-gb",
        "12",
        "--fit-strategy",
        "sequential",
        "--excluded-model-types",
        "RF,XT,KNN",
        "--num-folds-parallel",
        "1",
    ]


def test_run_background_disables_console_logging(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        pid = 1234

    def fake_popen(cmd, stdout, stderr, start_new_session, env):
        captured["cmd"] = cmd
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        captured["env_flag"] = env.get("LEAD_SCORING_DISABLE_CONSOLE_LOG")
        return FakeProcess()

    monkeypatch.setattr(run_script.subprocess, "Popen", fake_popen)
    pid = run_script.run_background(
        script_path=str(Path("scripts/training/train_ohab.py")),
        args=["--preset", "good_quality"],
        log_dir=str(tmp_path),
    )

    assert pid == 1234
    assert captured["env_flag"] == "1"
