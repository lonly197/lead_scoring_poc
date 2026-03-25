import sys

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


def test_run_py_rejects_test_size_for_smart_split_tasks(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "train_arrive", "--test-size", "0.3"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_script.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--test-size 仅适用于 train_test_drive" in captured.err


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
        "--train-end",
        "2026-03-15",
        "--label-mode",
        "hab",
        "--enable-model-comparison",
        "--baseline-family",
        "gbm",
    ]
