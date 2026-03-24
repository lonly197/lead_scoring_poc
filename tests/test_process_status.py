import pytest

from pathlib import Path

from src.utils import helpers


def _read_status(info_path: Path) -> dict:
    return helpers.load_json(str(info_path))


def test_complete_process_marks_running_as_completed(monkeypatch, tmp_path):
    process_dir = tmp_path / ".process"
    monkeypatch.setattr(helpers, "PROCESS_DIR", process_dir)

    info_path = helpers.save_process_info(
        task_name="train_arrive",
        pid=12345,
        command="python scripts/train_arrive.py",
        log_file="outputs/logs/train_arrive.log",
    )

    helpers.complete_process_if_running("train_arrive", 12345)

    info = _read_status(info_path)
    assert info["status"] == "completed"
    assert "end_time" in info


@pytest.mark.parametrize("terminal_status", ["failed", "stopped"])
def test_complete_process_keeps_terminal_status(monkeypatch, tmp_path, terminal_status):
    process_dir = tmp_path / ".process"
    monkeypatch.setattr(helpers, "PROCESS_DIR", process_dir)

    info_path = helpers.save_process_info(
        task_name="train_ohab",
        pid=54321,
        command="python scripts/train_ohab.py",
        log_file="outputs/logs/train_ohab.log",
    )
    update_kwargs = {"error": "boom"} if terminal_status == "failed" else {}
    helpers.update_process_status("train_ohab", 54321, terminal_status, **update_kwargs)

    helpers.complete_process_if_running("train_ohab", 54321)

    info = _read_status(info_path)
    assert info["status"] == terminal_status
    if terminal_status == "failed":
        assert info["error"] == "boom"
