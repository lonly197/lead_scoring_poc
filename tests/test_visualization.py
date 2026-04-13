import logging
import warnings

import pytest

# visualization.py needs real matplotlib; skip if not available
pytest.importorskip("matplotlib")

import pandas as pd

import src.utils.visualization as visualization


def test_plot_feature_importance_suppresses_glyph_warnings(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(visualization, "_available_font_names", lambda: set())
    monkeypatch.setattr(visualization, "_FONT_CONFIGURED", False)
    monkeypatch.setattr(visualization, "_HAS_CHINESE_FONT", False)

    importance_df = pd.DataFrame(
        [
            {"feature": "所在城市", "importance": 0.4},
            {"feature": "提及试驾", "importance": 0.3},
        ]
    )

    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        visualization.plot_feature_importance(importance_df, str(tmp_path / "feature_importance.png"))

    assert (tmp_path / "feature_importance.png").exists()
    assert any("未检测到可用中文字体" in record.getMessage() for record in caplog.records)
    assert not any("Glyph" in str(warning.message) for warning in caught)
    assert not any(isinstance(warning.message, FutureWarning) for warning in caught)
