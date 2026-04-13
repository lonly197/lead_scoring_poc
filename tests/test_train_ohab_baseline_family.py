"""Tests for _candidate_prefixes_for_family from train_ohab.py.

This function is pure logic (no autogluon/matplotlib dependencies),
but train_ohab.py imports heavy modules at the module level, so we
skip when those dependencies are unavailable.
"""
import pytest

pytest.importorskip("autogluon.tabular")
pytest.importorskip("matplotlib")

import sys
import types

sklearn_module = types.ModuleType("sklearn")
metrics_module = types.ModuleType("sklearn.metrics")
metrics_module.balanced_accuracy_score = lambda *args, **kwargs: 0.0
metrics_module.classification_report = lambda *args, **kwargs: {}
metrics_module.confusion_matrix = lambda *args, **kwargs: []
metrics_module.f1_score = lambda *args, **kwargs: 0.0
metrics_module.recall_score = lambda *args, **kwargs: 0.0
sklearn_module.metrics = metrics_module
sys.modules.setdefault("sklearn", sklearn_module)
sys.modules.setdefault("sklearn.metrics", metrics_module)

from scripts.training.train_ohab import _candidate_prefixes_for_family


def test_candidate_prefixes_accepts_gbdt_alias():
    assert _candidate_prefixes_for_family("gbdt") == (
        "LightGBM",
        "LightGBMXT",
        "LightGBMLarge",
    )


def test_candidate_prefixes_accepts_lightgbm_alias():
    assert _candidate_prefixes_for_family("lightgbm") == (
        "LightGBM",
        "LightGBMXT",
        "LightGBMLarge",
    )
