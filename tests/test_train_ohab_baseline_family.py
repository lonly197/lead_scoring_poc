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

matplotlib_module = types.ModuleType("matplotlib")
pyplot_module = types.ModuleType("matplotlib.pyplot")
pyplot_module.figure = lambda *args, **kwargs: None
pyplot_module.savefig = lambda *args, **kwargs: None
pyplot_module.close = lambda *args, **kwargs: None
pyplot_module.rcParams = {}
font_manager_module = types.ModuleType("matplotlib.font_manager")
font_manager_module.fontManager = types.SimpleNamespace(ttflist=[])
matplotlib_module.pyplot = pyplot_module
matplotlib_module.font_manager = font_manager_module
sys.modules.setdefault("matplotlib", matplotlib_module)
sys.modules.setdefault("matplotlib.pyplot", pyplot_module)
sys.modules.setdefault("matplotlib.font_manager", font_manager_module)

from scripts.train_ohab import _candidate_prefixes_for_family


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
