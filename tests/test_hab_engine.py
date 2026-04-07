import pytest
import pandas as pd
from src.models.hab_engine import HABDecisionEngine

def test_explain_single():
    engine = HABDecisionEngine(thresholds={"H": 0.5, "A": 0.4, "B": 0.3})
    prob_dict = {"proba_7d": 0.6, "proba_14d": 0.2, "proba_21d": 0.1}
    result = engine.explain_single(prob_dict)
    assert result["rating"] == "H"
    assert "P(7天试驾)=60.00%" in result["explanation"]

def test_predict_batch():
    engine = HABDecisionEngine(thresholds={"H": 0.5, "A": 0.4, "B": 0.3})
    df_proba = pd.DataFrame({
        "proba_7d": [0.6, 0.2, 0.1, 0.0],
        "proba_14d": [0.2, 0.5, 0.1, 0.0],
        "proba_21d": [0.1, 0.1, 0.4, 0.0]
    })
    result = engine.predict_batch(df_proba)
    assert list(result["ratings"]) == ["H", "A", "B", "N"]
