import pytest
import pandas as pd
from src.models.hab_engine import HABDecisionEngine

def test_explain_single():
    engine = HABDecisionEngine(thresholds={"H": 0.5, "A": 0.4, "B": 0.3})
    # 使用模型实际输出的列名 H/A/B
    prob_dict = {"H": 0.6, "A": 0.2, "B": 0.1, "O": 0.1}
    result = engine.explain_single(prob_dict)
    assert result["rating"] == "H"
    assert "P(H)=60.00%" in result["explanation"]

def test_predict_batch():
    engine = HABDecisionEngine(thresholds={"H": 0.5, "A": 0.4, "B": 0.3})
    # 使用模型实际输出的列名 H/A/B
    df_proba = pd.DataFrame({
        "H": [0.6, 0.2, 0.1, 0.0],
        "A": [0.2, 0.5, 0.1, 0.0],
        "B": [0.1, 0.1, 0.4, 0.0],
        "O": [0.1, 0.2, 0.4, 1.0]
    })
    result = engine.predict_batch(df_proba)
    assert list(result["ratings"]) == ["H", "A", "B", "N"]
