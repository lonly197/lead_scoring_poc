from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class HABDecisionEngine:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None, mode: str = "two_stage"):
        self.thresholds = thresholds or {"H": 0.5, "A": 0.5, "B": 0.5}
        self.mode = mode

    def explain_single(self, prob_dict: Dict[str, float], feature_contributions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        p7 = prob_dict.get("proba_7d", 0.0)
        p14 = prob_dict.get("proba_14d", 0.0)
        p21 = prob_dict.get("proba_21d", 0.0)
        
        if p7 >= self.thresholds.get("H", 0.5):
            rating = "H"
            explanation = f"高意向，7天内计划试驾（P(7天试驾)={p7:.2%}）"
        elif p14 >= self.thresholds.get("A", 0.5):
            rating = "A"
            explanation = f"中意向，14天内计划试驾（P(14天试驾)={p14:.2%}）"
        elif p21 >= self.thresholds.get("B", 0.5):
            rating = "B"
            explanation = f"低意向，21天内计划试驾（P(21天试驾)={p21:.2%}）"
        else:
            rating = "N"
            explanation = "无意向，暂无试驾计划"
            
        return {
            "rating": rating,
            "explanation": explanation,
            "positive_factors": [], # Simplified for plan
            "risk_factors": []      # Simplified for plan
        }

    def predict_batch(self, df_proba: pd.DataFrame, df_status: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        n = len(df_proba)
        ratings = np.full(n, "N", dtype=object)
        
        p7 = df_proba.get("proba_7d", pd.Series(np.zeros(n))).values
        p14 = df_proba.get("proba_14d", pd.Series(np.zeros(n))).values
        p21 = df_proba.get("proba_21d", pd.Series(np.zeros(n))).values
        
        t_h = self.thresholds.get("H", 0.5)
        t_a = self.thresholds.get("A", 0.5)
        t_b = self.thresholds.get("B", 0.5)
        
        ratings[p21 >= t_b] = "B"
        ratings[p14 >= t_a] = "A"
        ratings[p7 >= t_h] = "H"
        
        return {"ratings": ratings, "distribution": pd.Series(ratings).value_counts().to_dict()}
