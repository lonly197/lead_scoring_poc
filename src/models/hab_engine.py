from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class HABDecisionEngine:
    """
    H/A/B 评级决策引擎

    业务规则：
    - H 级：7天内试驾/下订（高意向）
    - A 级：14天内试驾/下订（中意向）
    - B 级：21天内试驾/下订（低意向）

    模型输出的概率列名：H, A, B, O
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None, mode: str = "two_stage"):
        self.thresholds = thresholds or {"H": 0.5, "A": 0.4, "B": 0.3}
        self.mode = mode

    def explain_single(self, prob_dict: Dict[str, float], feature_contributions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        解释单条预测结果

        Args:
            prob_dict: 概率字典，键为 H/A/B/O
            feature_contributions: 特征贡献度（可选）

        Returns:
            包含评级、解释、正向因素、风险因素的字典
        """
        # 使用模型输出的 H/A/B 概率
        h_prob = prob_dict.get("H", 0.0)
        a_prob = prob_dict.get("A", 0.0)
        b_prob = prob_dict.get("B", 0.0)

        # 按阈值判断等级（优先级：H > A > B）
        if h_prob >= self.thresholds.get("H", 0.5):
            rating = "H"
            explanation = f"高意向，7天内计划试驾（P(H)={h_prob:.2%}）"
        elif a_prob >= self.thresholds.get("A", 0.4):
            rating = "A"
            explanation = f"中意向，14天内计划试驾（P(A)={a_prob:.2%}）"
        elif b_prob >= self.thresholds.get("B", 0.3):
            rating = "B"
            explanation = f"低意向，21天内计划试驾（P(B)={b_prob:.2%}）"
        else:
            rating = "N"
            explanation = "无意向，暂无试驾计划"

        return {
            "rating": rating,
            "explanation": explanation,
            "probabilities": {"H": h_prob, "A": a_prob, "B": b_prob},
            "positive_factors": [],  # TODO: 从 feature_contributions 提取
            "risk_factors": []       # TODO: 从 feature_contributions 提取
        }

    def predict_batch(self, df_proba: pd.DataFrame, df_status: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        批量预测评级

        Args:
            df_proba: 概率 DataFrame，包含 H/A/B/O 列
            df_status: 状态 DataFrame（可选，用于特殊处理 O 级）

        Returns:
            包含评级数组和分布统计的字典
        """
        n = len(df_proba)
        ratings = np.full(n, "N", dtype=object)

        # 获取概率列（使用模型输出的 H/A/B 列名）
        h_prob = df_proba.get("H", pd.Series(np.zeros(n))).values
        a_prob = df_proba.get("A", pd.Series(np.zeros(n))).values
        b_prob = df_proba.get("B", pd.Series(np.zeros(n))).values

        t_h = self.thresholds.get("H", 0.5)
        t_a = self.thresholds.get("A", 0.4)
        t_b = self.thresholds.get("B", 0.3)

        # 按优先级赋值（H > A > B）
        ratings[b_prob >= t_b] = "B"
        ratings[a_prob >= t_a] = "A"
        ratings[h_prob >= t_h] = "H"

        return {
            "ratings": ratings,
            "distribution": pd.Series(ratings).value_counts().to_dict(),
            "probabilities": pd.DataFrame({
                "H": h_prob,
                "A": a_prob,
                "B": b_prob
            })
        }