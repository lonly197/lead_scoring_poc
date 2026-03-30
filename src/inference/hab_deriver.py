"""
HAB 评级推导模块

基于业务规则从试驾概率推导 H/A/B 评级。

业务规则（来自《O/H/A/B定级业务规则》）：
- H 级：客户计划 7 天内试驾，下次联络最晚时间 < 3 天
- A 级：客户计划 14 天内试驾，下次联络最晚时间 < 7 天
- B 级：客户计划 21 天内试驾，下次联络最晚时间 < 14 天

推导逻辑：
1. 若 P(7天内试驾) >= threshold → H 级
2. 若 P(14天内试驾) >= threshold 且 P(7天内试驾) < threshold → A 级
3. 若 P(21天内试驾) >= threshold 且 P(14天内试驾) < threshold → B 级
4. 所有概率 < threshold → 无意向（N 级）
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class HABRating(Enum):
    """HAB 评级枚举"""
    H = "H"  # 高意向：7天内试驾
    A = "A"  # 中意向：14天内试驾
    B = "B"  # 低意向：21天内试驾
    N = "N"  # 无意向
    O = "O"  # 已成交（终态）

    @property
    def priority(self) -> int:
        """优先级数值（越大优先级越高）"""
        priorities = {"H": 3, "A": 2, "B": 1, "N": 0, "O": 4}
        return priorities[self.value]

    @property
    def description(self) -> str:
        """评级描述"""
        descriptions = {
            "H": "高意向，7天内计划试驾",
            "A": "中意向，14天内计划试驾",
            "B": "低意向，21天内计划试驾",
            "N": "无意向，暂无试驾计划",
            "O": "已成交",
        }
        return descriptions[self.value]


@dataclass
class HABDerivationResult:
    """HAB 推导结果"""
    rating: HABRating
    prob_7d: float
    prob_14d: float
    prob_21d: float
    confidence: float
    explanation: str
    risk_factors: List[str]
    positive_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rating": self.rating.value,
            "prob_7d": self.prob_7d,
            "prob_14d": self.prob_14d,
            "prob_21d": self.prob_21d,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "risk_factors": self.risk_factors,
            "positive_factors": self.positive_factors,
        }


class HABDeriver:
    """
    HAB 评级推导器

    从三个时间窗口的试驾概率推导 H/A/B 评级。
    """

    # 默认阈值（可调整）
    DEFAULT_THRESHOLD = 0.5

    # 下次联络间隔建议（业务规则）
    NEXT_CONTACT_DAYS = {
        HABRating.H: 3,   # H 级：3 天内联络
        HABRating.A: 7,   # A 级：7 天内联络
        HABRating.B: 14,  # B 级：14 天内联络
    }

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """
        初始化推导器

        Args:
            threshold: 评级判定阈值（默认 0.5）
        """
        self.threshold = threshold

    def derive(
        self,
        prob_7d: float,
        prob_14d: float,
        prob_21d: float,
        feature_contributions: Optional[Dict[str, float]] = None,
    ) -> HABDerivationResult:
        """
        从概率推导 HAB 评级

        Args:
            prob_7d: 7天内试驾概率
            prob_14d: 14天内试驾概率
            prob_21d: 21天内试驾概率
            feature_contributions: 特征贡献度（可选，用于生成可解释性说明）

        Returns:
            HABDerivationResult 推导结果
        """
        # 评级判定（优先检查高意向）
        if prob_7d >= self.threshold:
            rating = HABRating.H
        elif prob_14d >= self.threshold:
            rating = HABRating.A
        elif prob_21d >= self.threshold:
            rating = HABRating.B
        else:
            rating = HABRating.N

        # 计算置信度（最高概率与阈值之间的差距）
        max_prob = max(prob_7d, prob_14d, prob_21d)
        confidence = min(max_prob - self.threshold + self.threshold, 1.0) if rating != HABRating.N else max_prob

        # 生成解释
        explanation = self._generate_explanation(rating, prob_7d, prob_14d, prob_21d)

        # 提取正向/风险因素
        positive_factors, risk_factors = self._extract_factors(
            rating, feature_contributions
        )

        return HABDerivationResult(
            rating=rating,
            prob_7d=prob_7d,
            prob_14d=prob_14d,
            prob_21d=prob_21d,
            confidence=confidence,
            explanation=explanation,
            risk_factors=risk_factors,
            positive_factors=positive_factors,
        )

    def derive_batch(
        self,
        prob_7d_list: List[float],
        prob_14d_list: List[float],
        prob_21d_list: List[float],
        feature_contributions_list: Optional[List[Dict[str, float]]] = None,
    ) -> List[HABDerivationResult]:
        """
        批量推导 HAB 评级

        Args:
            prob_7d_list: 7天内试驾概率列表
            prob_14d_list: 14天内试驾概率列表
            prob_21d_list: 21天内试驾概率列表
            feature_contributions_list: 特征贡献度列表（可选）

        Returns:
            List[HABDerivationResult] 推导结果列表
        """
        results = []
        n = len(prob_7d_list)

        for i in range(n):
            fc = None
            if feature_contributions_list and i < len(feature_contributions_list):
                fc = feature_contributions_list[i]

            result = self.derive(
                prob_7d=prob_7d_list[i],
                prob_14d=prob_14d_list[i],
                prob_21d=prob_21d_list[i],
                feature_contributions=fc,
            )
            results.append(result)

        return results

    def get_next_contact_days(self, rating: HABRating) -> Optional[int]:
        """
        获取下次联络建议间隔

        Args:
            rating: HAB 评级

        Returns:
            建议联络间隔天数（N 级和 O 级无建议）
        """
        return self.NEXT_CONTACT_DAYS.get(rating)

    def _generate_explanation(
        self,
        rating: HABRating,
        prob_7d: float,
        prob_14d: float,
        prob_21d: float,
    ) -> str:
        """生成评级解释"""
        desc = rating.description

        if rating == HABRating.H:
            return f"{desc}（P(7天试驾)={prob_7d:.2%}）"
        elif rating == HABRating.A:
            return f"{desc}（P(14天试驾)={prob_14d:.2%}，P(7天试驾)={prob_7d:.2%}）"
        elif rating == HABRating.B:
            return f"{desc}（P(21天试驾)={prob_21d:.2%}，P(14天试驾)={prob_14d:.2%}）"
        else:
            return f"{desc}（所有时间窗口概率均低于阈值{self.threshold:.0%}）"

    def _extract_factors(
        self,
        rating: HABRating,
        feature_contributions: Optional[Dict[str, float]],
    ) -> Tuple[List[str], List[str]]:
        """
        提取正向因素和风险因素

        Args:
            rating: HAB 评级
            feature_contributions: 特征贡献度

        Returns:
            (正向因素列表, 风险因素列表)
        """
        if not feature_contributions:
            return [], []

        # 正向因素：贡献度 > 0 的特征
        positive = [
            f"{k}: {v:.2%}"
            for k, v in feature_contributions.items()
            if v > 0.01  # 只显示显著因素
        ]

        # 风险因素：贡献度 < 0 的特征（如果有）
        risk = [
            f"{k}: {v:.2%}"
            for k, v in feature_contributions.items()
            if v < -0.01
        ]

        # 取前 3 个最重要的因素
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_positive = [
            f"{k}: {v:.2%}"
            for k, v in sorted_contributions[:3]
            if v > 0
        ]

        top_risk = [
            f"{k}: {v:.2%}"
            for k, v in sorted_contributions[:5]
            if v < 0
        ]

        return top_positive, top_risk


def derive_hab_from_models(
    model_7d_proba: List[float],
    model_14d_proba: List[float],
    model_21d_proba: List[float],
    threshold: float = 0.5,
) -> List[str]:
    """
    简化版 HAB 推导（仅返回评级字符串）

    Args:
        model_7d_proba: 7天模型预测概率列表
        model_14d_proba: 14天模型预测概率列表
        model_21d_proba: 21天模型预测概率列表
        threshold: 判定阈值

    Returns:
        List[str] 评级字符串列表（"H", "A", "B", "N"）
    """
    deriver = HABDeriver(threshold=threshold)
    results = deriver.derive_batch(model_7d_proba, model_14d_proba, model_21d_proba)
    return [r.rating.value for r in results]


def get_hab_distribution_summary(results: List[HABDerivationResult]) -> Dict[str, int]:
    """
    统计 HAB 评级分布

    Args:
        results: HAB 推导结果列表

    Returns:
        Dict[str, int] 各评级数量
    """
    distribution = {"H": 0, "A": 0, "B": 0, "N": 0, "O": 0}

    for r in results:
        distribution[r.rating.value] += 1

    return distribution