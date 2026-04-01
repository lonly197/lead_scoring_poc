"""
OHABCN 评级推导模块

支持三种预测模式：
- simple: 简单模式，使用单模型（14天试驾概率）推断评级
- medium: 中等模式，使用三模型集成（7/14/21天试驾概率）推断评级
- advanced: 高等模式，分阶段预测（试驾前+试驾后），完全符合业务规则

评级定义：
- O 级：已成交（100分）
- H 级：7天内试驾/下订（80-99分，高意向）
- A 级：14天内试驾/下订（60-79分，中意向）
- B 级：21天内试驾/下订（40-59分，低意向）
- C 级：有意向但超过21天（20-39分，超长尾意向）
- N 级：无效线索（0分，无电话、已购买竞品、明确拒绝等）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionMode(str, Enum):
    """预测模式枚举"""
    SIMPLE = "simple"      # 简单模式：单模型
    MEDIUM = "medium"      # 中等模式：试驾三模型集成
    ADVANCED = "advanced"  # 高等模式：试驾前+试驾后双阶段


# 评级分值范围定义
RATING_SCORE_RANGE = {
    "O": (100, 100),  # 已成交：固定100分
    "H": (80, 99),    # 高意向：80-99分
    "A": (60, 79),    # 中意向：60-79分
    "B": (40, 59),    # 低意向：40-59分
    "C": (20, 39),    # 超长尾意向：20-39分
    "N": (0, 0),      # 无效线索：0分
}

# 评级顺序（用于分布统计）
RATING_ORDER = ["O", "H", "A", "B", "C", "N"]


@dataclass
class OHABCNResult:
    """OHABCN 评级结果"""
    ratings: np.ndarray                    # 评级数组（O/H/A/B/C/N）
    scores: np.ndarray                     # 分值数组（0-100）
    stage: np.ndarray                      # 阶段数组（"O", "试驾前", "试驾后", "无效"）
    proba_7d: Optional[np.ndarray] = None  # 7天概率
    proba_14d: Optional[np.ndarray] = None  # 14天概率
    proba_21d: Optional[np.ndarray] = None  # 21天概率
    order_proba_7d: Optional[np.ndarray] = None   # 7天下订概率
    order_proba_14d: Optional[np.ndarray] = None  # 14天下订概率
    order_proba_21d: Optional[np.ndarray] = None  # 21天下订概率
    distribution: Dict[str, int] = field(default_factory=dict)


def calculate_score_from_proba(
    rating: str,
    proba: float,
    proba_7d: float = 0.0,
    proba_14d: float = 0.0,
    proba_21d: float = 0.0,
) -> float:
    """
    根据评级和概率计算分值

    分值计算规则：
    - O级：固定100分
    - H级：80 + proba_7d * 19（80-99分）
    - A级：60 + proba_14d * 19（60-79分）
    - B级：40 + proba_21d * 19（40-59分）
    - C级：20 + proba_21d * 19（20-39分，基于21天概率的残余意向）
    - N级：固定0分

    Args:
        rating: 评级（O/H/A/B/C/N）
        proba: 主概率（用于简单模式）
        proba_7d: 7天概率
        proba_14d: 14天概率
        proba_21d: 21天概率

    Returns:
        分值（0-100）
    """
    if rating == "O":
        return 100.0
    elif rating == "H":
        # H级：基于7天概率，范围80-99
        base_score = 80.0
        return base_score + min(proba_7d if proba_7d > 0 else proba, 1.0) * 19
    elif rating == "A":
        # A级：基于14天概率，范围60-79
        base_score = 60.0
        return base_score + min(proba_14d if proba_14d > 0 else proba, 1.0) * 19
    elif rating == "B":
        # B级：基于21天概率，范围40-59
        base_score = 40.0
        return base_score + min(proba_21d if proba_21d > 0 else proba, 1.0) * 19
    elif rating == "C":
        # C级：有残余意向，范围20-39
        base_score = 20.0
        return base_score + min(proba_21d if proba_21d > 0 else 0.1, 1.0) * 19
    else:  # N级
        return 0.0


def calculate_scores_batch(
    ratings: np.ndarray,
    proba_7d: Optional[np.ndarray] = None,
    proba_14d: Optional[np.ndarray] = None,
    proba_21d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    批量计算分值

    Args:
        ratings: 评级数组
        proba_7d: 7天概率数组
        proba_14d: 14天概率数组
        proba_21d: 21天概率数组

    Returns:
        分值数组
    """
    n_samples = len(ratings)
    scores = np.zeros(n_samples, dtype=float)

    # 初始化概率数组
    p7 = proba_7d if proba_7d is not None else np.zeros(n_samples)
    p14 = proba_14d if proba_14d is not None else np.zeros(n_samples)
    p21 = proba_21d if proba_21d is not None else np.zeros(n_samples)

    for i in range(n_samples):
        scores[i] = calculate_score_from_proba(
            rating=ratings[i],
            proba=0.5,  # 默认概率
            proba_7d=p7[i] if i < len(p7) else 0.0,
            proba_14d=p14[i] if i < len(p14) else 0.0,
            proba_21d=p21[i] if i < len(p21) else 0.0,
        )

    return scores


def detect_ordered_status(df: pd.DataFrame) -> np.ndarray:
    """
    检测已成交状态（O 级）

    根据以下字段判断线索是否已下定/成交：
    - 下订时间不为空
    - 成交标签 = 1
    - is_final_ordered = 1
    - 下定状态/订单状态包含已下定/已成交关键词
    - 意向金支付状态为已支付
    - 成交日期/结算日期不为空
    - 订单号不为空

    Args:
        df: 输入数据 DataFrame

    Returns:
        布尔数组，True 表示已成交
    """
    is_ordered = np.zeros(len(df), dtype=bool)

    # 1. 检查下订时间
    if "下订时间" in df.columns:
        is_ordered |= df["下订时间"].notna()

    # 2. 检查成交标签
    if "成交标签" in df.columns:
        is_ordered |= (df["成交标签"] == 1)

    # 3. 检查 is_final_ordered
    if "is_final_ordered" in df.columns:
        is_ordered |= (df["is_final_ordered"] == 1)

    # 4. 检查下定状态
    if "下定状态" in df.columns:
        ordered_keywords = ["已下定", "已成交", "已订车", "成交", "订车", "已下单"]
        for kw in ordered_keywords:
            is_ordered |= df["下定状态"].astype(str).str.contains(kw, na=False)

    # 5. 检查订单状态
    if "订单状态" in df.columns:
        ordered_keywords = ["已下定", "已成交", "已订车", "成交", "订车", "已完成", "已下单"]
        for kw in ordered_keywords:
            is_ordered |= df["订单状态"].astype(str).str.contains(kw, na=False)

    # 6. 检查意向金支付状态
    if "意向金支付状态" in df.columns:
        paid_keywords = ["已支付", "已付", "支付成功"]
        for kw in paid_keywords:
            is_ordered |= df["意向金支付状态"].astype(str).str.contains(kw, na=False)

    # 7. 检查成交日期
    if "成交日期" in df.columns:
        is_ordered |= df["成交日期"].notna()

    # 8. 检查结算日期
    if "结算日期" in df.columns:
        is_ordered |= df["结算日期"].notna()

    # 9. 检查订单号
    for col in ["订单号", "customer_order_no"]:
        if col in df.columns:
            is_ordered |= df[col].notna()

    return is_ordered


def detect_driven_status(df: pd.DataFrame) -> np.ndarray:
    """
    检测已试驾状态

    根据以下字段判断线索是否已试驾：
    - 试驾时间不为空
    - 试驾标签相关字段

    Args:
        df: 输入数据 DataFrame

    Returns:
        布尔数组，True 表示已试驾
    """
    is_driven = np.zeros(len(df), dtype=bool)

    # 1. 检查试驾时间
    if "试驾时间" in df.columns:
        is_driven |= df["试驾时间"].notna()

    # 2. 检查试驾状态
    if "试驾状态" in df.columns:
        driven_keywords = ["已试驾", "试驾完成", "完成试驾"]
        for kw in driven_keywords:
            is_driven |= df["试驾状态"].astype(str).str.contains(kw, na=False)

    return is_driven


def detect_invalid_status(df: pd.DataFrame) -> np.ndarray:
    """
    检测无效线索状态（N 级）

    根据以下字段判断线索是否为无效线索：
    - 无电话号码（手机号为空或无效）
    - 已购买竞品
    - 明确拒绝/战败
    - 线索评级为 N 或 无效
    - 战败原因包含竞品、拒绝等关键词

    Args:
        df: 输入数据 DataFrame

    Returns:
        布尔数组，True 表示无效线索
    """
    is_invalid = np.zeros(len(df), dtype=bool)

    # 1. 检查手机号是否为空
    phone_cols = ["手机号", "手机号_脱敏", "客户手机", "phone"]
    for col in phone_cols:
        if col in df.columns:
            # 手机号为空或全是空格
            is_invalid |= df[col].isna() | (df[col].astype(str).str.strip() == "")
            # 手机号无效（长度不足11位或全是0）
            phone_str = df[col].astype(str).str.strip()
            is_invalid |= phone_str.str.len() < 11
            is_invalid |= phone_str == "00000000000"

    # 2. 检查战败原因
    if "战败原因" in df.columns:
        defeat_keywords = [
            "已购买竞品", "购买竞品", "已购车", "购买其他品牌",
            "明确拒绝", "拒绝跟进", "无意向", "无购车意向",
            "放弃购车", "不考虑", "已取消", "无效线索",
            "竞品", "已在别处购买", "流失",
        ]
        for kw in defeat_keywords:
            is_invalid |= df["战败原因"].astype(str).str.contains(kw, na=False)

    # 3. 检查线索评级结果
    if "线索评级结果" in df.columns:
        invalid_ratings = ["N", "无效", "战败", "流失"]
        for rating in invalid_ratings:
            is_invalid |= df["线索评级结果"].astype(str).str.upper() == rating.upper()

    # 4. 检查客户状态
    if "客户状态" in df.columns:
        invalid_status_keywords = ["无效", "战败", "流失", "已购车", "放弃"]
        for kw in invalid_status_keywords:
            is_invalid |= df["客户状态"].astype(str).str.contains(kw, na=False)

    # 5. 检查是否已购买竞品标记
    if "已购买竞品" in df.columns:
        is_invalid |= (df["已购买竞品"] == 1) | (df["已购买竞品"].astype(str).str.upper() == "是")

    # 6. 检查线索类型是否为无效
    if "线索类型" in df.columns:
        invalid_types = ["无效", "战败", "重复", "测试"]
        for t in invalid_types:
            is_invalid |= df["线索类型"].astype(str).str.contains(t, na=False)

    return is_invalid


def derive_ohabcn_simple(
    y_proba: np.ndarray,
    is_ordered: np.ndarray,
    is_invalid: np.ndarray,
    threshold: float = 0.5,
) -> OHABCNResult:
    """
    简单模式 OHABCN 推导

    使用单个模型（14天试驾概率）推断评级。

    推导逻辑：
    - O 级：已成交状态（优先级最高）
    - N 级：无效线索（次高优先级）
    - H 级：P >= threshold（高意向）
    - A 级：threshold * 0.7 <= P < threshold（中意向）
    - B 级：threshold * 0.3 <= P < threshold * 0.7（低意向）
    - C 级：P < threshold * 0.3（超长尾意向，有意向但概率低）

    Args:
        y_proba: 预测概率数组（14 天试驾概率）
        is_ordered: 已成交状态数组
        is_invalid: 无效线索状态数组
        threshold: 评级判定阈值

    Returns:
        OHABCNResult 评级结果
    """
    n_samples = len(y_proba)
    ratings = np.full(n_samples, "C", dtype=object)  # 默认 C 级
    stage = np.full(n_samples, "试驾前", dtype=object)

    # 1. O 级：已成交（最高优先级）
    ratings[is_ordered] = "O"
    stage[is_ordered] = "O"

    # 2. N 级：无效线索（次高优先级，排除已成交）
    invalid_mask = is_invalid & ~is_ordered
    ratings[invalid_mask] = "N"
    stage[invalid_mask] = "无效"

    # 3. 有效样本：根据概率判断 H/A/B/C
    valid_mask = ~is_ordered & ~is_invalid

    # H 级：高意向
    h_threshold = threshold
    h_mask = valid_mask & (y_proba >= h_threshold)
    ratings[h_mask] = "H"

    # A 级：中意向
    a_threshold = threshold * 0.7
    a_mask = valid_mask & (y_proba >= a_threshold) & (y_proba < h_threshold)
    ratings[a_mask] = "A"

    # B 级：低意向
    b_threshold = threshold * 0.3
    b_mask = valid_mask & (y_proba >= b_threshold) & (y_proba < a_threshold)
    ratings[b_mask] = "B"

    # C 级：超长尾意向（已默认）

    # 计算分值
    scores = calculate_scores_batch(
        ratings=ratings,
        proba_14d=y_proba,
    )

    # 统计分布
    distribution = {r: int((ratings == r).sum()) for r in RATING_ORDER}

    return OHABCNResult(
        ratings=ratings,
        scores=scores,
        stage=stage,
        proba_14d=y_proba,
        distribution=distribution,
    )


def derive_ohabcn_medium(
    proba_7d: np.ndarray,
    proba_14d: np.ndarray,
    proba_21d: np.ndarray,
    is_ordered: np.ndarray,
    is_invalid: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> OHABCNResult:
    """
    中等模式 OHABCN 推导

    使用三模型集成（7/14/21天试驾概率）推断评级。
    完全符合试驾前邀约阶段的业务规则。

    业务规则：
    - H 级：计划 7 天内试驾（P(7天) >= threshold）
    - A 级：计划 14 天内试驾（P(14天) >= threshold 且非 H）
    - B 级：计划 21 天内试驾（P(21天) >= threshold 且非 H/A）
    - C 级：有意向但概率低于阈值
    - N 级：无效线索

    Args:
        proba_7d: 7天试驾概率数组
        proba_14d: 14天试驾概率数组
        proba_21d: 21天试驾概率数组
        is_ordered: 已成交状态数组
        is_invalid: 无效线索状态数组
        thresholds: 各级别阈值，默认 {"H": 0.5, "A": 0.5, "B": 0.5}

    Returns:
        OHABCNResult 评级结果
    """
    thresholds = thresholds or {"H": 0.5, "A": 0.5, "B": 0.5}
    n_samples = len(proba_7d)
    ratings = np.full(n_samples, "C", dtype=object)  # 默认 C 级
    stage = np.full(n_samples, "试驾前", dtype=object)

    # 1. O 级：已成交（最高优先级）
    ratings[is_ordered] = "O"
    stage[is_ordered] = "O"

    # 2. N 级：无效线索（次高优先级，排除已成交）
    invalid_mask = is_invalid & ~is_ordered
    ratings[invalid_mask] = "N"
    stage[invalid_mask] = "无效"

    # 3. 有效样本：按业务规则推导
    valid_mask = ~is_ordered & ~is_invalid

    # H 级：7天内试驾概率 >= threshold
    h_mask = valid_mask & (proba_7d >= thresholds["H"])
    ratings[h_mask] = "H"

    # A 级：非 H，但 14天内试驾概率 >= threshold
    remaining = valid_mask & (ratings == "C")
    a_mask = remaining & (proba_14d >= thresholds["A"])
    ratings[a_mask] = "A"

    # B 级：非 H/A，但 21天内试驾概率 >= threshold
    remaining = valid_mask & (ratings == "C")
    b_mask = remaining & (proba_21d >= thresholds["B"])
    ratings[b_mask] = "B"

    # C 级：已默认（有意向但概率低）

    # 计算分值
    scores = calculate_scores_batch(
        ratings=ratings,
        proba_7d=proba_7d,
        proba_14d=proba_14d,
        proba_21d=proba_21d,
    )

    # 统计分布
    distribution = {r: int((ratings == r).sum()) for r in RATING_ORDER}

    return OHABCNResult(
        ratings=ratings,
        scores=scores,
        stage=stage,
        proba_7d=proba_7d,
        proba_14d=proba_14d,
        proba_21d=proba_21d,
        distribution=distribution,
    )


def derive_ohabcn_advanced(
    # 试驾前邀约阶段模型概率
    drive_proba_7d: np.ndarray,
    drive_proba_14d: np.ndarray,
    drive_proba_21d: np.ndarray,
    # 试驾后下订商谈阶段模型概率
    order_proba_7d: np.ndarray,
    order_proba_14d: np.ndarray,
    order_proba_21d: np.ndarray,
    # 状态标记
    is_ordered: np.ndarray,
    is_driven: np.ndarray,
    is_invalid: np.ndarray,
    # 阈值
    thresholds: Optional[Dict[str, float]] = None,
) -> OHABCNResult:
    """
    高等模式 OHABCN 推导

    分阶段预测，完全符合业务规则：
    - 已成交 → O 级
    - 无效线索 → N 级
    - 已试驾未成交 → 使用试驾后下订商谈阶段模型（7/14/21天下订概率）
    - 未试驾未成交 → 使用试驾前邀约阶段模型（7/14/21天试驾概率）

    业务规则：
    试驾前邀约阶段：
    - H 级：7天内试驾
    - A 级：14天内试驾
    - B 级：21天内试驾

    试驾后下订商谈阶段：
    - H 级：7天内下订
    - A 级：14天内下订
    - B 级：21天内下订

    Args:
        drive_proba_7d: 7天试驾概率数组
        drive_proba_14d: 14天试驾概率数组
        drive_proba_21d: 21天试驾概率数组
        order_proba_7d: 7天下订概率数组
        order_proba_14d: 14天下订概率数组
        order_proba_21d: 21天下订概率数组
        is_ordered: 已成交状态数组
        is_driven: 已试驾状态数组
        is_invalid: 无效线索状态数组
        thresholds: 各级别阈值

    Returns:
        OHABCNResult 评级结果
    """
    thresholds = thresholds or {"H": 0.5, "A": 0.5, "B": 0.5}
    n_samples = len(drive_proba_7d)
    ratings = np.full(n_samples, "C", dtype=object)  # 默认 C 级
    stage = np.full(n_samples, "试驾前", dtype=object)

    # 1. O 级：已成交（最高优先级）
    ratings[is_ordered] = "O"
    stage[is_ordered] = "O"

    # 2. N 级：无效线索（次高优先级，排除已成交）
    invalid_mask = is_invalid & ~is_ordered
    ratings[invalid_mask] = "N"
    stage[invalid_mask] = "无效"

    # 3. 已试驾未成交 → 试驾后下订商谈阶段
    driven_not_ordered = is_driven & ~is_ordered & ~is_invalid
    stage[driven_not_ordered] = "试驾后"

    # 使用下订概率推导
    # H 级：7天内下订概率 >= threshold
    h_mask = driven_not_ordered & (order_proba_7d >= thresholds["H"])
    ratings[h_mask] = "H"

    # A 级：非 H，但 14天内下订概率 >= threshold
    remaining = driven_not_ordered & (ratings == "C")
    a_mask = remaining & (order_proba_14d >= thresholds["A"])
    ratings[a_mask] = "A"

    # B 级：非 H/A，但 21天内下订概率 >= threshold
    remaining = driven_not_ordered & (ratings == "C")
    b_mask = remaining & (order_proba_21d >= thresholds["B"])
    ratings[b_mask] = "B"

    # 4. 未试驾未成交 → 试驾前邀约阶段
    not_driven_not_ordered = ~is_driven & ~is_ordered & ~is_invalid

    # 使用试驾概率推导
    # H 级：7天内试驾概率 >= threshold
    h_mask = not_driven_not_ordered & (drive_proba_7d >= thresholds["H"])
    ratings[h_mask] = "H"

    # A 级：非 H，但 14天内试驾概率 >= threshold
    remaining = not_driven_not_ordered & (ratings == "C")
    a_mask = remaining & (drive_proba_14d >= thresholds["A"])
    ratings[a_mask] = "A"

    # B 级：非 H/A，但 21天内试驾概率 >= threshold
    remaining = not_driven_not_ordered & (ratings == "C")
    b_mask = remaining & (drive_proba_21d >= thresholds["B"])
    ratings[b_mask] = "B"

    # C 级：已默认（有意向但概率低）

    # 计算分值
    scores = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        if ratings[i] == "O":
            scores[i] = 100.0
        elif ratings[i] == "N":
            scores[i] = 0.0
        elif stage[i] == "试驾后":
            # 使用下订概率计算分值
            scores[i] = calculate_score_from_proba(
                rating=ratings[i],
                proba_7d=order_proba_7d[i],
                proba_14d=order_proba_14d[i],
                proba_21d=order_proba_21d[i],
            )
        else:
            # 使用试驾概率计算分值
            scores[i] = calculate_score_from_proba(
                rating=ratings[i],
                proba_7d=drive_proba_7d[i],
                proba_14d=drive_proba_14d[i],
                proba_21d=drive_proba_21d[i],
            )

    # 统计分布
    distribution = {r: int((ratings == r).sum()) for r in RATING_ORDER}

    return OHABCNResult(
        ratings=ratings,
        scores=scores,
        stage=stage,
        proba_7d=drive_proba_7d,
        proba_14d=drive_proba_14d,
        proba_21d=drive_proba_21d,
        order_proba_7d=order_proba_7d,
        order_proba_14d=order_proba_14d,
        order_proba_21d=order_proba_21d,
        distribution=distribution,
    )


class OHABCNRater:
    """
    OHABCN 评级器

    支持三种预测模式的 OHABCN 评级推导。
    """

    def __init__(
        self,
        mode: Union[str, PredictionMode] = PredictionMode.SIMPLE,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        初始化评级器

        Args:
            mode: 预测模式（simple/medium/advanced）
            thresholds: 各级别阈值 {"H": 0.5, "A": 0.5, "B": 0.5}
        """
        if isinstance(mode, str):
            mode = PredictionMode(mode.lower())
        self.mode = mode
        self.thresholds = thresholds or {"H": 0.5, "A": 0.5, "B": 0.5}

    def derive(
        self,
        df: pd.DataFrame,
        # 简单模式参数
        proba_14d: Optional[np.ndarray] = None,
        # 中等模式参数
        drive_proba_7d: Optional[np.ndarray] = None,
        drive_proba_14d: Optional[np.ndarray] = None,
        drive_proba_21d: Optional[np.ndarray] = None,
        # 高等模式额外参数
        order_proba_7d: Optional[np.ndarray] = None,
        order_proba_14d: Optional[np.ndarray] = None,
        order_proba_21d: Optional[np.ndarray] = None,
    ) -> OHABCNResult:
        """
        推导 OHABCN 评级

        Args:
            df: 输入数据 DataFrame（用于检测状态）
            proba_14d: 简单模式的 14天试驾概率
            drive_proba_7d/14d/21d: 试驾概率（中等/高等模式）
            order_proba_7d/14d/21d: 下订概率（高等模式）

        Returns:
            OHABCNResult 评级结果
        """
        # 检测状态
        is_ordered = detect_ordered_status(df)
        is_invalid = detect_invalid_status(df)

        if self.mode == PredictionMode.SIMPLE:
            if proba_14d is None:
                raise ValueError("简单模式需要提供 proba_14d 参数")
            return derive_ohabcn_simple(
                y_proba=proba_14d,
                is_ordered=is_ordered,
                is_invalid=is_invalid,
                threshold=self.thresholds.get("H", 0.5),
            )

        elif self.mode == PredictionMode.MEDIUM:
            if any(p is None for p in [drive_proba_7d, drive_proba_14d, drive_proba_21d]):
                raise ValueError("中等模式需要提供 drive_proba_7d/14d/21d 参数")
            return derive_ohabcn_medium(
                proba_7d=drive_proba_7d,
                proba_14d=drive_proba_14d,
                proba_21d=drive_proba_21d,
                is_ordered=is_ordered,
                is_invalid=is_invalid,
                thresholds=self.thresholds,
            )

        elif self.mode == PredictionMode.ADVANCED:
            if any(p is None for p in [drive_proba_7d, drive_proba_14d, drive_proba_21d,
                                        order_proba_7d, order_proba_14d, order_proba_21d]):
                raise ValueError("高等模式需要提供所有概率参数")
            # 检测已试驾状态
            is_driven = detect_driven_status(df)
            return derive_ohabcn_advanced(
                drive_proba_7d=drive_proba_7d,
                drive_proba_14d=drive_proba_14d,
                drive_proba_21d=drive_proba_21d,
                order_proba_7d=order_proba_7d,
                order_proba_14d=order_proba_14d,
                order_proba_21d=order_proba_21d,
                is_ordered=is_ordered,
                is_driven=is_driven,
                is_invalid=is_invalid,
                thresholds=self.thresholds,
            )

        else:
            raise ValueError(f"未知的预测模式: {self.mode}")

    def add_to_dataframe(
        self,
        df: pd.DataFrame,
        result: OHABCNResult,
        include_proba: bool = True,
        include_score: bool = True,
    ) -> pd.DataFrame:
        """
        将评级结果添加到 DataFrame

        Args:
            df: 原始 DataFrame
            result: 评级结果
            include_proba: 是否包含概率列
            include_score: 是否包含分值列

        Returns:
            添加了评级列的 DataFrame
        """
        df = df.copy()
        df["评级"] = result.ratings
        df["阶段"] = result.stage

        if include_score:
            df["分值"] = result.scores

        if include_proba:
            if result.proba_7d is not None:
                df["试驾概率_7天"] = result.proba_7d
            if result.proba_14d is not None:
                df["试驾概率_14天"] = result.proba_14d
            if result.proba_21d is not None:
                df["试驾概率_21天"] = result.proba_21d
            if result.order_proba_7d is not None:
                df["下订概率_7天"] = result.order_proba_7d
            if result.order_proba_14d is not None:
                df["下订概率_14天"] = result.order_proba_14d
            if result.order_proba_21d is not None:
                df["下订概率_21天"] = result.order_proba_21d

        return df


# 向后兼容别名
OHABResult = OHABCNResult
OHABRater = OHABCNRater