"""
OHAB 评级推导模块

支持三种预测模式：
- simple: 简单模式，使用单模型（14天试驾概率）推断 OHAB
- medium: 中等模式，使用三模型集成（7/14/21天试驾概率）推断 OHAB
- advanced: 高等模式，分阶段预测（试驾前+试驾后），完全符合业务规则

业务规则（来自《O/H/A/B定级业务规则》）：
- O 级：已订车、已成交
- 试驾前邀约阶段：
  - H 级：7天内试驾
  - A 级：14天内试驾
  - B 级：21天内试驾
- 试驾后下订商谈阶段：
  - H 级：7天内下订
  - A 级：14天内下订
  - B 级：21天内下订
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


@dataclass
class OHABResult:
    """OHAB 评级结果"""
    ratings: np.ndarray                    # OHAB 评级数组
    stage: np.ndarray                      # 阶段数组（"O", "试驾前", "试驾后"）
    proba_7d: Optional[np.ndarray] = None  # 7天概率
    proba_14d: Optional[np.ndarray] = None  # 14天概率
    proba_21d: Optional[np.ndarray] = None  # 21天概率
    order_proba_7d: Optional[np.ndarray] = None   # 7天下订概率
    order_proba_14d: Optional[np.ndarray] = None  # 14天下订概率
    order_proba_21d: Optional[np.ndarray] = None  # 21天下订概率
    distribution: Dict[str, int] = field(default_factory=dict)


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


def derive_ohab_simple(
    y_proba: np.ndarray,
    is_ordered: np.ndarray,
    threshold: float = 0.5,
) -> OHABResult:
    """
    简单模式 OHAB 推导

    使用单个模型（14天试驾概率）推断 OHAB 评级。

    推导逻辑：
    - O 级：已成交状态（优先级最高）
    - H 级：P(14天试驾) >= threshold（高意向）
    - A 级：threshold * 0.7 <= P < threshold（中意向）
    - B 级：threshold * 0.3 <= P < threshold * 0.7（低意向）
    - N 级：P < threshold * 0.3（无意向）

    Args:
        y_proba: 预测概率数组（14 天试驾概率）
        is_ordered: 已成交状态数组
        threshold: 评级判定阈值

    Returns:
        OHABResult 评级结果
    """
    n_samples = len(y_proba)
    ratings = np.full(n_samples, "N", dtype=object)
    stage = np.full(n_samples, "试驾前", dtype=object)

    # 1. O 级：已成交
    ratings[is_ordered] = "O"
    stage[is_ordered] = "O"

    # 2. 非成交样本：根据概率判断 H/A/B/N
    not_ordered = ~is_ordered

    # H 级
    h_mask = not_ordered & (y_proba >= threshold)
    ratings[h_mask] = "H"

    # A 级
    a_mask = not_ordered & (y_proba >= threshold * 0.7) & (y_proba < threshold)
    ratings[a_mask] = "A"

    # B 级
    b_mask = not_ordered & (y_proba >= threshold * 0.3) & (y_proba < threshold * 0.7)
    ratings[b_mask] = "B"

    # N 级已默认

    # 统计分布
    distribution = {r: int((ratings == r).sum()) for r in ["O", "H", "A", "B", "N"]}

    return OHABResult(
        ratings=ratings,
        stage=stage,
        proba_14d=y_proba,
        distribution=distribution,
    )


def derive_ohab_medium(
    proba_7d: np.ndarray,
    proba_14d: np.ndarray,
    proba_21d: np.ndarray,
    is_ordered: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> OHABResult:
    """
    中等模式 OHAB 推导

    使用三模型集成（7/14/21天试驾概率）推断 OHAB 评级。
    完全符合试驾前邀约阶段的业务规则。

    业务规则：
    - H 级：计划 7 天内试驾（P(7天) >= threshold）
    - A 级：计划 14 天内试驾（P(14天) >= threshold 且非 H）
    - B 级：计划 21 天内试驾（P(21天) >= threshold 且非 H/A）
    - N 级：无计划

    Args:
        proba_7d: 7天试驾概率数组
        proba_14d: 14天试驾概率数组
        proba_21d: 21天试驾概率数组
        is_ordered: 已成交状态数组
        thresholds: 各级别阈值，默认 {"H": 0.5, "A": 0.5, "B": 0.5}

    Returns:
        OHABResult 评级结果
    """
    thresholds = thresholds or {"H": 0.5, "A": 0.5, "B": 0.5}
    n_samples = len(proba_7d)
    ratings = np.full(n_samples, "N", dtype=object)
    stage = np.full(n_samples, "试驾前", dtype=object)

    # 1. O 级：已成交
    ratings[is_ordered] = "O"
    stage[is_ordered] = "O"

    # 2. 非成交样本：按业务规则推导
    not_ordered = ~is_ordered

    # H 级：7天内试驾概率 >= threshold
    h_mask = not_ordered & (proba_7d >= thresholds["H"])
    ratings[h_mask] = "H"

    # A 级：非 H，但 14天内试驾概率 >= threshold
    remaining = not_ordered & (ratings == "N")
    a_mask = remaining & (proba_14d >= thresholds["A"])
    ratings[a_mask] = "A"

    # B 级：非 H/A，但 21天内试驾概率 >= threshold
    remaining = not_ordered & (ratings == "N")
    b_mask = remaining & (proba_21d >= thresholds["B"])
    ratings[b_mask] = "B"

    # N 级已默认

    # 统计分布
    distribution = {r: int((ratings == r).sum()) for r in ["O", "H", "A", "B", "N"]}

    return OHABResult(
        ratings=ratings,
        stage=stage,
        proba_7d=proba_7d,
        proba_14d=proba_14d,
        proba_21d=proba_21d,
        distribution=distribution,
    )


def derive_ohab_advanced(
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
    # 阈值
    thresholds: Optional[Dict[str, float]] = None,
) -> OHABResult:
    """
    高等模式 OHAB 推导

    分阶段预测，完全符合业务规则：
    - 已成交 → O 级
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
        thresholds: 各级别阈值

    Returns:
        OHABResult 评级结果
    """
    thresholds = thresholds or {"H": 0.5, "A": 0.5, "B": 0.5}
    n_samples = len(drive_proba_7d)
    ratings = np.full(n_samples, "N", dtype=object)
    stage = np.full(n_samples, "试驾前", dtype=object)

    # 1. O 级：已成交（最高优先级）
    ratings[is_ordered] = "O"
    stage[is_ordered] = "O"

    # 2. 已试驾未成交 → 试驾后下订商谈阶段
    driven_not_ordered = is_driven & ~is_ordered
    stage[driven_not_ordered] = "试驾后"

    # 使用下订概率推导
    # H 级：7天内下订概率 >= threshold
    h_mask = driven_not_ordered & (order_proba_7d >= thresholds["H"])
    ratings[h_mask] = "H"

    # A 级：非 H，但 14天内下订概率 >= threshold
    remaining = driven_not_ordered & (ratings == "N")
    a_mask = remaining & (order_proba_14d >= thresholds["A"])
    ratings[a_mask] = "A"

    # B 级：非 H/A，但 21天内下订概率 >= threshold
    remaining = driven_not_ordered & (ratings == "N")
    b_mask = remaining & (order_proba_21d >= thresholds["B"])
    ratings[b_mask] = "B"

    # 3. 未试驾未成交 → 试驾前邀约阶段
    not_driven_not_ordered = ~is_driven & ~is_ordered

    # 使用试驾概率推导
    # H 级：7天内试驾概率 >= threshold
    h_mask = not_driven_not_ordered & (drive_proba_7d >= thresholds["H"])
    ratings[h_mask] = "H"

    # A 级：非 H，但 14天内试驾概率 >= threshold
    remaining = not_driven_not_ordered & (ratings == "N")
    a_mask = remaining & (drive_proba_14d >= thresholds["A"])
    ratings[a_mask] = "A"

    # B 级：非 H/A，但 21天内试驾概率 >= threshold
    remaining = not_driven_not_ordered & (ratings == "N")
    b_mask = remaining & (drive_proba_21d >= thresholds["B"])
    ratings[b_mask] = "B"

    # N 级已默认

    # 统计分布
    distribution = {r: int((ratings == r).sum()) for r in ["O", "H", "A", "B", "N"]}

    return OHABResult(
        ratings=ratings,
        stage=stage,
        proba_7d=drive_proba_7d,
        proba_14d=drive_proba_14d,
        proba_21d=drive_proba_21d,
        order_proba_7d=order_proba_7d,
        order_proba_14d=order_proba_14d,
        order_proba_21d=order_proba_21d,
        distribution=distribution,
    )


class OHABRater:
    """
    OHAB 评级器

    支持三种预测模式的 OHAB 评级推导。
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
    ) -> OHABResult:
        """
        推导 OHAB 评级

        Args:
            df: 输入数据 DataFrame（用于检测状态）
            proba_14d: 简单模式的 14天试驾概率
            drive_proba_7d/14d/21d: 试驾概率（中等/高等模式）
            order_proba_7d/14d/21d: 下订概率（高等模式）

        Returns:
            OHABResult 评级结果
        """
        # 检测已成交状态
        is_ordered = detect_ordered_status(df)

        if self.mode == PredictionMode.SIMPLE:
            if proba_14d is None:
                raise ValueError("简单模式需要提供 proba_14d 参数")
            return derive_ohab_simple(
                y_proba=proba_14d,
                is_ordered=is_ordered,
                threshold=self.thresholds.get("H", 0.5),
            )

        elif self.mode == PredictionMode.MEDIUM:
            if any(p is None for p in [drive_proba_7d, drive_proba_14d, drive_proba_21d]):
                raise ValueError("中等模式需要提供 drive_proba_7d/14d/21d 参数")
            return derive_ohab_medium(
                proba_7d=drive_proba_7d,
                proba_14d=drive_proba_14d,
                proba_21d=drive_proba_21d,
                is_ordered=is_ordered,
                thresholds=self.thresholds,
            )

        elif self.mode == PredictionMode.ADVANCED:
            if any(p is None for p in [drive_proba_7d, drive_proba_14d, drive_proba_21d,
                                        order_proba_7d, order_proba_14d, order_proba_21d]):
                raise ValueError("高等模式需要提供所有概率参数")
            # 检测已试驾状态
            is_driven = detect_driven_status(df)
            return derive_ohab_advanced(
                drive_proba_7d=drive_proba_7d,
                drive_proba_14d=drive_proba_14d,
                drive_proba_21d=drive_proba_21d,
                order_proba_7d=order_proba_7d,
                order_proba_14d=order_proba_14d,
                order_proba_21d=order_proba_21d,
                is_ordered=is_ordered,
                is_driven=is_driven,
                thresholds=self.thresholds,
            )

        else:
            raise ValueError(f"未知的预测模式: {self.mode}")

    def add_to_dataframe(
        self,
        df: pd.DataFrame,
        result: OHABResult,
        include_proba: bool = True,
    ) -> pd.DataFrame:
        """
        将评级结果添加到 DataFrame

        Args:
            df: 原始 DataFrame
            result: OHABResult 评级结果
            include_proba: 是否包含概率列

        Returns:
            添加评级列后的 DataFrame
        """
        df = df.copy()
        df["OHAB评级"] = result.ratings
        df["评级阶段"] = result.stage

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