"""
业务逻辑语义层。

职责：
1. 维护技术特征到业务维度的唯一映射。
2. 提供 HAB 等级对应的 SOP 文案。
3. 将单线索特征翻译为可读的原因短语。

本模块不负责：
- 读写文件
- 生成整篇客户报告
- 伪造 SHAP 或未来窗口概率
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


BUSINESS_DIMENSION_MAP = {
    "基础特征": [
        "一级渠道名称", "二级渠道名称", "三级渠道名称", "四级渠道名称",
        "所在城市", "首触意向车型", "预算区间", "线索类型",
        "渠道组合", "城市车型热度",
    ],
    "画像特征": [
        "客户性别", "历史订单次数", "历史到店次数", "历史试驾次数",
        "历史订单次数_备用", "历史到店次数_备用",
    ],
    "行为特征": [
        "通话次数", "通话总时长", "平均通话时长", "通话时长是否大于等于45秒",
        "平均通话时长_派生", "是否接通", "客户是否同意加微信", "意向金支付状态", "意向金支付状态_备用",
        "跟进总次数", "接通次数", "未接通次数", "接通率", "继续跟进次数",
        "平均通话时长_秒", "最长通话时长_秒", "有效通话", "AI语义标签可用", "语义标签命中数",
        "JSON跟进明细可用",
    ],
    "时序特征": [
        "线索创建星期几", "线索创建小时", "线索创建时间_day_of_week", "线索创建时间_hour", "线索创建时间_is_weekend",
        "分配时间_day_of_week", "分配时间_hour", "分配时间_is_weekend",
        "线索下发时间_day_of_week", "线索下发时间_hour", "线索下发时间_is_weekend",
        "首触时间_day_of_week", "首触时间_hour", "首触时间_is_weekend",
        "首触线索是否及时外呼", "首触线索当天是否联通实体卡外呼", "首触响应时长_小时", "响应及时性",
    ],
    "意图特征": [
        "客户是否主动询问交车时间", "客户是否主动询问购车权益", "客户是否主动询问金融政策",
        "客户是否表示门店距离太远拒绝到店", "SOP开口标签", "SOP开口标签_备用",
        "提及价格", "提及试驾", "提及到店", "提及竞品", "意向级别下降", "最终战败",
    ],
}

HAB_SOP_MAP = {
    "H": "24小时内优先跟进，优先邀约试驾/到店，明确车型、权益和到店时间。",
    "A": "48小时内完成跟进，重点沟通优惠权益、金融方案和顾虑消除。",
    "B": "进入培育节奏，按周触达，优先用内容运营或自动化私域继续激活。",
}

LABEL_DISPLAY_NAME = {
    "H": "高意向",
    "A": "中高意向",
    "B": "待培育",
}


def _is_truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip()
        return normalized in {"1", "true", "True", "是", "有", "已支付", "支付成功", "已转定金"}
    if isinstance(value, (int, float)):
        return float(value) > 0
    return bool(value)


def _to_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_feature_business_dimension(feature_name: str) -> str:
    """获取特征所属业务维度。"""
    for dimension, features in BUSINESS_DIMENSION_MAP.items():
        if feature_name in features:
            return dimension
        for feature in features:
            if feature_name.startswith(feature) or feature in feature_name:
                return dimension
    return "其他特征"


def calculate_dimension_contribution(feature_importance: Dict[str, float]) -> Dict[str, float]:
    """计算业务维度贡献。"""
    dimension_scores = {dimension: 0.0 for dimension in BUSINESS_DIMENSION_MAP}
    dimension_scores["其他特征"] = 0.0
    for feature, importance in feature_importance.items():
        dimension = get_feature_business_dimension(feature)
        dimension_scores[dimension] += importance
    return dimension_scores


def get_business_logic_explanation(top_features: List[Tuple[str, float]]) -> str:
    """根据重要特征生成简洁说明。"""
    explanations = []
    for feature, importance in top_features:
        dimension = get_feature_business_dimension(feature)
        explanations.append(f"- 【{dimension}】{feature}: 影响力 {importance:.4f}")
    return "\n".join(explanations)


def get_sop_for_label(label: str) -> str:
    """返回标签对应的 SOP。"""
    return HAB_SOP_MAP.get(label, "按标准销售节奏跟进。")


def _reason_candidates_from_row(row: pd.Series) -> List[str]:
    candidates: List[str] = []

    if _is_truthy(row.get("提及试驾")):
        candidates.append("客户在跟进中主动提及试驾，处于较强体验决策阶段。")
    if _is_truthy(row.get("提及到店")):
        candidates.append("客户已表现出到店意向，具备线下转化基础。")
    if _is_truthy(row.get("提及价格")):
        candidates.append("客户开始关注价格和权益，已进入比价决策阶段。")
    if _is_truthy(row.get("客户是否主动询问金融政策")):
        candidates.append("客户主动询问金融政策，说明在评估支付方案。")
    if _is_truthy(row.get("客户是否主动询问交车时间")):
        candidates.append("客户主动关注交车时间，存在较明确的购车时间预期。")
    if _is_truthy(row.get("客户是否同意加微信")):
        candidates.append("客户愿意添加微信，便于持续私域跟进。")
    if _is_truthy(row.get("有效通话")):
        candidates.append("存在有效通话，销售与客户已发生实质沟通。")
    if _to_float(row.get("接通率")) >= 0.5:
        candidates.append("历史接通率较高，客户触达效率较好。")
    if 0 < _to_float(row.get("首触响应时长_小时")) <= 1:
        candidates.append("首触响应较快，销售及时介入了窗口期。")
    if _to_float(row.get("历史到店次数")) > 0:
        candidates.append("客户历史上有到店记录，对品牌已有实际接触。")
    if _to_float(row.get("历史试驾次数")) > 0:
        candidates.append("客户历史上已有试驾行为，成交基础相对更好。")

    if _is_truthy(row.get("客户是否表示门店距离太远拒绝到店")):
        candidates.append("客户表达过门店距离顾虑，到店转化阻力较大。")
    if _is_truthy(row.get("最终战败")):
        candidates.append("历史跟进中出现战败信号，短期转化意愿偏弱。")
    if _is_truthy(row.get("意向级别下降")):
        candidates.append("跟进过程中意向级别出现下降，需谨慎投入高强度资源。")
    if _to_float(row.get("未接通次数")) > _to_float(row.get("接通次数")):
        candidates.append("未接通次数偏多，当前客户触达稳定性不足。")
    if _to_float(row.get("有效通话")) == 0 and _to_float(row.get("通话次数")) > 0:
        candidates.append("虽然发生过外呼，但有效沟通深度仍然不足。")

    return candidates


def build_reason_codes(
    row: pd.Series,
    predicted_label: str,
    max_reasons: int = 3,
) -> List[str]:
    """根据单线索特征生成原因短语。"""
    reasons = _reason_candidates_from_row(row)

    if predicted_label == "H":
        prioritized = [reason for reason in reasons if "主动" in reason or "有效通话" in reason or "到店" in reason or "试驾" in reason]
    elif predicted_label == "B":
        prioritized = [reason for reason in reasons if "阻力" in reason or "下降" in reason or "不足" in reason or "战败" in reason]
    else:
        prioritized = reasons

    ordered = prioritized + [reason for reason in reasons if reason not in prioritized]

    if not ordered:
        fallback = {
            "H": "近期互动质量和意向信号整体较强，建议优先转化。",
            "A": "客户有一定兴趣，但仍处于比较和考虑阶段。",
            "B": "当前显性高意向信号不足，适合进入培育节奏。",
        }
        return [fallback.get(predicted_label, "当前业务信号有限，建议按标准流程跟进。")]

    return ordered[:max_reasons]


def build_lead_action_record(
    row: pd.Series,
    predicted_label: str,
    probability_map: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """构造单条线索的业务下发记录。"""
    reasons = build_reason_codes(row, predicted_label)
    probability_map = probability_map or {}
    lead_id = row.get("线索唯一ID")
    if pd.isna(lead_id):
        lead_id = row.name

    return {
        "线索唯一ID": lead_id,
        "预测HAB": predicted_label,
        "等级说明": LABEL_DISPLAY_NAME.get(predicted_label, predicted_label),
        "建议SOP": get_sop_for_label(predicted_label),
        "原因1": reasons[0] if len(reasons) > 0 else "",
        "原因2": reasons[1] if len(reasons) > 1 else "",
        "原因3": reasons[2] if len(reasons) > 2 else "",
        "概率_H": float(probability_map.get("H", 0.0)),
        "概率_A": float(probability_map.get("A", 0.0)),
        "概率_B": float(probability_map.get("B", 0.0)),
    }


def summarize_top_dimensions(dimension_contribution: Dict[str, float], limit: int = 3) -> List[Tuple[str, float]]:
    """返回贡献最高的业务维度。"""
    return sorted(dimension_contribution.items(), key=lambda item: item[1], reverse=True)[:limit]


def build_bucket_summary_text(bucket_rows: Sequence[Dict[str, object]]) -> List[str]:
    """将桶摘要转成简洁业务描述。"""
    lines = []
    for row in bucket_rows:
        bucket = str(row.get("bucket", ""))
        sample_ratio = float(row.get("sample_ratio", 0.0))
        drive_rate = float(row.get("试驾标签_14天_rate", 0.0))
        arrive_rate = float(row.get("到店标签_14天_rate", 0.0))
        lines.append(
            f"{bucket} 桶占比 {sample_ratio:.1%}，14天到店率 {arrive_rate:.1%}，14天试驾率 {drive_rate:.1%}。"
        )
    return lines
