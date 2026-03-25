"""
业务逻辑处理模块

负责将技术特征映射到业务维度，并提供业务解释性相关的工具函数。
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# 定义业务维度映射
# 参考《线索OHAB智能评级特征体系_完整版.md》
BUSINESS_DIMENSION_MAP = {
    "基础特征": [
        "一级渠道名称", "二级渠道名称", "三级渠道名称", "四级渠道名称",
        "所在城市", "首触意向车型", "预算区间", "线索类型",
        "渠道组合", "城市车型热度",
    ],
    "画像特征": [
        "客户性别", "历史订单次数", "历史到店次数", "历史试驾次数",
        "历史订单次数_备用", "历史到店次数_备用"
    ],
    "行为特征": [
        "通话次数", "通话总时长", "平均通话时长", "通话时长是否大于等于45秒",
        "平均通话时长_派生", "是否接通", "客户是否同意加微信", "意向金支付状态", "意向金支付状态_备用",
        "跟进总次数", "接通次数", "未接通次数", "接通率", "继续跟进次数",
        "平均通话时长_秒", "最长通话时长_秒", "有效通话",
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
    ]
}

def get_feature_business_dimension(feature_name: str) -> str:
    """
    获取特征所属的业务维度

    Args:
        feature_name: 特征名称

    Returns:
        业务维度名称，如果未匹配则返回 "其他特征"
    """
    for dimension, features in BUSINESS_DIMENSION_MAP.items():
        # 精确匹配或前缀匹配（处理衍生特征）
        if feature_name in features:
            return dimension
        
        # 处理 AutoGluon 可能添加的前缀或后缀，或者时间衍生特征
        for f in features:
            if feature_name.startswith(f) or f in feature_name:
                return dimension
                
    return "其他特征"

def calculate_dimension_contribution(feature_importance: Dict[str, float]) -> Dict[str, float]:
    """
    计算业务维度的综合贡献分（基于特征重要性之和）

    Args:
        feature_importance: 特征重要性字典 {feature_name: importance_value}

    Returns:
        业务维度贡献分字典 {dimension_name: contribution_value}
    """
    dimension_scores = {dim: 0.0 for dim in BUSINESS_DIMENSION_MAP.keys()}
    dimension_scores["其他特征"] = 0.0
    
    total_importance = sum(feature_importance.values())
    if total_importance == 0:
        return dimension_scores
        
    for feature, importance in feature_importance.items():
        dimension = get_feature_business_dimension(feature)
        dimension_scores[dimension] += importance
        
    # 归一化为百分比（可选，此处保持原始和）
    return dimension_scores

def get_business_logic_explanation(top_features: List[Tuple[str, float]]) -> str:
    """
    根据 Top 特征生成简单的业务逻辑说明（本地规则版）
    注：更复杂的自然语言说明建议通过 scripts/generate_business_report.py 调用 LLM 生成
    """
    explanations = []
    for feature, importance in top_features:
        dimension = get_feature_business_dimension(feature)
        explanations.append(f"- 【{dimension}】{feature}: 影响力 {importance:.4f}")
    
    return "\n".join(explanations)
