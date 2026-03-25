"""
线索评级可解释性输出模块

将模型预测结果转换为业务可理解的OHAB评级报告：
1. 概率 → OHAB等级映射
2. SHAP特征归因分析
3. 单线索报告生成
4. 群体统计报告生成

设计原则：
- 预测层：预测真实用户行为（试驾/到店概率）
- 映射层：概率阈值 → OHAB等级
- 解释层：SHAP归因回答"为什么"
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OHABResult:
    """OHAB评级结果"""
    level: str  # H/A/B/C
    prob_7day: float  # 7天内试驾概率
    prob_14day: float  # 14天内试驾概率
    prob_21day: float  # 21天内试驾概率
    confidence: float  # 预测置信度
    feature_contributions: Dict[str, float]  # 特征贡献值 (SHAP)
    top_positive_features: List[Tuple[str, float]]  # 正向特征TOP3
    top_negative_features: List[Tuple[str, float]]  # 负向特征TOP3
    action_suggestion: str  # 跟进建议
    risk_alert: Optional[str]  # 风险提示


def probability_to_ohab(
    prob_7day: float,
    prob_14day: float,
    prob_21day: float,
    thresholds: Optional[Dict] = None
) -> Tuple[str, float]:
    """
    将预测概率映射为OHAB等级

    映射逻辑（基于特征体系文档的评分阈值）：
    - H级（75-89分对应）：7天概率≥60% 或 14天概率≥75%
    - A级（60-74分对应）：14天概率≥50% 或 21天概率≥70%
    - B级（45-59分对应）：21天概率≥40%
    - C级（<45分对应）：各时间窗口概率都低

    Args:
        prob_7day: 7天内行为概率
        prob_14day: 14天内行为概率
        prob_21day: 21天内行为概率
        thresholds: 自定义阈值（可选）

    Returns:
        (OHAB等级, 预测置信度)
    """
    # 默认阈值（可根据业务调优）
    default_thresholds = {
        "H_7day": 0.60,      # 7天概率≥60% → H
        "H_14day": 0.75,     # 14天概率≥75% → H
        "A_14day": 0.50,     # 14天概率≥50% → A
        "A_21day": 0.70,     # 21天概率≥70% → A
        "B_21day": 0.40,     # 21天概率≥40% → B
    }

    thresh = thresholds or default_thresholds

    # H级判断
    if prob_7day >= thresh["H_7day"]:
        return "H", prob_7day
    if prob_14day >= thresh["H_14day"]:
        return "H", prob_14day

    # A级判断
    if prob_14day >= thresh["A_14day"]:
        return "A", prob_14day
    if prob_21day >= thresh["A_21day"]:
        return "A", prob_21day

    # B级判断
    if prob_21day >= thresh["B_21day"]:
        return "B", prob_21day

    # C级（培育池）
    return "C", prob_21day


def get_action_suggestion(level: str, top_features: List[Tuple[str, float]]) -> str:
    """
    根据等级和关键特征生成跟进建议

    Args:
        level: OHAB等级
        top_features: TOP正向特征 [(特征名, 贡献值), ...]

    Returns:
        跟进建议文本
    """
    suggestions = {
        "H": "优先级★★★★★，24小时内跟进。确认试驾/到店时间，准备金融方案。",
        "A": "优先级★★★★，48小时内跟进。持续培育，强化核心卖点。",
        "B": "优先级★★★，本周内跟进。发送车型资料，激活意向。",
        "C": "优先级★★，纳入培育池。定期推送内容，等待激活信号。"
    }

    base_suggestion = suggestions.get(level, "建议跟进")

    # 根据关键特征补充建议
    feature_tips = []
    feature_names = [f[0] for f in top_features]

    if "提及价格" in feature_names or "提及优惠" in feature_names:
        feature_tips.append("准备价格对比话术")
    if "提及试驾" in feature_names:
        feature_tips.append("确认试驾时间安排")
    if "提及到店" in feature_names:
        feature_tips.append("发送门店位置指引")
    if "提及竞品" in feature_names:
        feature_tips.append("准备竞品对比材料")
    if "首触响应时长_小时" in feature_names:
        feature_tips.append("客户响应积极，趁热打铁")

    if feature_tips:
        return f"{base_suggestion} 重点关注：{'、'.join(feature_tips[:3])}。"
    return base_suggestion


def get_risk_alert(feature_contributions: Dict[str, float]) -> Optional[str]:
    """
    识别风险因素

    Args:
        feature_contributions: 特征贡献值字典

    Returns:
        风险提示文本，无风险返回None
    """
    risks = []

    # 负向贡献特征
    if feature_contributions.get("距上次跟进天数", 0) < -0.05:
        risks.append("距上次跟进时间较长，存在流失风险")
    if feature_contributions.get("意向级别下降", 0) < -0.03:
        risks.append("意向级别呈下降趋势")
    if feature_contributions.get("最终战败", 0) < -0.1:
        risks.append("存在战败记录")

    if risks:
        return "⚠️ " + "；".join(risks)
    return None


def generate_single_report(
    lead_id: str,
    result: OHABResult,
    feature_descriptions: Optional[Dict[str, str]] = None
) -> str:
    """
    生成单线索评级报告（Markdown格式）

    Args:
        lead_id: 线索ID
        result: OHAB评级结果
        feature_descriptions: 特征业务描述映射

    Returns:
        Markdown格式报告
    """
    # 特征业务描述
    default_desc = {
        "首触响应时长_小时": "首次响应速度",
        "通话总时长": "沟通深度",
        "提及价格": "价格敏感度",
        "提及试驾": "试驾意向",
        "提及到店": "到店意向",
        "提及竞品": "竞品对比",
        "渠道组合": "渠道质量",
        "跟进总次数": "跟进频次",
        "接通率": "接通情况",
        "城市车型热度": "热门程度",
    }
    desc = feature_descriptions or default_desc

    # 构建报告
    lines = []
    lines.append(f"## 线索评级报告")
    lines.append("")
    lines.append(f"**线索ID**: {lead_id}")
    lines.append(f"**评级结果**: {result.level}级 ({_get_level_desc(result.level)})")
    lines.append(f"**预测置信度**: {result.confidence:.1%}")
    lines.append("")

    # 预测概率
    lines.append("### 预测概率")
    lines.append("")
    lines.append(f"| 时间窗口 | 试驾概率 |")
    lines.append("|---------|---------|")
    lines.append(f"| 7天内   | {result.prob_7day:.1%} |")
    lines.append(f"| 14天内  | {result.prob_14day:.1%} |")
    lines.append(f"| 21天内  | {result.prob_21day:.1%} |")
    lines.append("")

    # 特征贡献
    lines.append("### 评级依据")
    lines.append("")
    lines.append("| 特征 | 贡献值 | 方向 | 业务含义 |")
    lines.append("|------|--------|------|----------|")

    # 正向特征
    for feat, val in result.top_positive_features[:5]:
        direction = "↑ 提升" if val > 0 else "— 中性"
        meaning = desc.get(feat, feat)
        lines.append(f"| {feat} | +{val:.3f} | {direction} | {meaning} |")

    # 负向特征
    for feat, val in result.top_negative_features[:3]:
        direction = "↓ 降低"
        meaning = desc.get(feat, feat)
        lines.append(f"| {feat} | {val:.3f} | {direction} | {meaning} |")

    lines.append("")

    # 跟进建议
    lines.append("### 跟进建议")
    lines.append("")
    lines.append(f"✅ {result.action_suggestion}")
    lines.append("")

    # 风险提示
    if result.risk_alert:
        lines.append("### 风险提示")
        lines.append("")
        lines.append(result.risk_alert)
        lines.append("")

    return "\n".join(lines)


def _get_level_desc(level: str) -> str:
    """获取等级中文描述"""
    desc_map = {
        "H": "高意向 - 7天内试驾/下订",
        "A": "中高意向 - 14天内试驾/下订",
        "B": "中低意向 - 21天内试驾/下订",
        "C": "低意向 - 需培育",
    }
    return desc_map.get(level, "未知")


def generate_batch_report(
    results: List[OHABResult],
    lead_ids: List[str],
    output_dir: str
) -> Dict:
    """
    批量生成报告并输出统计

    Args:
        results: OHAB评级结果列表
        lead_ids: 线索ID列表
        output_dir: 输出目录

    Returns:
        统计摘要字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 统计
    level_counts = {"H": 0, "A": 0, "B": 0, "C": 0}
    prob_stats = {
        "prob_7day": [],
        "prob_14day": [],
        "prob_21day": []
    }

    # 生成单线索报告
    reports_dir = output_path / "individual_reports"
    reports_dir.mkdir(exist_ok=True)

    for lead_id, result in zip(lead_ids, results):
        level_counts[result.level] += 1
        prob_stats["prob_7day"].append(result.prob_7day)
        prob_stats["prob_14day"].append(result.prob_14day)
        prob_stats["prob_21day"].append(result.prob_21day)

        # 保存单线索报告
        report = generate_single_report(lead_id, result)
        report_path = reports_dir / f"{lead_id}.md"
        report_path.write_text(report, encoding="utf-8")

    # 生成汇总报告
    summary = {
        "total_leads": len(results),
        "level_distribution": level_counts,
        "level_percentage": {k: v/len(results)*100 for k, v in level_counts.items()},
        "probability_statistics": {
            k: {
                "mean": np.mean(v),
                "median": np.median(v),
                "std": np.std(v)
            }
            for k, v in prob_stats.items()
        }
    }

    # 保存汇总报告
    summary_path = output_path / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"批量报告已生成: {output_path}")
    logger.info(f"等级分布: H={level_counts['H']}, A={level_counts['A']}, B={level_counts['B']}, C={level_counts['C']}")

    return summary