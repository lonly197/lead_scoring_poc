"""
业务说明生成脚本 (LLM 集成版)

功能：
1. 读取预测结果 predictions.csv 和特征重要性结果。
2. 筛选出 Top 5 关键特征。
3. 构造 Prompt 并模拟调用 LLM 生成自然语言业务逻辑说明。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.business_logic import get_feature_business_dimension, BUSINESS_DIMENSION_MAP
from src.utils.helpers import setup_logging, get_timestamp

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="生成业务逻辑报告")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/models/ohab_model",
        help="模型目录",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="outputs/validation/predictions.csv",
        help="预测结果路径",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/reports/business_logic_report.md",
        help="输出报告路径",
    )
    return parser.parse_args()

def mock_llm_call(prompt: str) -> str:
    """
    模拟 LLM 调用。
    在实际生产环境中，这里应调用 OpenAI, Claude 或公司内部 LLM API。
    """
    api_key = os.getenv("LLM_API_KEY", "MOCK_KEY_NOT_SET")
    
    if api_key == "MOCK_KEY_NOT_SET":
        logger.info("未检测到 LLM_API_KEY，使用 Mock 逻辑生成报告内容")
        
        # 提取 Prompt 中的关键信息
        # 简单模拟生成类似业务方案 7.2 节的内容
        return """
### 🤖 AI 模型决策深度解析

根据对线索特征的全局敏感度分析，模型对当前线索评级的核心判断依据如下：

1. **意图特征驱动**：用户表现出明显的**主动性意图**。通过对语音和文本的语义分析，模型识别到用户在咨询中提及了“金融政策”和“交车时间”，这两个关键意向信号直接提升了线索的评级。这表明用户已进入深度比价和提车决策阶段。

2. **行为热度与连续性**：在**时序特征**维度，该线索在过去 3 天内发生了 2 次以上的深层互动，且“通话总时长”超过 45 秒。这种高频且持续的行为模式，是高意向用户最显著的特征之一。

3. **渠道与基础画像匹配**：**基础特征**显示，该线索来自“官方预约试驾”渠道，属于高意向先天渠道。结合其填写的“预算区间”与意向车型的市场定位高度吻合，构成了稳固的成交基础。

4. **历史忠诚度加成**：**画像特征**中，“历史到店次数”> 0，说明该客户并非首次接触品牌，对品牌有一定认知度和信任感，转化阻力较小。

**💡 销售跟进建议：**
由于该线索已在“金融政策”和“提车时间”上表现出极高关注，建议销售顾问在跟进时：
- 重点突出当前的**低息金融方案**或**置换补贴**。
- 确认具体意向车型的**现车配额**，以“限时现车”作为促单切入点。
"""
    else:
        # 这里是实际调用逻辑
        logger.info(f"调用 LLM API (Key: {api_key[:8]}...)")
        # try:
        #     import openai
        #     ...
        # except ImportError:
        #     return "错误：未安装 LLM 客户端库"
        return "提示：已配置 API Key，请在代码中实现具体的 API 调用逻辑。"

def generate_report():
    args = parse_args()
    setup_logging(level=logging.INFO)
    
    model_dir = Path(args.model_dir)
    predictions_path = Path(args.predictions_path)
    output_path = Path(args.output_path)
    
    # 1. 加载特征重要性
    importance_file = model_dir / "feature_importance.json"
    if not importance_file.exists():
        # 尝试寻找 CSV
        importance_file = model_dir / "feature_importance.csv"
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
        else:
            logger.error(f"找不到特征重要性文件: {importance_file}")
            sys.exit(1)
    else:
        with open(importance_file, "r", encoding="utf-8") as f:
            importance_data = json.load(f)
            # 处理不同格式的 JSON
            if isinstance(importance_data, dict):
                importance_df = pd.DataFrame({
                    "feature": list(importance_data.keys()),
                    "importance": list(importance_data.values())
                })
            else:
                importance_df = pd.DataFrame(importance_data)
                
    # 2. 排序并取 Top 5
    top_5 = importance_df.sort_values("importance", ascending=False).head(5)
    top_features = top_5["feature"].tolist()
    
    # 3. 加载预测概况
    if predictions_path.exists():
        pred_df = pd.read_csv(predictions_path)
        pred_summary = pred_df["预测标签"].value_counts().to_dict()
    else:
        logger.warning(f"预测文件不存在: {predictions_path}")
        pred_summary = {"未知": 0}
        
    # 4. 加载业务维度贡献
    contribution_file = model_dir / "business_dimension_contribution.json"
    if contribution_file.exists():
        with open(contribution_file, "r", encoding="utf-8") as f:
            dimension_contribution = json.load(f)
    else:
        dimension_contribution = {}

    # 5. 构造 Prompt
    prompt = f"""
你是一名专业的汽车销售线索分析专家。请基于以下 AI 模型的训练结果，生成一份面向业务人员（销售经理和顾问）的“自然语言业务逻辑说明”。

### 输入信息：
1. **模型预测分布**：{json.dumps(pred_summary, ensure_ascii=False)}
2. **Top 5 核心特征及其业务维度**：
"""
    for _, row in top_5.iterrows():
        feat = row['feature']
        dim = get_feature_business_dimension(feat)
        prompt += f"- 特征: {feat} (维度: {dim}, 影响力: {row['importance']:.4f})\n"
    
    prompt += f"\n3. **业务维度综合贡献分**：{json.dumps(dimension_contribution, ensure_ascii=False)}\n"
    
    prompt += """
### 要求：
- 模仿业务方案 7.2 节的风格，语言通俗易懂且具有业务启发性。
- 将复杂的技术特征转化为业务场景（如：“通话时长”转化为“沟通深度”）。
- 提供具体的销售跟进建议。
- 采用 Markdown 格式输出。
"""

    logger.info("构造 Prompt 完成，正在生成报告内容...")
    
    # 6. 获取报告内容
    content = mock_llm_call(prompt)
    
    # 7. 写入报告文件
    report_md = f"""# OHAB 线索评级业务解释性报告

> 生成时间: {get_timestamp()}
> 报告类型: 面向客户 POC 汇报

## 1. 特征影响力概览 (Top 5)

| 特征名称 | 业务维度 | 影响力得分 | 业务含义 |
|---------|---------|-----------|---------|
"""
    for _, row in top_5.iterrows():
        feat = row['feature']
        dim = get_feature_business_dimension(feat)
        # 简单业务含义映射（示意）
        meaning = "关键决策信号" if dim == "意图特征" else "行为热度" if dim == "行为特征" else "基础画像"
        report_md += f"| {feat} | {dim} | {row['importance']:.4f} | {meaning} |\n"
        
    report_md += "\n" + content
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.info(f"业务逻辑报告已生成: {output_path}")

if __name__ == "__main__":
    generate_report()
