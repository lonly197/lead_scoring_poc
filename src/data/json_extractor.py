"""
JSON 字段特征提取模块

从跟进详情_JSON 提取关键业务特征：
- 跟进次数统计
- 意向级别变化轨迹
- 通话结果统计
- 跟进备忘关键词

作者: Lead Scoring Team
日期: 2026-03-25
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_followup_features(json_str: str) -> Dict:
    """
    从跟进详情_JSON 提取关键特征

    Args:
        json_str: JSON 字符串，格式为数组 [{"意向级别": "H", "通话结果": "已接通", ...}, ...]

    Returns:
        特征字典，包含以下字段：
        - 跟进总次数: int
        - 接通次数: int
        - 未接通次数: int
        - 接通率: float
        - 继续跟进次数: int
        - 最终战败: bool
        - 意向级别下降: bool
        - 提及价格: bool
        - 提及试驾: bool
        - 提及到店: bool
        - 提及竞品: bool
        - 平均通话时长_秒: float
        - 最长通话时长_秒: int
    """
    # 默认值（解析失败时返回）
    default_features = {
        '跟进总次数': 0,
        '接通次数': 0,
        '未接通次数': 0,
        '接通率': 0.0,
        '继续跟进次数': 0,
        '最终战败': False,
        '意向级别下降': False,
        '提及价格': False,
        '提及试驾': False,
        '提及到店': False,
        '提及竞品': False,
        '平均通话时长_秒': 0.0,
        '最长通话时长_秒': 0,
    }

    # 空值检查
    if pd.isna(json_str) or not json_str:
        return default_features.copy()

    try:
        records = json.loads(json_str)

        # 处理单条记录（非数组）的情况
        if not isinstance(records, list):
            records = [records] if records else []

        if not records:
            return default_features.copy()

        features = {}
        features['跟进总次数'] = len(records)

        # 1. 意向级别变化轨迹
        intn_levels = [r.get('意向级别') for r in records if r.get('意向级别')]
        if len(intn_levels) >= 2:
            # H → A/B 或 A → B 视为下降
            # 级别顺序: O(最高) > H > A > B
            level_order = {'O': 4, 'H': 3, 'A': 2, 'B': 1}
            first_level = intn_levels[0]
            last_level = intn_levels[-1]
            if first_level in level_order and last_level in level_order:
                features['意向级别下降'] = level_order[last_level] < level_order[first_level]

        # 2. 通话结果统计
        call_results = [r.get('通话结果') for r in records if r.get('通话结果')]
        features['接通次数'] = sum(1 for c in call_results if c == '已接通')
        features['未接通次数'] = sum(1 for c in call_results if c == '未接通')
        if call_results:
            features['接通率'] = features['接通次数'] / len(call_results)

        # 3. 跟进结果统计
        follow_results = [r.get('跟进结果') for r in records if r.get('跟进结果')]
        features['继续跟进次数'] = sum(1 for f in follow_results if f == '继续')
        features['最终战败'] = '战败' in follow_results

        # 4. 通话时长统计（解析 "分:秒" 格式）
        call_durations = []
        for r in records:
            duration_str = r.get('通话时长(分:秒)', '')
            if not duration_str:
                continue
            try:
                parts = duration_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    call_durations.append(minutes * 60 + seconds)
            except (ValueError, AttributeError):
                pass

        if call_durations:
            features['平均通话时长_秒'] = round(np.mean(call_durations), 2)
            features['最长通话时长_秒'] = max(call_durations)

        # 5. 备忘关键词提取（高价值信号）
        memos = ' '.join([r.get('跟进备忘', '') for r in records if r.get('跟进备忘')])

        # 价格相关关键词
        price_keywords = ['价格', '优惠', '便宜', '多少钱', '打折', '降价', '折扣', '活动']
        features['提及价格'] = any(kw in memos for kw in price_keywords)

        # 试驾相关
        features['提及试驾'] = '试驾' in memos

        # 到店相关
        store_keywords = ['到店', '来店', '进店', '过去看', '去看车', '到展厅']
        features['提及到店'] = any(kw in memos for kw in store_keywords)

        # 竞品相关
        competitor_keywords = ['比亚迪', '特斯拉', '小鹏', '蔚来', '理想', '问界', '极氪', '小米汽车']
        features['提及竞品'] = any(kw in memos for kw in competitor_keywords)

        # 填充未提取到的特征为默认值
        result = default_features.copy()
        result.update(features)
        return result

    except json.JSONDecodeError as e:
        logger.debug(f"JSON 解析失败: {str(json_str)[:50]}... 错误: {e}")
    except Exception as e:
        logger.debug(f"特征提取失败: {e}")

    return default_features.copy()


def batch_extract_json_features(
    df: pd.DataFrame,
    json_column: str = '跟进详情_JSON',
    drop_original: bool = False
) -> pd.DataFrame:
    """
    批量提取 JSON 特征

    Args:
        df: 数据框
        json_column: JSON 列名，默认 '跟进详情_JSON'
        drop_original: 是否删除原始 JSON 列，默认 False

    Returns:
        添加了特征的数据框

    Example:
        >>> df = pd.DataFrame({'跟进详情_JSON': ['[{"意向级别": "H"}]']})
        >>> result = batch_extract_json_features(df)
        >>> '跟进总次数' in result.columns
        True
    """
    if json_column not in df.columns:
        logger.warning(f"JSON 列 '{json_column}' 不存在，跳过特征提取")
        return df

    logger.info(f"从 '{json_column}' 提取特征，共 {len(df)} 条记录...")

    # 批量提取特征
    features_list = df[json_column].apply(extract_followup_features)
    features_df = pd.DataFrame(features_list.tolist(), index=df.index)

    # 合并到原数据框
    result_df = pd.concat([df, features_df], axis=1)

    logger.info(f"提取了 {len(features_df.columns)} 个新特征: {list(features_df.columns)}")

    # 统计非零特征比例
    for col in features_df.columns:
        non_zero = (features_df[col] != 0).sum() if features_df[col].dtype in [np.int64, np.float64] else features_df[col].sum()
        logger.debug(f"  {col}: {non_zero}/{len(df)} ({non_zero/len(df)*100:.1f}%)")

    if drop_original:
        result_df = result_df.drop(columns=[json_column])
        logger.info(f"已删除原始列 '{json_column}'")

    return result_df