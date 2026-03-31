"""
字段名规范化映射

将原始导出字段名中的 PostgreSQL 不合法字符替换为下划线：
- 中文括号 `（）` → `_`
- 英文括号 `()` → `_` 或省略
- 斜杠 `/` → `_`

原则：
1. 保持语义清晰
2. 下划线连接，避免歧义
3. 与 adapter.py 中 SCHEMA_ALIAS_MAPPING 保持一致

用法：
    from config.column_mapping import normalize_column_names, normalize_column_name

    # 单个字段名
    normalized = normalize_column_name("线索评级结果（最新）")
    # 返回: "线索评级结果_最新"

    # 批量处理
    rename_map = normalize_column_names(df.columns)
    # 返回: {"线索评级结果（最新）": "线索评级结果_最新", ...}
"""

import re
from typing import Dict, List


# PostgreSQL 不合法字符字段名映射（精确匹配）
COLUMN_NAME_MAPPING: Dict[str, str] = {
    # === 中文括号 → 下划线 ===
    "线索评级结果（最新）": "线索评级结果_最新",
    "行业标签分析结果（JSON）": "行业标签分析结果_JSON",
    "手机号（脱敏）": "手机号_脱敏",
    "客户是否主动询问购车权益（优惠）": "客户是否主动询问购车权益",

    # === 英文括号 → 下划线/省略 ===
    "客户ID(店端)": "客户ID_店端",
    "预算区间(购车预算)": "预算区间",

    # === 斜杠 → 省略 ===
    "首触意向车型/意向车型": "首触意向车型",
}


def normalize_column_name(name: str) -> str:
    """
    规范化单个字段名

    规则：
    1. 查映射表（精确匹配）
    2. 替换不合法字符：（）() / → _
    3. 清理连续下划线

    Args:
        name: 原始字段名

    Returns:
        规范化后的字段名
    """
    # 1. 精确匹配
    if name in COLUMN_NAME_MAPPING:
        return COLUMN_NAME_MAPPING[name]

    # 2. 通用替换：不合法字符 → 下划线
    normalized = name
    normalized = normalized.replace("（", "_").replace("）", "_")
    normalized = normalized.replace("(", "_").replace(")", "_")
    normalized = normalized.replace("/", "_")

    # 3. 清理连续下划线
    normalized = re.sub(r"_+", "_", normalized)

    # 4. 清理首尾下划线
    normalized = normalized.strip("_")

    return normalized


def normalize_column_names(columns: List[str]) -> Dict[str, str]:
    """
    批量规范化字段名，仅返回需要修改的字段

    Args:
        columns: 原始字段名列表

    Returns:
        {原始字段名: 规范化字段名} 映射字典（仅包含需要修改的字段）
    """
    return {
        col: normalize_column_name(col)
        for col in columns
        if col != normalize_column_name(col)
    }


def get_column_mapping() -> Dict[str, str]:
    """
    获取完整的字段名映射表

    Returns:
        字段名映射字典
    """
    return COLUMN_NAME_MAPPING.copy()