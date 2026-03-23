"""
数据适配器 - 支持多种数据格式

解决新旧数据格式差异：
1. 202603.csv：Tab分隔，无表头，46列
2. 20260308-v2.csv：逗号分隔，有表头，60列

功能：
- 自动检测数据格式
- 列名映射
- 目标变量计算（从原始时间字段派生）
- 时间特征衍生
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class DataFormatConfig:
    """数据格式配置"""

    # 分隔符
    sep: str = ","

    # 是否有表头
    header: Optional[int] = 0  # 0=第一行, None=无表头

    # 列名映射（当无表头时使用）
    column_names: List[str] = field(default_factory=list)


# 新数据格式配置（202603.csv）
NEW_DATA_FORMAT = DataFormatConfig(
    sep="\t",
    header=None,  # 无表头
    column_names=[
        "线索唯一ID",
        "客户ID_店端_备用",  # 列2（大部分为空）
        "客户ID",
        "手机号_脱敏",
        "线索创建时间",
        "一级渠道名称",
        "二级渠道名称",
        "三级渠道名称",
        "四级渠道名称",
        "线索类型",
        "客户性别_备用",  # 列11（大部分为空）
        "所在城市",
        "首触意向车型",
        "预算区间_备用",  # 列14（大部分为空）
        "分配时间",
        "线索下发时间",
        "跟进时间_备用",  # 列17-20（大部分为空）
        "跟进内容_备用",
        "跟进结果_备用",
        "跟进备注_备用",
        "首触时间",
        "通话次数",
        "通话总时长",
        "通话内容_备用",  # 列24-25（大部分为空）
        "通话备注_备用",
        "首触跟进记录",
        "首触线索评级",
        "非首触跟进时间",
        "非首触跟进记录",
        "线索评级变化时间",
        "线索评级结果",
        "客户是否主动询问交车时间",
        "客户是否主动询问购车权益",
        "客户是否主动询问金融政策",
        "客户是否同意加微信",
        "客户是否表示门店距离太远拒绝到店",
        "到店时间",
        "到店经销商ID",
        "试驾时间",
        "下订时间",
        "战败原因",
        "SOP开口标签",
        "意向金支付状态",
        "历史订单次数",
        "历史到店次数",
        "历史试驾次数",
    ]
)

# 原数据格式配置（20260308-v2.csv）
OLD_DATA_FORMAT = DataFormatConfig(
    sep=",",
    header=0,  # 有表头
    column_names=[]  # 从文件读取
)


def detect_data_format(file_path: str) -> DataFormatConfig:
    """
    自动检测数据格式

    Args:
        file_path: 数据文件路径

    Returns:
        数据格式配置
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    # 检测分隔符
    if '\t' in first_line:
        # Tab分隔，可能是新格式
        # 检查是否以 DIS 开头（线索ID）- 新格式无表头
        if first_line.startswith('DIS'):
            return NEW_DATA_FORMAT
    elif ',' in first_line:
        # 逗号分隔，检查是否有中文表头
        if '到店标签' in first_line or '线索唯一ID' in first_line:
            return OLD_DATA_FORMAT

    # 默认使用原格式
    return OLD_DATA_FORMAT


def calculate_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算目标标签（从原始时间字段派生）

    Args:
        df: 原始数据框

    Returns:
        添加了目标标签的数据框
    """
    df = df.copy()

    # 解析时间字段
    create_time = pd.to_datetime(df.get("线索创建时间"), errors='coerce')
    arrive_time = pd.to_datetime(df.get("到店时间"), errors='coerce')
    drive_time = pd.to_datetime(df.get("试驾时间"), errors='coerce')

    # 计算到店标签
    if arrive_time.notna().any():
        arrive_days = (arrive_time - create_time).dt.total_seconds() / 86400

        df["到店标签_7天"] = ((arrive_days >= 0) & (arrive_days <= 7)).astype(int)
        df["到店标签_14天"] = ((arrive_days >= 0) & (arrive_days <= 14)).astype(int)
        df["到店标签_30天"] = ((arrive_days >= 0) & (arrive_days <= 30)).astype(int)
    else:
        df["到店标签_7天"] = 0
        df["到店标签_14天"] = 0
        df["到店标签_30天"] = 0

    # 计算试驾标签
    if drive_time.notna().any():
        drive_days = (drive_time - create_time).dt.total_seconds() / 86400

        df["试驾标签_7天"] = ((drive_days >= 0) & (drive_days <= 7)).astype(int)
        df["试驾标签_14天"] = ((drive_days >= 0) & (drive_days <= 14)).astype(int)
        df["试驾标签_30天"] = ((drive_days >= 0) & (drive_days <= 30)).astype(int)
    else:
        df["试驾标签_7天"] = 0
        df["试驾标签_14天"] = 0
        df["试驾标签_30天"] = 0

    # 映射线索评级（如果存在线索评级结果但无线索评级_试驾前）
    if "线索评级结果" in df.columns and "线索评级_试驾前" not in df.columns:
        df["线索评级_试驾前"] = df["线索评级结果"].fillna("Unknown")

    # 成交标签（如果有下订时间）
    if "下订时间" in df.columns:
        df["成交标签"] = df["下订时间"].notna().astype(int)
    else:
        df["成交标签"] = 0

    return df


def derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    衍生时间特征

    Args:
        df: 原始数据框

    Returns:
        添加了时间特征的数据框
    """
    df = df.copy()

    # 解析线索创建时间
    create_time = pd.to_datetime(df.get("线索创建时间"), errors='coerce')

    if create_time.notna().any():
        # 星期几（1=周一，7=周日）
        df["线索创建星期几"] = create_time.dt.dayofweek + 1

        # 小时（0-23）
        df["线索创建小时"] = create_time.dt.hour

    return df


def load_and_adapt_data(
    file_path: str,
    format_config: Optional[DataFormatConfig] = None
) -> pd.DataFrame:
    """
    加载并适配数据

    Args:
        file_path: 数据文件路径
        format_config: 数据格式配置（None则自动检测）

    Returns:
        适配后的数据框
    """
    # 自动检测格式
    if format_config is None:
        format_config = detect_data_format(file_path)

    # 构建读取参数
    read_params = {
        "sep": format_config.sep,
        "header": format_config.header,
        "names": format_config.column_names if format_config.header is None else None,
    }

    # 新格式（无表头）需要特殊处理不规则行
    if format_config.header is None:
        read_params["engine"] = "python"
        read_params["on_bad_lines"] = "skip"
    else:
        read_params["low_memory"] = False

    # 加载数据
    df = pd.read_csv(file_path, **read_params)

    # 计算目标标签
    df = calculate_target_labels(df)

    # 衍生时间特征
    df = derive_time_features(df)

    return df


def get_column_mapping() -> Dict[str, str]:
    """
    获取新数据到标准列名的映射

    Returns:
        列名映射字典
    """
    return {
        # 新数据列名 -> 标准列名
        "客户性别_备用": "客户性别",
        "预算区间_备用": "预算区间",
        "首触意向车型": "首触意向车型",
        "通话总时长": "通话总时长",
        "平均通话时长": "平均通话时长",  # 新数据没有此列，需要计算
        "最后一次通话距今天数": "最后一次通话距今天数",  # 新数据没有此列
        "首触线索是否及时外呼": "首触线索是否及时外呼",
        "首触线索当天是否联通实体卡外呼": "首触线索当天是否联通实体卡外呼",
        "通话时长是否>=45秒": "通话时长是否大于等于45秒",
    }