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


# 新数据格式配置（202603.csv - 46列）
# 根据实际数据推断的列映射
NEW_DATA_FORMAT = DataFormatConfig(
    sep="\t",
    header=None,  # 无表头
    column_names=[
        "线索唯一ID",           # 列0: DIS260301001002008
        "客户ID_店端_备用",     # 列1: 另一个ID
        "客户ID",               # 列2: 2027778329376829441
        "手机号_脱敏",          # 列3: 18562660372
        "线索创建时间",         # 列4: 2026-03-01 00:10:03
        "一级渠道名称",         # 列5: 垂媒集采
        "二级渠道名称",         # 列6: 汽车之家-车商汇
        "三级渠道名称",         # 列7: 留资
        "四级渠道名称",         # 列8: 12月项目/直播等
        "线索类型_备用",        # 列9: 空
        "数值字段_备用",        # 列10: 数值
        "所在城市",             # 列11: 青岛市
        "首触意向车型",         # 列12: 铂智3X
        "预算区间_备用",        # 列13: 空
        "分配时间",             # 列14: 时间
        "线索下发时间",         # 列15: 时间
        "跟进时间_备用",        # 列16: 空
        "跟进内容_备用",        # 列17: 空
        "跟进结果_备用",        # 列18: 空
        "跟进备注_备用",        # 列19: 空
        "首触时间",             # 列20: 时间
        "通话次数",             # 列21: 数值
        "通话总时长",           # 列22: 数值
        "首触跟进记录_备用",    # 列23: 空
        "非首触跟进时间_备用",  # 列24: 空
        "跟进记录_JSON",        # 列25: JSON
        "线索评级结果",         # 列26: H/A/B/O ✓ OHAB评级
        "线索评级变化时间_备用", # 列27: 时间
        "跟进详情_JSON",        # 列28: JSON
        "非首触跟进时间_备用2", # 列29: 时间
        "线索评级_试驾后",      # 列30: H/A/B/O 另一个OHAB评级
        "客户是否主动询问交车时间",   # 列31: 是/否
        "客户是否主动询问购车权益",   # 列32: 是/否
        "客户是否主动询问金融政策",   # 列33: 是/否
        "客户是否同意加微信",         # 列34: 是/否
        "客户是否表示门店距离太远拒绝到店",  # 列35: 是/否
        "到店时间",             # 列36: 时间
        "到店经销商ID_备用",    # 列37: 文本
        "到店日期_备用",        # 列38: 文本
        "试驾时间",             # 列39: 时间
        "战败原因",             # 列40: 文本
        "SOP开口标签_备用",     # 列41: 空
        "意向金支付状态_备用",  # 列42: 空
        "备用字段_43",          # 列43: 空
        "历史订单次数_备用",    # 列44: 数值
        "历史到店次数_备用",    # 列45: 数值
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

    # === O 级合并逻辑 ===
    # 将 O 级强制归并为 H 级（仅针对模型训练的评级字段），以提升高优样本浓度并避免分类不平衡
    if "线索评级_试驾前" in df.columns:
        df["线索评级_试驾前"] = df["线索评级_试驾前"].replace({"O": "H"})

    # 成交标签（如果有下订时间）
    if "下订时间" in df.columns:
        df["成交标签"] = df["下订时间"].notna().astype(int)
        # === 终态验证隔离列 ===
        # 绝不用于训练，仅供 validate_model.py 验证最终转化率使用
        df["is_final_ordered"] = df["下订时间"].notna().astype(int)
    else:
        df["成交标签"] = 0
        df["is_final_ordered"] = 0

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

    # 新格式（无表头）需要特殊处理
    if format_config.header is None:
        # 先读取第一行检测实际列数
        # 注意：不能用strip()，否则会删除末尾的空制表符导致列数计算错误
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        actual_columns = len(first_line.split(format_config.sep))

        defined_columns = len(format_config.column_names)

        if actual_columns != defined_columns:
            # 列数不匹配时，扩展或截断列名
            if actual_columns > defined_columns:
                # 扩展列名：为额外列添加占位名称
                extra_names = [f"_未命名列_{i}" for i in range(actual_columns - defined_columns)]
                read_params["names"] = format_config.column_names + extra_names
                print(f"警告: 数据列数({actual_columns})多于定义列数({defined_columns})，已添加占位列名")
            else:
                # 截断列名
                read_params["names"] = format_config.column_names[:actual_columns]
                print(f"警告: 数据列数({actual_columns})少于定义列数({defined_columns})，已截断列名")

        read_params["engine"] = "python"
        read_params["on_bad_lines"] = "skip"
    else:
        read_params["low_memory"] = False

    # 加载数据
    df = pd.read_csv(file_path, **read_params)

    # 验证关键列是否存在
    required_columns = ["线索创建时间", "线索唯一ID"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"数据格式错误：缺少必需列 {missing_columns}。"
            f"当前列名: {list(df.columns[:10])}...（共{len(df.columns)}列）"
        )

    # 计算目标标签
    df = calculate_target_labels(df)

    # 衍生时间特征
    df = derive_time_features(df)

    # === JSON 特征提取 ===
    # 从跟进详情_JSON 提取高价值业务特征
    from .json_extractor import batch_extract_json_features

    if "跟进详情_JSON" in df.columns:
        df = batch_extract_json_features(df, "跟进详情_JSON", drop_original=False)
        print("JSON 特征提取完成")

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