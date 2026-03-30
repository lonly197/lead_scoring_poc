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
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataFormatConfig:
    """数据格式配置"""

    # 分隔符
    sep: str = ","

    # 是否有表头
    header: Optional[int] = 0  # 0=第一行, None=无表头

    # 列名映射（当无表头时使用）
    column_names: List[str] = field(default_factory=list)


# 新数据格式配置（202602~03.tsv / 202603.tsv - 46列）
# 无表头 TSV 直接按 SQL 导出列顺序定义标准字段名，避免在中间产物中暴露“_备用”占位名。
NEW_DATA_FORMAT = DataFormatConfig(
    sep="\t",
    header=None,  # 无表头
    column_names=[
        "线索唯一ID",           # 列0: DIS260301001002008
        "客户ID",               # 列1: customer_id
        "客户ID_店端",          # 列2: dealer_customer_id
        "手机号_脱敏",          # 列3: 18562660372
        "线索创建时间",         # 列4: 2026-03-01 00:10:03
        "一级渠道名称",         # 列5: 垂媒集采
        "二级渠道名称",         # 列6: 汽车之家-车商汇
        "三级渠道名称",         # 列7: 留资
        "四级渠道名称",         # 列8: 12月项目/直播等
        "线索类型",             # 列9: clue_type
        "客户性别",             # 列10: gender
        "所在城市",             # 列11: 青岛市
        "首触意向车型",         # 列12: 铂智3X
        "预算区间",             # 列13: buy_budget
        "分配时间",             # 列14: 时间
        "线索下发时间",         # 列15: 时间
        "通话次数",             # 列16: call_cnt
        "通话总时长",           # 列17: call_duration_sum
        "平均通话时长",         # 列18: call_duration_avg
        "最后一次通话距今天数", # 列19: last_call_days
        "首触时间",             # 列20: 时间
        "首触线索是否及时外呼",         # 列21
        "首触线索当天是否联通实体卡外呼", # 列22
        "通话时长是否大于等于45秒",      # 列23
        "是否接通",                      # 列24
        "首触跟进记录",                  # 列25: 现网导出中通常为 JSON 字符串
        "首触线索评级",                  # 列26
        "非首触跟进时间",                # 列27
        "非首触跟进记录",                # 列28: 现网导出中通常为 JSON 字符串
        "线索评级变化时间",              # 列29
        "线索评级结果",                  # 列30
        "客户是否主动询问交车时间",   # 列31: 是/否
        "客户是否主动询问购车权益",   # 列32: 是/否
        "客户是否主动询问金融政策",   # 列33: 是/否
        "客户是否同意加微信",         # 列34: 是/否
        "客户是否表示门店距离太远拒绝到店",  # 列35: 是/否
        "到店时间",             # 列36: 时间
        "到店经销商ID",         # 列37
        "试驾时间",             # 列38
        "下订时间",             # 列39
        "战败原因",             # 列40
        "SOP开口标签",          # 列41
        "意向金支付状态",       # 列42
        "历史订单次数",         # 列43
        "历史到店次数",         # 列44
        "历史试驾次数",         # 列45
    ]
)

# 原数据格式配置（20260308-v2.csv）
OLD_DATA_FORMAT = DataFormatConfig(
    sep=",",
    header=0,  # 有表头
    column_names=[]  # 从文件读取
)


SCHEMA_CONTRACT_VERSION = "v2"
SCHEMA_ALIAS_MAPPING = {
    "客户ID_店端_备用": "客户ID_店端",
    "客户ID(店端)": "客户ID_店端",
    "手机号（脱敏）": "手机号_脱敏",
    "线索类型_备用": "线索类型",
    "客户性别_备用": "客户性别",
    "预算区间_备用": "预算区间",
    "预算区间(购车预算)": "预算区间",
    "首触意向车型/意向车型": "首触意向车型",
    "通话时长是否>=45秒": "通话时长是否大于等于45秒",
    "客户是否主动询问购车权益（优惠）": "客户是否主动询问购车权益",
    "跟进记录_JSON": "首触跟进记录",
    "跟进详情_JSON": "非首触跟进记录",
    "首触跟进记录_备用": "首触跟进记录",
    "非首触跟进时间_备用": "非首触跟进时间",
    "非首触跟进时间_备用2": "非首触跟进时间",
    "线索评级变化时间_备用": "线索评级变化时间",
    "到店经销商ID_备用": "到店经销商ID",
    "到店日期_备用": "试驾时间",
    "SOP开口标签_备用": "SOP开口标签",
    "意向金支付状态_备用": "意向金支付状态",
    "历史订单次数_备用": "历史订单次数",
    "历史到店次数_备用": "历史到店次数",
    "历史试驾次数_备用": "历史试驾次数",
}


def _series_is_missing(series: pd.Series) -> pd.Series:
    """判断列值是否为空，兼容空字符串。"""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.isna() | series.astype(str).str.strip().eq("")
    return series.isna()


def normalize_schema_contract(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    将原始导出列收口为训练期使用的标准字段名，并返回收口元数据。
    """
    df = df.copy()
    applied_aliases: Dict[str, str] = {}
    filled_aliases: Dict[str, str] = {}
    dropped_aliases: List[str] = []

    for alias, canonical in SCHEMA_ALIAS_MAPPING.items():
        if alias not in df.columns:
            continue

        if canonical not in df.columns:
            df = df.rename(columns={alias: canonical})
            applied_aliases[alias] = canonical
            continue

        alias_missing = _series_is_missing(df[alias])
        canonical_missing = _series_is_missing(df[canonical])
        fill_mask = canonical_missing & ~alias_missing
        if fill_mask.any():
            df.loc[fill_mask, canonical] = df.loc[fill_mask, alias]
            filled_aliases[alias] = canonical

        if alias != canonical:
            df = df.drop(columns=[alias])
            dropped_aliases.append(alias)

    schema_contract = {
        "version": SCHEMA_CONTRACT_VERSION,
        "applied_aliases": applied_aliases,
        "filled_aliases": filled_aliases,
        "dropped_aliases": dropped_aliases,
        "canonical_columns": sorted(df.columns.tolist()),
    }

    return df, schema_contract


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
        df["试驾标签_21天"] = ((drive_days >= 0) & (drive_days <= 21)).astype(int)
        df["试驾标签_30天"] = ((drive_days >= 0) & (drive_days <= 30)).astype(int)
    else:
        df["试驾标签_7天"] = 0
        df["试驾标签_14天"] = 0
        df["试驾标签_21天"] = 0
        df["试驾标签_30天"] = 0

    # 保留历史兼容目标列，默认主口径仍应使用“线索评级结果”
    if "线索评级结果" in df.columns and "线索评级_试驾前" not in df.columns:
        df["线索评级_试驾前"] = df["线索评级结果"].fillna("Unknown")

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
    format_config: Optional[DataFormatConfig] = None,
    return_metadata: bool = False,
) -> Any:
    """
    加载并适配数据

    Args:
        file_path: 数据文件路径
        format_config: 数据格式配置（None则自动检测）

    Returns:
        适配后的数据框
    """
    # 文件路径验证
    if not file_path or not file_path.strip():
        raise ValueError("数据文件路径不能为空")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"路径不是有效文件: {file_path}")

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

        read_params["low_memory"] = False
    else:
        read_params["low_memory"] = False

    # 加载数据
    try:
        df = pd.read_csv(file_path, **read_params)
    except Exception:
        if format_config.header is None:
            fallback_params = dict(read_params)
            fallback_params["engine"] = "python"
            fallback_params["on_bad_lines"] = "skip"
            df = pd.read_csv(file_path, **fallback_params)
        else:
            raise

    # 验证关键列是否存在
    required_columns = ["线索创建时间", "线索唯一ID"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"数据格式错误：缺少必需列 {missing_columns}。"
            f"当前列名: {list(df.columns[:10])}...（共{len(df.columns)}列）"
        )

    # 数据契约收口
    df, schema_contract = normalize_schema_contract(df)

    # 计算目标标签
    df = calculate_target_labels(df)

    # 衍生时间特征
    df = derive_time_features(df)

    # === JSON 特征提取 ===
    # 优先使用正式字段名“非首触跟进记录”，兼容旧技术名“跟进详情_JSON”
    from .json_extractor import batch_extract_json_features

    adaptation_metadata = {
        "schema_contract": schema_contract,
    }

    json_feature_source = None
    if "非首触跟进记录" in df.columns:
        json_feature_source = "非首触跟进记录"
    elif "跟进详情_JSON" in df.columns:
        json_feature_source = "跟进详情_JSON"

    if json_feature_source:
        try:
            df = batch_extract_json_features(df, json_feature_source, drop_original=False)
            print("JSON 特征提取完成")
            adaptation_metadata["json_feature_source"] = json_feature_source
            adaptation_metadata["json_extraction_status"] = "success"
        except Exception as e:
            logger.warning(f"JSON 特征提取失败: {e}")
            adaptation_metadata["json_extraction_status"] = "failed"
            adaptation_metadata["json_extraction_error"] = str(e)

    if return_metadata:
        return df, adaptation_metadata

    df.attrs["adaptation_metadata"] = adaptation_metadata

    return df
