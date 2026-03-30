"""
数据管道配置模块

定义管道的默认配置和参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    """
    数据管道配置

    包含各步骤的默认参数和阈值设置。
    """

    # ====================
    # 合并配置 (merge)
    # ====================
    merge: Dict = field(default_factory=lambda: {
        "output_format": "parquet",
        "dmp_columns": [
            "dmp_手机号", "dmp_事件时间", "dmp_事件名称", "dmp_车型代码",
            "dmp_品牌", "dmp_页面路径", "dmp_数值", "dmp_渠道",
            "dmp_分类", "dmp_未知", "dmp_平台",
        ],
        "phone_column_patterns": ["手机", "phone"],
    })

    # ====================
    # 探查配置 (profile)
    # ====================
    profile: Dict = field(default_factory=lambda: {
        "target_keywords": ["评级", "标签", "结果", "level", "label", "target"],
        "high_missing_threshold": 0.5,
        "medium_missing_threshold": 0.1,
        "imbalance_threshold": 10.0,
    })

    # ====================
    # 清洗配置 (clean)
    # ====================
    clean: Dict = field(default_factory=lambda: {
        # 基础清洗开关
        "drop_high_missing": True,
        "drop_duplicates": True,
        "high_missing_threshold": 0.5,  # 缺失率 > 50% 的列将被删除

        # 异常值检测配置
        # IQR 方法: Q1 - k*IQR ~ Q3 + k*IQR 范围外为异常值
        # k=1.5 为标准值，覆盖 ~99.3% 的正态分布数据
        # k=3.0 为保守值，覆盖 ~99.9% 的正态分布数据
        "detect_outliers": True,
        "outlier_method": "iqr",  # 可选: "iqr" 或 "zscore"
        "outlier_threshold": 1.5,  # IQR 乘数 k；zscore 模式下为标准差倍数

        # 偏斜分布检测
        # 偏度 |skewness| > 1 表示显著偏斜，建议进行 log/box-cox 变换
        # 正态分布偏度为 0，右偏分布偏度 > 0，左偏分布偏度 < 0
        "handle_skewed": True,
        "skew_threshold": 1.0,  # |偏度| > 1 时标记为偏斜分布

        # 高基数检测
        # 类别列唯一值 > 100 时，one-hot 编码会导致维度爆炸
        # 建议使用频率编码或目标编码
        "high_cardinality_threshold": 100,  # 唯一值数量阈值
        "drop_constant_columns": True,
    })

    # ====================
    # 脱敏配置 (desensitize)
    # ====================
    desensitize: Dict = field(default_factory=lambda: {
        "brand_mapping": {
            "广汽丰田": "品牌A",
            "广丰": "品牌A",
            "广汽": "集团A",
            "GTMC": "代号G",
            "广汽本地": "区域A",
        },
        "id_mask_columns": ["客户ID", "客户ID(店端)"],
        "phone_patterns": ["手机", "phone", "跟进", "战败", "记录"],
        "id_card_pattern": r"\d{15}[\dXx]?[\dXx]?[\dXx]?",
    })

    # ====================
    # 拆分配置 (split)
    # ====================
    split: Dict = field(default_factory=lambda: {
        "mode": "random",
        "target_column": "线索评级结果",
        "test_ratio": 0.2,
        "random_seed": 42,
        "time_column": "线索创建时间",
        "min_oot_days": 30,
    })

    # ====================
    # 通用配置
    # ====================
    default_output_dir: Path = field(default_factory=lambda: Path("./data"))
    reports_dir: Path = field(default_factory=lambda: Path("./reports"))
    cache_dir: Path = field(default_factory=lambda: Path("./data/cache"))
    log_level: str = "INFO"

    def get(self, step: str, key: str, default=None):
        """获取配置值"""
        step_config = getattr(self, step, {})
        return step_config.get(key, default)

    def set(self, step: str, key: str, value):
        """设置配置值"""
        if step not in ["merge", "profile", "clean", "desensitize", "split"]:
            raise ValueError(f"未知的步骤: {step}")
        getattr(self, step)[key] = value


# 默认配置实例
default_config = PipelineConfig()


# ====================
# 脱敏规则常量
# ====================

# 品牌关键词映射
BRAND_MAPPING = {
    "广汽丰田": "品牌A",
    "广丰": "品牌A",
    "广汽": "集团A",
    "GTMC": "代号G",
    "广汽本地": "区域A",
}

# ID 字段掩码列
ID_MASK_COLUMNS = [
    "客户ID",
    "客户ID(店端)",
]

# 需要品牌关键词替换的文本字段
BRAND_TEXT_COLUMNS = [
    "首触意向车型/意向车型",
    "一级渠道名称",
    "二级渠道名称",
    "三级渠道名称",
    "四级渠道名称",
    "首触跟进记录",
    "非首触跟进记录",
    "战败原因",
    "dmp_品牌",
    "dmp_车型代码",
]

# DMP 数据列定义
DMP_COLUMNS = [
    "dmp_手机号", "dmp_事件时间", "dmp_事件名称", "dmp_车型代码",
    "dmp_品牌", "dmp_页面路径", "dmp_数值", "dmp_渠道",
    "dmp_分类", "dmp_未知", "dmp_平台",
]