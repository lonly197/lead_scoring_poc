"""
配置管理模块

管理项目的所有配置项，包括数据路径、训练参数、特征定义等。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class DataConfig:
    """数据配置"""

    # 数据文件路径
    data_path: str = field(
        default_factory=lambda: os.getenv(
            "DATA_PATH", "./data/20260308-v2.csv"
        )
    )

    # 训练/测试集划分比例
    test_size: float = field(
        default_factory=lambda: float(os.getenv("TRAIN_TEST_SPLIT_RATIO", "0.2"))
    )

    # 随机种子
    random_seed: int = field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )

    # 数据格式：auto, old, new
    # auto: 自动检测
    # old: 逗号分隔，有表头（20260308-v2.csv）
    # new: Tab分隔，无表头（202603.csv）
    data_format: str = field(
        default_factory=lambda: os.getenv("DATA_FORMAT", "auto")
    )


@dataclass
class ModelConfig:
    """模型配置"""

    # 模型预设: best_quality, high_quality, good_quality, medium_quality
    preset: str = field(
        default_factory=lambda: os.getenv("MODEL_PRESET", "high_quality")
    )

    # 训练时间限制（秒）
    time_limit: int = field(
        default_factory=lambda: int(os.getenv("TIME_LIMIT", "3600"))
    )

    # 评估指标
    eval_metric: str = "roc_auc"

    # 目标变量
    target_label: str = field(
        default_factory=lambda: os.getenv("TARGET_LABEL", "到店标签_14天")
    )


@dataclass
class FeatureConfig:
    """特征配置 - 适配 20260308-v2.csv"""

    # ID 类字段（不用于训练）
    # 注：新数据格式可能缺少部分字段
    id_columns: List[str] = field(
        default_factory=lambda: [
            "线索唯一ID",
            "客户ID",
            "客户ID_店端",
            "手机号_脱敏",
            # 以下字段新数据可能缺失
            # "经销店代码",
            # "经销店名称",
        ]
    )

    # 目标泄漏风险字段（不用于训练）
    leakage_columns: List[str] = field(
        default_factory=lambda: [
            # 到店相关（泄漏到店标签）
            "到店时间",
            "到店经销商ID",
            "到店经销商ID_备用",
            "到店日期_备用",
            "首次到店时间",
            # 试驾相关（泄漏试驾标签）
            "试驾时间",
            "首次试驾完成时间",
            # 订单相关（泄漏成交标签）
            "订单状态",
            "下订时间",
            "成交日期",
            "结算日期",
            "订单号",
            "customer_order_no",
            "意向金支付状态",
            "意向金支付状态_备用",
            # 战败相关
            "战败原因",
            "defeat_reason_content",
            # 线索评级（预测到店时需排除）
            "线索评级_试驾前",  # 目标变量
            "线索评级结果",     # 原始评级，极度危险，会泄漏目标变量
            "线索评级结果_备用",
            "线索评级_试驾后",
            "线索评级变化时间",
            "线索评级变化时间_备用",
            "商机评级_试驾后",
            "latest_intn_level",
            # JSON 字段
            "跟进记录_JSON",
            "跟进详情_JSON",
            # 其他目标变量
            "到店标签_7天",
            "到店标签_14天",
            "到店标签_30天",
            "试驾标签_7天",
            "试驾标签_14天",
            "试驾标签_30天",
            "成交标签",
            "is_final_ordered",  # 终态验证隔离列，严禁用于训练
            # 动态后验特征（风险）
            "最后一次通话距今天数", # 隐含了数据导出后的未来信息
        ]
    )


    # 时间字段（需要转换为特征）
    time_columns: List[str] = field(
        default_factory=lambda: [
            "线索创建时间",
            "分配时间",
            "线索下发时间",
            "首触时间",
        ]
    )

    # 类别特征
    # 注：部分字段新数据可能缺失，在训练时动态过滤
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "一级渠道名称",
            "二级渠道名称",
            "三级渠道名称",
            "四级渠道名称",
            "线索类型",
            "客户性别",
            "所在城市",
            # "所在省份",  # 新数据可能缺失
            "首触意向车型",
            "预算区间",
            "首触线索评级", # 关键前置特征：用于预测 latest_intn_level 演变
            # AI 分析字段（可能为空）
            "客户是否主动询问交车时间",
            "客户是否主动询问购车权益",
            "客户是否主动询问金融政策",
            "客户是否同意加微信",
            "客户是否表示门店距离太远拒绝到店",
            # 业务标签
            "SOP开口标签",
            "意向金支付状态",
            # 新增交互特征（分类）
            "响应及时性",
            "渠道组合",
        ]
    )

    # 数值特征
    numeric_features: List[str] = field(
        default_factory=lambda: [
            # 原有特征
            "通话次数",
            "通话总时长",
            "平均通话时长",
            "平均通话时长_派生",
            "最后一次通话距今天数",
            "通话时长是否大于等于45秒",
            "是否接通",
            "首触线索是否及时外呼",
            "首触线索当天是否联通实体卡外呼",
            "历史订单次数",
            "历史到店次数",
            "历史试驾次数",
            "线索创建星期几",
            "线索创建小时",
            # 新增 JSON 提取特征
            "跟进总次数",
            "接通次数",
            "未接通次数",
            "接通率",
            "继续跟进次数",
            "平均通话时长_秒",
            "最长通话时长_秒",
            # 新增交互特征
            "首触响应时长_小时",
            "城市车型热度",
            "有效通话",
            "AI语义标签可用",
            "语义标签命中数",
            "JSON跟进明细可用",
        ]
    )


@dataclass
class OutputConfig:
    """输出配置"""

    # 输出目录
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./outputs"))
    )

    # 模型保存目录
    models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    topk_dir: Path = field(init=False)

    def __post_init__(self):
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.topk_dir = self.output_dir / "topk_lists"

        # 确保目录存在
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.topk_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """主配置类"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls()


# 全局配置实例
config = Config.from_env()


def get_excluded_columns(target_label: str) -> List[str]:
    """
    获取需要排除的列名列表

    Args:
        target_label: 目标变量名称

    Returns:
        需要排除的列名列表
    """
    excluded = config.feature.id_columns + config.feature.leakage_columns

    # 确保目标变量本身不被排除
    if target_label in excluded:
        excluded.remove(target_label)

    return excluded


def get_feature_columns(target_label: str, all_columns: List[str]) -> List[str]:
    """
    获取可用于训练的特征列

    Args:
        target_label: 目标变量名称
        all_columns: 数据集所有列名

    Returns:
        可用于训练的特征列列表
    """
    excluded = get_excluded_columns(target_label)
    features = [col for col in all_columns if col not in excluded]
    return features
