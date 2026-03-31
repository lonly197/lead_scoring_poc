"""
配置管理模块

管理项目的所有配置项，包括数据路径、训练参数、特征定义等。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - 测试环境可不安装 python-dotenv
    def load_dotenv(*args, **kwargs):
        return False

# 加载环境变量
load_dotenv()


def _optional_env_float(key: str) -> float | None:
    """获取可选的浮点数环境变量，空值返回 None。"""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _optional_env_int(key: str) -> int | None:
    """获取可选的整数环境变量，空值返回 None。"""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return None
    return int(value)


@dataclass
class DataConfig:
    """数据配置"""

    # 数据文件路径（动态拆分模式）
    data_path: str = field(
        default_factory=lambda: os.getenv(
            "DATA_PATH", "./data/20260308-v2.csv"
        )
    )

    # 提前拆分的数据路径（优先级更高）
    # 若配置了这两个路径，将优先使用提前拆分的数据文件
    train_data_path: str | None = field(
        default_factory=lambda: os.getenv("TRAIN_DATA_PATH") or None
    )
    test_data_path: str | None = field(
        default_factory=lambda: os.getenv("TEST_DATA_PATH") or None
    )

    # 通用随机切分比例（旧训练脚本/通用脚本仍可能使用）
    # OHAB 主流程已改为 smart_split_data，默认走随机分组切分 70/15/15
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

    def has_split_data(self) -> bool:
        """检查是否配置了提前拆分的数据路径"""
        return self.train_data_path is not None and self.test_data_path is not None


@dataclass
class ModelConfig:
    """通用模型配置（到店/试驾等非 OHAB 任务仍可复用）"""

    # 模型预设: best_quality, high_quality, good_quality, medium_quality
    preset: str = field(
        default_factory=lambda: os.getenv("MODEL_PRESET", "high_quality")
    )

    # 训练时间限制（秒）
    time_limit: int = field(
        default_factory=lambda: int(os.getenv("TIME_LIMIT", "3600"))
    )

    # 通用任务默认评估指标
    eval_metric: str = field(
        default_factory=lambda: os.getenv("MODEL_EVAL_METRIC", "roc_auc")
    )

    # 通用任务默认目标变量
    target_label: str = field(
        default_factory=lambda: os.getenv("TARGET_LABEL", "到店标签_14天")
    )

    # 指定训练的模型类型（白名单模式）
    # 可选值: CAT (CatBoost), GBM (LightGBM), XGB (XGBoost), RF (RandomForest), etc.
    # 多个模型用逗号分隔，如 "CAT,GBM"
    # 空值表示使用默认预设中的所有模型
    included_model_types: str = field(
        default_factory=lambda: os.getenv("MODEL_INCLUDED_TYPES", "")
    )

    # Bagging folds 数量（None=使用预设默认值，0=禁用 bagging，>=2=启用）
    # 空字符串表示 None（使用 preset 默认）
    # 注意：AutoGluon 不接受 num_bag_folds=1，必须是 0 或 >=2
    num_bag_folds: int | None = field(
        default_factory=lambda: _optional_env_int("MODEL_NUM_BAG_FOLDS")
    )

    # Stacking 层数（None=使用预设默认值，0=禁用 stacking）
    # 空字符串表示 None（使用 preset 默认）
    num_stack_levels: int | None = field(
        default_factory=lambda: _optional_env_int("MODEL_NUM_STACK_LEVELS")
    )


@dataclass
class OHABConfig:
    """OHAB/HAB 训练运行时默认配置。"""

    training_profile: str = field(
        default_factory=lambda: os.getenv("OHAB_TRAINING_PROFILE", "server_16g_compare")
    )
    model_preset: str = field(
        default_factory=lambda: os.getenv("OHAB_MODEL_PRESET", "good_quality")
    )
    time_limit: int = field(
        default_factory=lambda: int(os.getenv("OHAB_TIME_LIMIT", "5400"))
    )
    eval_metric: str = field(
        default_factory=lambda: os.getenv("OHAB_EVAL_METRIC", "balanced_accuracy")
    )
    num_bag_folds: int = field(
        default_factory=lambda: int(os.getenv("OHAB_NUM_BAG_FOLDS", "3"))
    )
    label_mode: str = field(
        default_factory=lambda: os.getenv("OHAB_LABEL_MODE", "hab")
    )
    enable_model_comparison: bool = field(
        default_factory=lambda: os.getenv("OHAB_ENABLE_MODEL_COMPARISON", "true").lower() in {"1", "true", "yes", "y", "on"}
    )
    baseline_family: str = field(
        default_factory=lambda: os.getenv("OHAB_BASELINE_FAMILY", "gbm")
    )
    memory_limit_gb: float | None = field(
        default_factory=lambda: _optional_env_float("OHAB_MEMORY_LIMIT_GB")
    )
    fit_strategy: str = field(
        default_factory=lambda: os.getenv("OHAB_FIT_STRATEGY", "sequential")
    )
    excluded_model_types: str = field(
        default_factory=lambda: os.getenv("OHAB_EXCLUDED_MODEL_TYPES", "RF,XT,KNN,FASTAI,NN_TORCH")
    )
    num_folds_parallel: int = field(
        default_factory=lambda: int(os.getenv("OHAB_NUM_FOLDS_PARALLEL", "1"))
    )
    max_memory_ratio: float = field(
        default_factory=lambda: float(os.getenv("OHAB_MAX_MEMORY_RATIO", "0.7"))
    )
    generate_plots: bool = field(
        default_factory=lambda: os.getenv("OHAB_GENERATE_PLOTS", "false").lower() in {"1", "true", "yes", "y", "on"}
    )
    split_mode: str = field(
        default_factory=lambda: os.getenv("OHAB_SPLIT_MODE", "random")
    )
    auto_oot_min_days: int = field(
        default_factory=lambda: int(os.getenv("OHAB_AUTO_OOT_MIN_DAYS", "90"))
    )
    pipeline_mode: str = field(
        default_factory=lambda: os.getenv("OHAB_PIPELINE_MODE", "two_stage")
    )
    split_group_mode: str = field(
        default_factory=lambda: os.getenv("OHAB_SPLIT_GROUP_MODE", "phone_or_lead")
    )
    feature_profile: str = field(
        default_factory=lambda: os.getenv("OHAB_FEATURE_PROFILE", "auto_scorecard")
    )


@dataclass
class EnsembleConfig:
    """三模型集成训练配置（7/14/21 天试驾预测）"""

    # 是否并行训练三个模型
    # true = 并行训练（需 32GB+ 内存）
    # false = 顺序训练（16GB 内存推荐）
    parallel_training: bool = field(
        default_factory=lambda: os.getenv("ENSEMBLE_PARALLEL_TRAINING", "false").lower() in {"1", "true", "yes", "y", "on"}
    )

    # 并行训练的最大进程数
    # 建议：32GB 内存设为 3，24GB 内存设为 2
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("ENSEMBLE_MAX_WORKERS", "3"))
    )


@dataclass
class FeatureConfig:
    """特征配置 - 适配 20260308-v2.csv"""

    # ID 类字段（不用于训练）
    # 注：新数据格式可能缺少部分字段
    id_columns: List[str] = field(
        default_factory=lambda: [
            # === 主键和唯一标识 ===
            "线索唯一ID",
            "客户ID",
            "客户ID_店端",
            "手机号_脱敏",
            "手机号码",  # 未脱敏版本
            "主键",
            "店端潜客编号",
            "split_group_key",
            # === 经销店相关 ===
            "经销店代码",  # ID 类，若需作为特征需单独处理
            "到店经销商ID",  # 后验 ID
            # 以下字段新数据可能缺失
            # "经销店名称",
        ]
    )

    # 目标泄漏风险字段（不用于训练）
    leakage_columns: List[str] = field(
        default_factory=lambda: [
            # ============================================================
            # 一、直接编码目标（致命泄漏，必须删除）
            # ============================================================
            "是否到店",  # 直接编码到店行为（致命！）
            "是否试驾",  # 直接编码试驾行为（致命！）
            "试驾天数差",  # 直接编码时间窗口（致命！）
            "试驾时间",  # 计算标签的来源
            "到店时间",  # 到店行为的后验信息
            "下订时间",  # 订单行为的后验信息
            "下定状态",  # 直接编码订单状态
            "意向金支付状态",  # 订单相关
            "战败原因",  # 战败后验
            "战败次数",  # 战败后验

            # ============================================================
            # 二、后验信息（大概率删除，若无评分时点冻结值）
            # ============================================================
            "线索当前意向级别",  # 可能包含试驾后的评级变化（后验）
            "线索评级结果_最新",  # 可能包含后验信息
            "线索评级结果",  # 原始评级，极度危险
            "线索评级变化时间",  # 评级变化的后验时间
            "线索评级变化时间_备用",
            "意向级别",  # 可能是后验的意向级别
            "购车阶段",  # 可能随时间变化的后验信息
            "线索评级_试驾前",  # 目标变量相关
            "线索评级_试驾后",  # 明确的后验信息
            "线索评级结果_备用",
            "商机评级_试驾后",
            "latest_intn_level",

            # ============================================================
            # 三、JSON 字段（包含后验信息或 100% 缺失）
            # ============================================================
            "行业标签分析结果_JSON",  # 包含"是否成功邀请到店/试驾"等后验标签
            "跟进记录_JSON",
            "跟进详情_JSON",
            "首触跟进记录",  # 可能包含后验信息
            "非首触跟进记录",  # 可能包含后验信息
            "关注点json",  # 100% 缺失
            "希望试驾体验的功能json",  # 100% 缺失
            "爱好json",  # 100% 缺失
            "顾虑点json",  # 可能包含后验信息
            "竞品车型及试驾json",  # 可能包含后验信息
            "备忘",  # 非结构化文本

            # ============================================================
            # 四、JSON 提取特征（后验泄漏风险）
            # ============================================================
            # 从跟进详情_JSON 提取的特征，可能包含试驾后的后验信息
            "提及试驾",  # 致命泄漏：备忘包含"已试驾"直接泄漏目标
            "提及到店",  # 高风险：试驾通常发生在到店，备忘可能记录"到店试驾"
            "最终战败",  # 中风险：战败可能在试驾后发生
            "意向级别下降",  # 中风险：意向级别可能在试驾后变化

            # ============================================================
            # 五、其他后验信息
            # ============================================================
            "最后一次通话距今天数",  # 隐含了数据导出后的未来信息
            "defeat_reason_content",

            # ============================================================
            # 六、订单相关（泄漏成交标签）
            # ============================================================
            "订单状态",
            "成交日期",
            "结算日期",
            "订单号",
            "customer_order_no",
            "意向金支付状态_备用",

            # ============================================================
            # 七、目标变量标签（严禁用于训练）
            # ============================================================
            "label_7天内试驾",  # 管道生成的标签
            "label_14天内试驾",  # 管道生成的标签
            "label_21天内试驾",  # 管道生成的标签
            "label_OHAB",  # 管道生成的 OHAB 级别
            "到店标签_7天",
            "到店标签_14天",
            "到店标签_30天",
            "试驾标签_7天",
            "试驾标签_14天",
            "试驾标签_30天",
            "成交标签",
            "is_final_ordered",  # 终态验证隔离列，严禁用于训练
        ]
    )


    # 时间字段（需要转换为特征）
    time_columns: List[str] = field(
        default_factory=lambda: [
            "线索创建时间",
            "分配时间",
            "线索下发时间",
            "首触时间",
            "客户建档时间",  # 可提取客户活跃时长
        ]
    )

    # 类别特征
    # 注：部分字段新数据可能缺失，在训练时动态过滤
    categorical_features: List[str] = field(
        default_factory=lambda: [
            # === 渠道特征 ===
            "一级渠道名称",
            "二级渠道名称",
            "三级渠道名称",
            "四级渠道名称",
            "首触渠道属性",
            # === 线索属性 ===
            "线索类型",
            "客户性别",
            "所在城市",
            "省份名称",
            "市名称",
            "行政区名称",
            "职业",
            # === 车型/预算 ===
            "首触意向车型",
            "预算区间",
            # === 前置评级（关键特征）===
            "首触线索评级",  # 关键前置特征：线索创建时的评级
            # === AI 分析字段（可能为空）===
            "客户是否主动询问交车时间",
            "客户是否主动询问购车权益",
            "客户是否主动询问金融政策",
            "客户是否同意加微信",
            "客户是否表示门店距离太远拒绝到店",
            # === 业务标签 ===
            "SOP开口标签",
            "客户类型",
            # === 交互特征（分类）===
            "响应及时性",
            "渠道组合",
        ]
    )

    # 数值特征
    numeric_features: List[str] = field(
        default_factory=lambda: [
            # === 通话特征 ===
            "通话次数",
            "通话总时长",
            "平均通话时长",
            "平均通话时长_派生",
            "通话时长是否大于等于45秒",
            "是否接通",
            "首触线索是否及时外呼",
            "首触线索当天是否联通实体卡外呼",
            # === 历史行为 ===
            "历史订单次数",
            "历史到店次数",
            "历史试驾次数",
            # === 时间特征 ===
            "线索创建星期几",
            "线索创建小时",
            # === JSON 提取特征 ===
            "跟进总次数",
            "接通次数",
            "未接通次数",
            "接通率",
            "继续跟进次数",
            "平均通话时长_秒",
            "最长通话时长_秒",
            # === 交互特征 ===
            "首触响应时长_小时",
            "城市车型热度",
            "有效通话",
            "AI语义标签可用",
            "语义标签命中数",
            "JSON跟进明细可用",
            # === 跟进特征 ===
            "跟进次数",
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
    ohab: OHABConfig = field(default_factory=OHABConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
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
