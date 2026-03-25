"""
数据加载与预处理模块

负责加载 CSV 数据、特征工程、数据划分等功能。
支持多种数据格式：
1. 原格式（20260308-v2.csv）：逗号分隔，有表头，60列
2. 新格式（202603.csv）：Tab分隔，无表头，46列
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .adapter import (
    detect_data_format,
    load_and_adapt_data,
    calculate_target_labels,
    derive_time_features,
    NEW_DATA_FORMAT,
    OLD_DATA_FORMAT,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器

    支持自动检测数据格式并适配：
    - 原格式：逗号分隔，有表头
    - 新格式：Tab分隔，无表头

    注意：适配功能默认关闭，需要显式启用。
    这是为了确保不干扰后续新数据的正常加载。
    """

    def __init__(
        self,
        data_path: str,
        random_seed: int = 42,
        auto_adapt: bool = False,  # 默认关闭适配
    ):
        """
        初始化数据加载器

        Args:
            data_path: 数据文件路径
            random_seed: 随机种子
            auto_adapt: 是否启用自动适配（计算目标变量、衍生特征等）
                       默认 False，保持原有加载行为
                       设为 True 时，会对新格式数据进行适配处理
        """
        self.data_path = Path(data_path)
        self.random_seed = random_seed
        self.auto_adapt = auto_adapt
        self._data: Optional[pd.DataFrame] = None
        self._data_format: Optional[str] = None
        self._adaptation_metadata: dict = {}

    def load(self) -> pd.DataFrame:
        """
        加载数据文件

        Returns:
            加载的 DataFrame
        """
        if self._data is not None:
            return self._data.copy()

        logger.info(f"加载数据: {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        # 根据是否启用适配选择加载方式
        if self.auto_adapt:
            # 启用适配：自动检测格式并处理
            format_config = detect_data_format(str(self.data_path))
            self._data_format = "new" if format_config == NEW_DATA_FORMAT else "old"
            logger.info(f"检测到数据格式: {self._data_format}（适配模式）")
            self._data, self._adaptation_metadata = load_and_adapt_data(
                str(self.data_path),
                return_metadata=True,
            )
        else:
            # 默认：保持原有加载行为
            suffix = self.data_path.suffix.lower()
            self._data_format = "standard"
            if suffix == ".csv":
                self._data = pd.read_csv(self.data_path, low_memory=False)
            elif suffix == ".tsv":
                self._data = pd.read_csv(self.data_path, sep='\t', low_memory=False)
            elif suffix == ".parquet":
                self._data = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"不支持的文件格式: {suffix}")

        logger.info(f"数据加载完成: {len(self._data)} 行, {len(self._data.columns)} 列")
        return self._data.copy()

    def get_data_format(self) -> Optional[str]:
        """
        获取检测到的数据格式

        Returns:
            格式标识，如果尚未加载则返回 None
        """
        return self._data_format

    def get_adaptation_metadata(self) -> dict:
        """获取 auto_adapt 过程中生成的元数据。"""
        return deepcopy(self._adaptation_metadata)

    def get_basic_stats(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        获取数据基本统计信息

        Args:
            df: 数据 DataFrame，如果为 None 则使用已加载的数据

        Returns:
            统计信息字典
        """
        if df is None:
            df = self.load()

        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": df.isnull().sum().to_dict(),
            "missing_ratio": (df.isnull().sum() / len(df) * 100).to_dict(),
        }

        # 目标变量分布 - 匹配中文命名风格（如 "到店标签_14天"、"线索评级_试驾前"）
        target_cols = [col for col in df.columns if "标签" in col or "评级" in col]
        if target_cols:
            stats["target_distribution"] = {
                col: df[col].value_counts().to_dict() for col in target_cols
            }

        # OHAB 分布 - 使用实际的列名
        if "线索评级_试驾前" in df.columns:
            stats["clue_level_distribution"] = df["线索评级_试驾前"].value_counts().to_dict()

        return stats


class FeatureEngineer:
    """特征工程处理器

    设计说明：
    本类仅处理 AutoGluon 不擅长的预处理工作：
    1. 时间特征提取：AutoGluon 不会自动提取 day_of_week、hour 等衍生特征
    2. 数值类型转换：确保数值列类型正确

    不处理的内容（交给 AutoGluon 自动处理）：
    - 类别编码（AutoGluon 内置 CategoryFeatureGenerator）
    - 缺失值填充（AutoGluon 内置 FillNaFeatureGenerator）
    - 异常值处理（AutoGluon 的 QuantileTransformer 自动处理偏斜分布）
    """

    def __init__(
        self,
        time_columns: List[str],
        numeric_columns: Optional[List[str]] = None,
        create_interactions: bool = True,  # 是否创建交互特征
        interaction_context: Optional[dict] = None,
    ):
        """
        初始化特征工程处理器

        Args:
            time_columns: 时间列名列表
            numeric_columns: 数值列名列表（可选，用于类型转换）
            create_interactions: 是否创建交互特征（默认 True）
        """
        self.time_columns = time_columns
        self.numeric_columns = numeric_columns or []
        self.create_interactions = create_interactions
        self.interaction_context = deepcopy(interaction_context) if interaction_context else {}

    @staticmethod
    def _build_city_car_key(city: object, car_model: object) -> str:
        city_value = "未知" if pd.isna(city) else str(city)
        car_value = "未知" if pd.isna(car_model) else str(car_model)
        return f"{city_value}|||{car_value}"

    def fit(self, df: pd.DataFrame) -> dict:
        """
        基于训练集学习交互特征所需的上下文，避免在验证/推理阶段重新拟合。
        """
        context: Dict[str, Dict[str, int]] = {}

        if self.create_interactions and {"所在城市", "首触意向车型"}.issubset(df.columns):
            city_car_keys = df.apply(
                lambda row: self._build_city_car_key(row.get("所在城市"), row.get("首触意向车型")),
                axis=1,
            )
            city_car_heat = city_car_keys.value_counts().astype(int).to_dict()
            context["city_car_heat"] = city_car_heat

        self.interaction_context = context
        return deepcopy(context)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """训练阶段：先学习上下文，再转换数据。"""
        self.fit(df)
        return self.transform(df)

    def transform(
        self,
        df: pd.DataFrame,
        interaction_context: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        验证/推理阶段：复用训练期上下文进行特征转换。
        """
        if interaction_context is not None:
            self.interaction_context = deepcopy(interaction_context)

        df = df.copy()
        metadata = {}

        df, time_features = self._extract_time_features(df)
        metadata["time_features"] = time_features

        df = self._convert_numeric_types(df)

        if self.create_interactions:
            df, interaction_features = self._create_interaction_features(
                df,
                interaction_context=self.interaction_context,
            )
            metadata["interaction_features"] = interaction_features
            metadata["interaction_context"] = deepcopy(self.interaction_context)
            logger.info(f"交互特征创建完成: {interaction_features}")

        logger.info("特征工程完成，其余预处理由 AutoGluon 自动处理")
        return df, metadata

    def process(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        执行特征工程

        Args:
            df: 输入 DataFrame

        Returns:
            处理后的 DataFrame 和元数据
        """
        return self.fit_transform(df)

    def _extract_time_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        从时间列提取特征

        Args:
            df: 输入 DataFrame

        Returns:
            处理后的 DataFrame 和新增的特征名列表
        """
        new_features = []

        for col in self.time_columns:
            if col not in df.columns:
                continue

            # 转换为 datetime
            try:
                dt_col = pd.to_datetime(df[col], errors="coerce")

                # 提取时间特征
                df[f"{col}_day_of_week"] = dt_col.dt.dayofweek  # 0=周一, 6=周日
                df[f"{col}_hour"] = dt_col.dt.hour
                df[f"{col}_is_weekend"] = dt_col.dt.dayofweek.isin([5, 6]).astype(int)

                new_features.extend([
                    f"{col}_day_of_week",
                    f"{col}_hour",
                    f"{col}_is_weekend",
                ])

                # 保留原始时间列（OOT 切分等场景需要）
                # 不再删除: df = df.drop(columns=[col])

                logger.debug(f"从 {col} 提取时间特征: {new_features[-3:]}")

            except Exception as e:
                logger.warning(f"时间特征提取失败 {col}: {e}")

        return df, new_features

    def _convert_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换数值类型（确保类型正确，不填充缺失值）

        Args:
            df: 输入 DataFrame

        Returns:
            处理后的 DataFrame
        """
        for col in self.numeric_columns:
            if col not in df.columns:
                continue

            # 转换为数值类型（保留 NaN，让 AutoGluon 处理）
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _create_interaction_features(
        self,
        df: pd.DataFrame,
        interaction_context: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        创建交互特征

        交互特征能够捕捉特征之间的组合效应，提升模型区分能力：
        - 时间响应特征：首触响应时长（及时响应的线索质量更高）
        - 渠道组合特征：一级×二级渠道组合（不同渠道组合转化率不同）
        - 城市×车型热度：城市车型热度（热门组合可能更容易成交）
        - 通话质量特征：有效通话（区分无效骚扰与有效沟通）

        Args:
            df: 输入 DataFrame

        Returns:
            处理后的 DataFrame 和新增的特征名列表
        """
        df = df.copy()
        new_features = []

        # 1. 时间响应特征：首触响应时长
        if '首触时间' in df.columns and '线索创建时间' in df.columns:
            create_time = pd.to_datetime(df['线索创建时间'], errors='coerce')
            first_touch = pd.to_datetime(df['首触时间'], errors='coerce')

            df['首触响应时长_小时'] = (first_touch - create_time).dt.total_seconds() / 3600
            new_features.append('首触响应时长_小时')

            # 响应及时性分档（业务含义：即时<1h，快速<4h，正常<24h，延迟>=24h）
            df['响应及时性'] = pd.cut(
                df['首触响应时长_小时'].fillna(-1),
                bins=[-1, 0, 1, 4, 24, float('inf')],
                labels=['未知', '即时', '快速', '正常', '延迟']
            ).astype(str)
            new_features.append('响应及时性')

            logger.debug(f"创建时间响应特征: 首触响应时长_小时, 响应及时性")

        # 2. 渠道组合特征
        if '一级渠道名称' in df.columns and '二级渠道名称' in df.columns:
            df['渠道组合'] = (
                df['一级渠道名称'].fillna('未知').astype(str) + '_' +
                df['二级渠道名称'].fillna('未知').astype(str)
            )
            new_features.append('渠道组合')
            logger.debug(f"创建渠道组合特征: 渠道组合")

        # 3. 城市×车型热度
        if '所在城市' in df.columns and '首触意向车型' in df.columns:
            try:
                context = interaction_context or {}
                city_car_heat = context.get("city_car_heat", {})
                if not city_car_heat:
                    logger.warning("未检测到训练期城市车型热度上下文，回退为基于当前数据拟合")
                    city_car_keys = df.apply(
                        lambda row: self._build_city_car_key(row.get("所在城市"), row.get("首触意向车型")),
                        axis=1,
                    )
                    city_car_heat = city_car_keys.value_counts().astype(int).to_dict()
                    self.interaction_context["city_car_heat"] = city_car_heat

                df['城市车型热度'] = df.apply(
                    lambda row: city_car_heat.get(
                        self._build_city_car_key(row.get("所在城市"), row.get("首触意向车型")),
                        0,
                    ),
                    axis=1,
                )
                new_features.append('城市车型热度')
                logger.debug(f"创建城市车型热度特征: 城市车型热度")
            except Exception as e:
                logger.warning(f"城市车型热度特征创建失败: {e}")

        # 4. 通话质量特征
        if '通话次数' in df.columns and '通话总时长' in df.columns:
            # 派生平均通话时长，避免覆盖原始字段
            df['平均通话时长_派生'] = df['通话总时长'] / df['通话次数'].replace(0, np.nan)
            df['平均通话时长_派生'] = df['平均通话时长_派生'].fillna(0)
            new_features.append('平均通话时长_派生')

            # 有效通话：通话次数>0 且 通话总时长>60秒（排除无效骚扰）
            df['有效通话'] = ((df['通话次数'] > 0) & (df['通话总时长'] > 60)).astype(int)
            new_features.append('有效通话')

            logger.debug(f"创建通话质量特征: 平均通话时长_派生, 有效通话")

        return df, new_features


def split_data(
    df: pd.DataFrame,
    target_label: str,
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    划分训练集和测试集（随机切分）

    Args:
        df: 数据 DataFrame
        target_label: 目标变量名
        test_size: 测试集比例
        random_seed: 随机种子
        stratify: 是否分层采样

    Returns:
        训练集和测试集 DataFrame
    """
    logger.info(f"划分数据集: test_size={test_size}, stratify={stratify}")

    # 过滤掉目标变量为空的行
    valid_mask = df[target_label].notna()
    df_valid = df[valid_mask].copy()

    if len(df_valid) < len(df):
        logger.warning(f"过滤掉 {len(df) - len(df_valid)} 行目标变量为空的数据")

    # 分层采样
    stratify_col = df_valid[target_label] if stratify else None

    train_df, test_df = train_test_split(
        df_valid,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_col,
    )

    logger.info(f"训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")

    # 打印目标变量分布
    if stratify:
        train_dist = train_df[target_label].value_counts(normalize=True)
        test_dist = test_df[target_label].value_counts(normalize=True)
        logger.info(f"训练集目标分布:\n{train_dist}")
        logger.info(f"测试集目标分布:\n{test_dist}")

    return train_df, test_df


def split_data_oot(
    df: pd.DataFrame,
    target_label: str,
    time_column: str,
    cutoff_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    OOT（Out-of-Time）时间切分验证

    使用时间切分而非随机切分，更好地模拟真实预测场景：
    用历史数据训练，预测未来数据。

    Args:
        df: 数据 DataFrame
        target_label: 目标变量名
        time_column: 时间列名
        cutoff_date: 切分日期，格式如 '2026-02-01'
            - 此日期之前的数据作为训练集
            - 此日期及之后的数据作为测试集

    Returns:
        训练集和测试集 DataFrame

    Example:
        >>> train_df, test_df = split_data_oot(
        ...     df,
        ...     target_label="到店标签_14天",
        ...     time_column="线索创建时间",
        ...     cutoff_date="2026-02-01"
        ... )
    """
    logger.info(f"OOT 时间切分: time_column={time_column}, cutoff={cutoff_date}")

    # 转换时间列
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    # 过滤无效时间
    valid_time_mask = df[time_column].notna()
    if valid_time_mask.sum() < len(df):
        logger.warning(f"过滤掉 {(~valid_time_mask).sum()} 行时间无效的数据")
        df = df[valid_time_mask]

    # 过滤目标变量为空
    valid_label_mask = df[target_label].notna()
    if valid_label_mask.sum() < len(df):
        logger.warning(f"过滤掉 {(~valid_label_mask).sum()} 行目标变量为空的数据")
        df = df[valid_label_mask]

    # 时间切分
    cutoff = pd.Timestamp(cutoff_date)
    train_df = df[df[time_column] < cutoff].copy()
    test_df = df[df[time_column] >= cutoff].copy()

    logger.info(f"训练集: {len(train_df)} 行 (时间 < {cutoff_date})")
    logger.info(f"测试集: {len(test_df)} 行 (时间 >= {cutoff_date})")

    # 检查数据量是否合理
    if len(train_df) == 0:
        raise ValueError(f"训练集为空，请检查 cutoff_date 是否早于数据时间范围")
    if len(test_df) == 0:
        raise ValueError(f"测试集为空，请检查 cutoff_date 是否晚于数据时间范围")

    # 打印目标变量分布
    train_dist = train_df[target_label].value_counts(normalize=True)
    test_dist = test_df[target_label].value_counts(normalize=True)
    logger.info(f"训练集目标分布:\n{train_dist}")
    logger.info(f"测试集目标分布:\n{test_dist}")

    # 打印时间范围
    logger.info(f"训练集时间范围: {train_df[time_column].min()} ~ {train_df[time_column].max()}")
    logger.info(f"测试集时间范围: {test_df[time_column].min()} ~ {test_df[time_column].max()}")

    return train_df, test_df


def split_data_oot_three_way(
    df: pd.DataFrame,
    target_label: str,
    time_column: str,
    train_end: str,
    valid_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    OOT 三层时间切分（训练/验证/测试）

    更严谨的评估方式：用验证集调参，测试集评估最终效果。

    Args:
        df: 数据 DataFrame
        target_label: 目标变量名
        time_column: 时间列名
        train_end: 训练集截止日期（不包含），此日期前为训练集
        valid_end: 验证集截止日期（不包含），train_end 到 valid_end 为验证集

    Returns:
        训练集、验证集、测试集 DataFrame

    Example:
        >>> train_df, valid_df, test_df = split_data_oot_three_way(
        ...     df,
        ...     target_label="到店标签_7天",
        ...     time_column="线索创建时间",
        ...     train_end="2026-03-11",  # 3月11日前为训练集
        ...     valid_end="2026-03-16",  # 3月11-16日为验证集，16日后为测试集
        ... )
    """
    logger.info(f"OOT 三层切分: train_end={train_end}, valid_end={valid_end}")

    # 转换时间列
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    # 过滤无效数据
    valid_mask = df[time_column].notna() & df[target_label].notna()
    if valid_mask.sum() < len(df):
        logger.warning(f"过滤掉 {(~valid_mask).sum()} 行无效数据")
        df = df[valid_mask]

    # 时间切分
    train_cutoff = pd.Timestamp(train_end)
    valid_cutoff = pd.Timestamp(valid_end)

    train_df = df[df[time_column] < train_cutoff].copy()
    valid_df = df[(df[time_column] >= train_cutoff) & (df[time_column] < valid_cutoff)].copy()
    test_df = df[df[time_column] >= valid_cutoff].copy()

    logger.info(f"训练集: {len(train_df)} 行 (时间 < {train_end})")
    logger.info(f"验证集: {len(valid_df)} 行 ({train_end} <= 时间 < {valid_end})")
    logger.info(f"测试集: {len(test_df)} 行 (时间 >= {valid_end})")

    # 检查数据量
    if len(train_df) == 0:
        raise ValueError("训练集为空")
    if len(valid_df) == 0:
        raise ValueError("验证集为空")
    if len(test_df) == 0:
        raise ValueError("测试集为空")

    # 打印统计信息
    # 检测目标变量类型，区分二分类和多分类
    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_label])

    for name, subset in [("训练集", train_df), ("验证集", valid_df), ("测试集", test_df)]:
        dist = subset[target_label].value_counts(normalize=True)
        if is_numeric_target:
            # 二分类：计算正样本率
            positive_rate = subset[target_label].mean() * 100
            logger.info(f"{name}: 正样本率 {positive_rate:.2f}%, 分布: {dist.to_dict()}")
        else:
            # 多分类：只显示分布
            logger.info(f"{name}: 目标分布: {dist.to_dict()}")

    return train_df, valid_df, test_df


def smart_split_data(
    df: pd.DataFrame,
    target_label: str,
    time_column: str = "线索创建时间",
    min_oot_days: int = 14,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    智能数据切分（自适应 OOT 或 随机切分）

    自动探查数据的时间跨度：
    - 若跨度 >= min_oot_days：按照 70% 训练，15% 验证，15% 测试的时间比例自动进行 OOT 切分。
    - 若跨度 < min_oot_days：降级为随机切分（80%训练，20%测试），并在返回的元数据中记录测试集索引以防泄漏。

    Args:
        df: 数据 DataFrame
        target_label: 目标变量名
        time_column: 时间列名
        min_oot_days: 触发 OOT 的最少时间跨度（天数）
        random_seed: 随机种子

    Returns:
        训练集, 验证集(若降级则为空), 测试集, 元数据字典(包含切分模式和测试集标记)
    """
    logger.info(f"执行智能数据切分 (Smart Split)...")

    # 预处理时间列
    df = df.copy()
    # 为保证可追溯，确保索引唯一且重置
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    valid_mask = df[time_column].notna() & df[target_label].notna()
    df_valid = df[valid_mask].copy()

    if len(df_valid) == 0:
        raise ValueError("有效数据为空，无法进行切分")

    min_date = df_valid[time_column].min()
    max_date = df_valid[time_column].max()
    time_span = (max_date - min_date).days

    logger.info(f"数据时间范围: {min_date.date()} 至 {max_date.date()} (跨度: {time_span} 天)")

    split_metadata = {
        "time_span_days": time_span,
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()),
    }

    if time_span >= min_oot_days:
        logger.info(f"跨度满足 OOT 要求 (>= {min_oot_days} 天)，触发自动 OOT 切分")
        # 按 70% / 15% / 15% 的时间跨度划分
        total_seconds = (max_date - min_date).total_seconds()
        train_end_date = min_date + pd.Timedelta(seconds=total_seconds * 0.70)
        valid_end_date = min_date + pd.Timedelta(seconds=total_seconds * 0.85)

        # 转换为字符串日期（去除时分秒，按天切分）
        train_end_str = str(train_end_date.date())
        valid_end_str = str(valid_end_date.date())

        logger.info(f"自动计算切分点: train_end={train_end_str}, valid_end={valid_end_str}")

        train_df, valid_df, test_df = split_data_oot_three_way(
            df_valid, target_label, time_column, train_end_str, valid_end_str
        )

        split_metadata["mode"] = "oot"
        split_metadata["train_end"] = train_end_str
        split_metadata["valid_end"] = valid_end_str
        split_metadata["test_indices"] = [] # OOT 模式通过时间过滤，不需要记索引

    else:
        logger.warning(f"跨度不满足 OOT 要求 (< {min_oot_days} 天)，降级为随机切分 (Random Split)")
        
        train_df, test_df = split_data(
            df_valid, target_label, test_size=0.2, random_seed=random_seed, stratify=True
        )
        # 验证集为空
        valid_df = pd.DataFrame(columns=df_valid.columns)
        
        split_metadata["mode"] = "random"
        # 记录测试集的原始索引或唯一 ID，这里使用 dataframe 的 index (需保存下来)
        # 假设原始 df 的 index 具有业务唯一性，推荐记录 "线索唯一ID"
        id_col = "线索唯一ID"
        if id_col in test_df.columns:
            split_metadata["test_ids"] = test_df[id_col].tolist()
            split_metadata["id_column"] = id_col
            logger.info(f"已记录 {len(test_df)} 个测试集 {id_col} 用于后续防泄漏验证")
        else:
            split_metadata["test_indices"] = test_df.index.tolist()
            logger.warning(f"未找到 '{id_col}'，记录测试集物理行索引用于防泄漏验证（存在一定风险）")

    return train_df, valid_df, test_df, split_metadata


def prepare_features(
    df: pd.DataFrame,
    excluded_columns: List[str],
    target_label: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备特征和目标变量

    Args:
        df: 数据 DataFrame
        excluded_columns: 需要排除的列
        target_label: 目标变量名

    Returns:
        特征 DataFrame 和目标 Series
    """
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in excluded_columns]

    # 确保目标变量不在特征中
    if target_label in feature_cols:
        feature_cols.remove(target_label)

    X = df[feature_cols].copy()
    y = df[target_label].copy()

    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"目标变量分布:\n{y.value_counts()}")

    return X, y
