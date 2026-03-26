"""
AutoML 模型封装模块

封装模型训练、预测、评估等功能。
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _hook_params_without_self(method) -> list[inspect.Parameter]:
    return list(inspect.signature(method).parameters.values())[1:]


def _callback_hook_is_compatible(base_hook, callback_hook) -> bool:
    """检查 callback hook 是否兼容当前基类签名。"""
    base_params = _hook_params_without_self(base_hook)
    callback_params = _hook_params_without_self(callback_hook)
    callback_by_name = {param.name: param for param in callback_params}

    for base_param in base_params:
        callback_param = callback_by_name.get(base_param.name)
        if callback_param is None:
            return False
        if callback_param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return False

    return True


def _is_progress_callback_compatible(callback_cls) -> bool:
    """探测 progress callback 是否可安全注入。"""
    try:
        from autogluon.core.callbacks import AbstractCallback
    except Exception as exc:
        logger.debug("无法导入回调基类，跳过训练进度监控: %s", exc)
        return False

    try:
        before_ok = _callback_hook_is_compatible(
            AbstractCallback._before_model_fit,
            callback_cls._before_model_fit,
        )
        after_ok = _callback_hook_is_compatible(
            AbstractCallback._after_model_fit,
            callback_cls._after_model_fit,
        )
    except Exception as exc:
        logger.debug("检查训练进度回调兼容性失败: %s", exc)
        return False

    return before_ok and after_ok


class LeadScoringPredictor:
    """线索评级预测器"""

    def __init__(
        self,
        label: str,
        output_path: str,
        eval_metric: str = "roc_auc",
        problem_type: Optional[str] = None,
        sample_weight: Optional[str] = None,
        weight_evaluation: bool = False,
        # 资源控制参数
        max_memory_usage_ratio: Optional[float] = None,
        memory_limit_gb: Optional[float] = None,
        fit_strategy: Optional[str] = None,
        excluded_model_types: Optional[List[str]] = None,
        num_folds_parallel: Optional[int] = None,
    ):
        """
        初始化预测器

        Args:
            label: 目标变量名
            output_path: 模型保存路径
            eval_metric: 评估指标
            problem_type: 问题类型 ('binary', 'multiclass', 'regression')
            sample_weight: 样本权重配置
                - None: 不使用权重
                - 'balance_weight': 自动平衡类别权重（推荐用于不平衡数据）
                - 'auto_weight': 自动权重策略
                - str: 指定权重列名
            weight_evaluation: 是否在评估时使用样本权重
            max_memory_usage_ratio: 模型级最大内存使用比例（不推荐默认启用）
            memory_limit_gb: AutoML 总内存软限制（GB）
            fit_strategy: 模型级训练策略（sequential/parallel）
            excluded_model_types: 排除的模型类型列表（内存密集型：KNN, RF, XT）
            num_folds_parallel: 并行训练的 fold 数量（None=自动，低内存机器建议 1）
        """
        self.label = label
        self.output_path = Path(output_path)
        self.eval_metric = eval_metric
        self.problem_type = problem_type
        self.sample_weight = sample_weight
        self.weight_evaluation = weight_evaluation
        self.max_memory_usage_ratio = max_memory_usage_ratio
        self.memory_limit_gb = memory_limit_gb
        self.fit_strategy = fit_strategy
        self.excluded_model_types = excluded_model_types
        self.num_folds_parallel = num_folds_parallel

        self._predictor = None
        self._feature_metadata = None
        self._feature_columns: Optional[List[str]] = None
        self._extra_metadata: Dict[str, Any] = {}

    @staticmethod
    def _format_column_preview(columns: List[str], limit: int = 5) -> str:
        """格式化列名预览。"""
        preview = columns[:limit]
        suffix = "..." if len(columns) > limit else ""
        return ", ".join(preview) + suffix

    def _align_frame_to_features(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        feature_columns: List[str],
        include_label: bool,
    ) -> pd.DataFrame:
        """
        将数据对齐到指定特征列集合。

        Args:
            data: 待对齐数据
            dataset_name: 数据集名称，用于日志和报错
            feature_columns: 训练期特征列顺序
            include_label: 是否要求并保留目标列

        Returns:
            对齐后的 DataFrame
        """
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            preview = self._format_column_preview(missing_features)
            raise ValueError(
                f"{dataset_name} 缺少 {len(missing_features)} 个训练特征列: {preview}"
            )

        ordered_columns = feature_columns.copy()
        if include_label:
            if self.label not in data.columns:
                raise ValueError(f"{dataset_name} 缺少目标列: {self.label}")
            ordered_columns.append(self.label)

        allowed_columns = set(ordered_columns)
        extra_columns = [col for col in data.columns if col not in allowed_columns]
        if extra_columns:
            logger.info(
                f"{dataset_name} 丢弃 {len(extra_columns)} 个额外列: "
                f"{self._format_column_preview(extra_columns)}"
            )

        return data.loc[:, ordered_columns].copy()

    def _require_feature_columns(self) -> List[str]:
        """确保模型已记录训练特征列。"""
        if not self._feature_columns:
            raise ValueError("模型未记录训练特征列，请重新训练或加载包含 metadata 的模型")
        return self._feature_columns

    def _prepare_inference_data(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        include_label: bool = False,
    ) -> pd.DataFrame:
        """基于训练期特征列对齐推理/评估数据。"""
        feature_columns = self._require_feature_columns()
        return self._align_frame_to_features(
            data=data,
            dataset_name=dataset_name,
            feature_columns=feature_columns,
            include_label=include_label,
        )

    def _configure_bagging_holdout(self, fit_kwargs: Dict[str, Any]) -> None:
        """
        兼容 AutoGluon 1.5 对 bagging + tuning_data 的约束。

        当启用 bagging 且传入 tuning_data 时，AutoGluon 1.5 需要显式
        设置 use_bag_holdout=True，否则会在 fit 时直接报错。
        """
        tuning_data = fit_kwargs.get("tuning_data")
        has_tuning_data = isinstance(tuning_data, pd.DataFrame) and len(tuning_data) > 0

        num_bag_folds = fit_kwargs.get("num_bag_folds")
        has_bagging = isinstance(num_bag_folds, int) and num_bag_folds > 0

        if not (has_tuning_data and has_bagging):
            return

        use_bag_holdout = fit_kwargs.get("use_bag_holdout")
        if use_bag_holdout is False:
            raise ValueError(
                "检测到 tuning_data 与 num_bag_folds>0 同时启用，但 use_bag_holdout=False。"
                "AutoGluon 1.5 下 bagging 模式不能直接使用 tuning_data，"
                "请改为 use_bag_holdout=True。"
            )

        if use_bag_holdout is None:
            fit_kwargs["use_bag_holdout"] = True
            logger.info(
                "检测到 bagging + tuning_data，已自动启用 use_bag_holdout=True "
                "以兼容 AutoGluon 1.5"
            )

    def train(
        self,
        train_data: pd.DataFrame,
        presets: str = "high_quality",
        time_limit: int = 3600,
        excluded_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "LeadScoringPredictor":
        """
        训练模型

        Args:
            train_data: 训练数据
            presets: 模型预设 ('best_quality', 'high_quality', 'good_quality', 'medium_quality')
            time_limit: 训练时间限制（秒）
            excluded_columns: 需要排除的列
            **kwargs: 其他训练参数

        Returns:
            self
        """
        from autogluon.tabular import TabularPredictor

        logger.info(f"开始训练模型: label={self.label}, presets={presets}")
        logger.info(f"训练数据: {len(train_data)} 行")

        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.label not in train_data.columns:
            raise ValueError(f"train_data 缺少目标列: {self.label}")

        # 统一确定训练期特征列集合，并对训练/验证数据执行同一套对齐。
        excluded_set = set(excluded_columns or [])
        self._feature_columns = [
            col for col in train_data.columns
            if col not in excluded_set and col != self.label
        ]
        train_data = self._align_frame_to_features(
            data=train_data,
            dataset_name="train_data",
            feature_columns=self._feature_columns,
            include_label=True,
        )

        logger.info(f"最终训练特征数: {len(self._feature_columns)}")

        # 初始化 predictor（包含样本权重配置）
        init_kwargs = {
            "label": self.label,
            "eval_metric": self.eval_metric,
            "path": str(self.output_path),
            "problem_type": self.problem_type,
        }
        if self.sample_weight:
            init_kwargs["sample_weight"] = self.sample_weight
            init_kwargs["weight_evaluation"] = self.weight_evaluation
            logger.info(f"启用样本权重: {self.sample_weight}")

        self._predictor = TabularPredictor(**init_kwargs)

        # 训练参数
        fit_kwargs = {
            "presets": presets,
            "time_limit": time_limit,
            "verbosity": 2,
            **kwargs,
        }

        # 资源控制参数
        if self.memory_limit_gb is not None:
            fit_kwargs["memory_limit"] = self.memory_limit_gb
            logger.info(f"总内存软限制: {self.memory_limit_gb} GB")

        if self.fit_strategy:
            fit_kwargs["fit_strategy"] = self.fit_strategy
            logger.info(f"模型训练策略: {self.fit_strategy}")

        if self.max_memory_usage_ratio is not None:
            ag_args_fit = dict(fit_kwargs.get("ag_args_fit") or {})
            ag_args_fit["ag.max_memory_usage_ratio"] = self.max_memory_usage_ratio
            fit_kwargs["ag_args_fit"] = ag_args_fit
            logger.info(f"内存使用比例限制: {self.max_memory_usage_ratio}")

        # 排除内存密集型模型
        if self.excluded_model_types:
            fit_kwargs["excluded_model_types"] = self.excluded_model_types
            logger.info(f"排除模型类型: {self.excluded_model_types}")

        # 限制并行 fold 数量
        if self.num_folds_parallel is not None:
            ag_args_ensemble = dict(fit_kwargs.get("ag_args_ensemble") or {})
            ag_args_ensemble["num_folds_parallel"] = self.num_folds_parallel
            fit_kwargs["ag_args_ensemble"] = ag_args_ensemble
            logger.info(f"并行 fold 数量限制: {self.num_folds_parallel}")

        for dataset_name, include_label in (
            ("tuning_data", True),
            ("test_data", True),
            ("unlabeled_data", False),
        ):
            dataset = fit_kwargs.get(dataset_name)
            if isinstance(dataset, pd.DataFrame):
                fit_kwargs[dataset_name] = self._align_frame_to_features(
                    data=dataset,
                    dataset_name=dataset_name,
                    feature_columns=self._feature_columns,
                    include_label=include_label,
                )
                logger.info(
                    f"{dataset_name} 对齐后特征数: "
                    f"{len(fit_kwargs[dataset_name].columns) - (1 if include_label else 0)}"
                )

        self._configure_bagging_holdout(fit_kwargs)

        # 添加进度监控回调
        callbacks = fit_kwargs.pop("callbacks", None)
        if callbacks is None:
            callbacks = []
        else:
            callbacks = list(callbacks)
        if not any(hasattr(cb, '__class__') and 'Progress' in cb.__class__.__name__ for cb in callbacks):
            try:
                from src.training.progress_callback import TrainingProgressCallback
                if _is_progress_callback_compatible(TrainingProgressCallback):
                    progress_callback = TrainingProgressCallback(time_limit=time_limit)
                    callbacks.append(progress_callback)
                    logger.info("已启用训练进度监控")
                else:
                    logger.warning("训练进度回调与当前 AutoGluon 版本不兼容，已跳过进度监控")
            except Exception as exc:
                logger.warning("初始化训练进度回调失败，已跳过进度监控: %s", exc)

        fit_kwargs["callbacks"] = callbacks

        # 训练
        self._predictor.fit(train_data, **fit_kwargs)

        logger.info("模型训练完成")

        # 输出训练摘要
        for cb in callbacks:
            if hasattr(cb, 'get_summary'):
                summary = cb.get_summary()
                if summary.get("best_model"):
                    logger.info(
                        f"训练摘要: 共训练 {summary['total_models_trained']} 个模型, "
                        f"最佳模型: {summary['best_model']}, "
                        f"分数: {summary['best_score']:.4f}, "
                        f"总耗时: {summary['total_time_formatted']}"
                    )

        return self

    def predict(self, data: pd.DataFrame, model: Optional[str] = None) -> np.ndarray:
        """
        预测类别

        Args:
            data: 输入数据

        Returns:
            预测结果
        """
        if self._predictor is None:
            raise ValueError("模型未训练，请先调用 train() 或 load()")

        aligned_data = self._prepare_inference_data(
            data=data,
            dataset_name="predict_data",
            include_label=False,
        )
        predict_kwargs = {}
        if model:
            predict_kwargs["model"] = model
        return self._predictor.predict(aligned_data, **predict_kwargs)

    def predict_proba(self, data: pd.DataFrame, model: Optional[str] = None) -> pd.DataFrame:
        """
        预测概率

        Args:
            data: 输入数据

        Returns:
            预测概率 DataFrame
        """
        if self._predictor is None:
            raise ValueError("模型未训练，请先调用 train() 或 load()")

        aligned_data = self._prepare_inference_data(
            data=data,
            dataset_name="predict_data",
            include_label=False,
        )
        predict_kwargs = {}
        if model:
            predict_kwargs["model"] = model
        return self._predictor.predict_proba(aligned_data, **predict_kwargs)

    def get_positive_proba(self, data: pd.DataFrame, model: Optional[str] = None) -> np.ndarray:
        """
        获取正类概率（用于二分类 Top-K 排序）

        Args:
            data: 输入数据

        Returns:
            正类概率数组
        """
        proba = self.predict_proba(data, model=model)

        if self.problem_type == "multiclass" or proba.shape[1] > 2:
            # 多分类，返回最大概率
            return proba.max(axis=1).values
        else:
            # 二分类，使用 positive_class 属性确定正类位置
            positive_class = self._predictor.positive_class
            return proba[positive_class].values

    def get_class_proba(
        self,
        data: pd.DataFrame,
        target_class: str,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """
        获取指定类别的预测概率（用于多分类 Top-K 排序）

        Args:
            data: 输入数据
            target_class: 目标类别名
            model: 指定模型名（可选）

        Returns:
            指定类别的预测概率数组
        """
        proba = self.predict_proba(data, model=model)
        target_class_str = str(target_class)

        if target_class in proba.columns:
            return proba[target_class].values

        column_mapping = {str(column): column for column in proba.columns}
        if target_class_str in column_mapping:
            return proba[column_mapping[target_class_str]].values

        available_classes = ", ".join(str(column) for column in proba.columns)
        raise ValueError(
            f"目标类别 {target_class_str} 不存在于预测概率输出中，可用类别: {available_classes}"
        )

    def evaluate(
        self, test_data: pd.DataFrame, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            test_data: 测试数据
            metrics: 额外评估指标

        Returns:
            评估指标字典
        """
        if self._predictor is None:
            raise ValueError("模型未训练，请先调用 train() 或 load()")

        logger.info("评估模型...")
        aligned_test_data = self._prepare_inference_data(
            data=test_data,
            dataset_name="test_data",
            include_label=True,
        )

        # 使用框架内置评估
        eval_result = self._predictor.evaluate(aligned_test_data, silent=True)

        # 计算额外指标
        y_true = aligned_test_data[self.label].values
        y_pred = self.predict(aligned_test_data)

        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        extra_metrics = {}

        try:
            # 二分类指标
            if self.problem_type != "multiclass":
                y_proba = self.get_positive_proba(aligned_test_data)

                extra_metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                extra_metrics["precision"] = precision_score(
                    y_true, y_pred, average="binary", zero_division=0
                )
                extra_metrics["recall"] = recall_score(
                    y_true, y_pred, average="binary", zero_division=0
                )
                extra_metrics["f1"] = f1_score(
                    y_true, y_pred, average="binary", zero_division=0
                )
            else:
                extra_metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
                extra_metrics["macro_f1"] = f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
                extra_metrics["weighted_f1"] = f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
                extra_metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

            extra_metrics["accuracy"] = accuracy_score(y_true, y_pred)

        except Exception as e:
            logger.warning(f"计算额外指标失败: {e}")

        # 合并结果
        result = {**eval_result, **extra_metrics}

        logger.info(f"评估结果: {result}")

        return result

    def get_leaderboard(
        self, test_data: Optional[pd.DataFrame] = None, silent: bool = False
    ) -> pd.DataFrame:
        """
        获取模型排行榜

        Args:
            test_data: 测试数据（可选，用于额外评估）
            silent: 是否静默模式

        Returns:
            排行榜 DataFrame
        """
        if self._predictor is None:
            raise ValueError("模型未训练，请先调用 train() 或 load()")

        aligned_test_data = None
        if test_data is not None:
            aligned_test_data = self._prepare_inference_data(
                data=test_data,
                dataset_name="leaderboard_data",
                include_label=True,
            )

        return self._predictor.leaderboard(aligned_test_data, silent=silent)

    def get_model_names(self) -> List[str]:
        """返回当前 predictor 持有的全部模型名。"""
        if self._predictor is None:
            raise ValueError("模型未训练，请先调用 train() 或 load()")
        return list(self._predictor.model_names())

    def get_feature_importance(
        self,
        test_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            test_data: 测试数据
            feature_names: 特征名列表（可选）

        Returns:
            特征重要性 DataFrame
        """
        if self._predictor is None:
            raise ValueError("模型未训练，请先调用 train() 或 load()")

        logger.info("计算特征重要性...")
        aligned_test_data = self._prepare_inference_data(
            data=test_data,
            dataset_name="test_data",
            include_label=True,
        )

        importance = self._predictor.feature_importance(aligned_test_data)

        if isinstance(importance, pd.Series):
            importance_df = importance.rename("importance").to_frame().reset_index()
        elif isinstance(importance, pd.DataFrame):
            importance_df = importance.copy().reset_index()
            if "importance" not in importance_df.columns:
                non_feature_columns = [col for col in importance_df.columns if col != "feature"]
                if len(non_feature_columns) == 1:
                    importance_df = importance_df.rename(
                        columns={non_feature_columns[0]: "importance"}
                    )
                else:
                    raise ValueError(
                        "feature_importance 返回 DataFrame，但未找到 importance 列"
                    )
        else:
            raise TypeError(
                "feature_importance 返回了不支持的类型: "
                f"{type(importance).__name__}"
            )

        if "feature" not in importance_df.columns:
            if "index" in importance_df.columns:
                importance_df = importance_df.rename(columns={"index": "feature"})
            else:
                importance_df = importance_df.rename(
                    columns={importance_df.columns[0]: "feature"}
                )

        ordered_columns = ["feature", "importance"] + [
            col for col in importance_df.columns
            if col not in {"feature", "importance"}
        ]
        importance_df = (
            importance_df.loc[:, ordered_columns]
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return importance_df

    def save(self, path: Optional[str] = None, extra_metadata: Optional[Dict[str, Any]] = None):
        """
        保存模型（框架自动保存，此方法保存额外元数据）

        Args:
            path: 保存路径（可选）
        """
        save_path = Path(path) if path else self.output_path

        # 保存元数据
        metadata = {
            "label": self.label,
            "eval_metric": self.eval_metric,
            "problem_type": self.problem_type,
            "output_path": str(save_path),
            "sample_weight": self.sample_weight,
            "weight_evaluation": self.weight_evaluation,
            "feature_columns": self._feature_columns or [],
        }
        merged_extra_metadata = extra_metadata or self._extra_metadata
        if merged_extra_metadata:
            metadata["extra_metadata"] = merged_extra_metadata

        metadata_path = save_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"模型元数据已保存: {metadata_path}")

    @classmethod
    def load(cls, path: str) -> "LeadScoringPredictor":
        """
        加载模型

        Args:
            path: 模型路径

        Returns:
            加载的预测器实例
        """
        from autogluon.tabular import TabularPredictor

        load_path = Path(path)

        # 加载元数据
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            # 从底层 predictor 推断
            metadata = {}

        # 加载底层 predictor（跳过版本检查以避免兼容性问题）
        predictor = TabularPredictor.load(str(load_path), require_version_match=False)

        feature_columns = metadata.get("feature_columns")
        if not feature_columns:
            feature_metadata = getattr(predictor, "feature_metadata_in", None)
            if feature_metadata is not None and hasattr(feature_metadata, "get_features"):
                feature_columns = list(feature_metadata.get_features())

        # 创建实例
        instance = cls(
            label=metadata.get("label", predictor.label),
            output_path=str(load_path),
            eval_metric=metadata.get("eval_metric", predictor.eval_metric),
            problem_type=metadata.get("problem_type", predictor.problem_type),
            sample_weight=metadata.get("sample_weight"),
            weight_evaluation=metadata.get("weight_evaluation", False),
        )
        instance._predictor = predictor
        instance._feature_columns = feature_columns or None
        instance._extra_metadata = metadata.get("extra_metadata", {})

        logger.info(f"模型已加载: {load_path}")
        if instance.sample_weight:
            logger.info(f"样本权重配置: {instance.sample_weight}")
        if instance._feature_columns:
            logger.info(f"已恢复 {len(instance._feature_columns)} 个训练特征列")

        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        if self._predictor is None:
            return {"status": "not_trained"}

        info = {
            "label": self.label,
            "eval_metric": self.eval_metric,
            "problem_type": self.problem_type,
            "output_path": str(self.output_path),
            "sample_weight": self.sample_weight,
            "weight_evaluation": self.weight_evaluation,
            "feature_count": len(self._feature_columns or []),
            "model_types": list(self._predictor.model_names()),
            "best_model": self._predictor.model_best,
        }

        return info

    def cleanup(
        self,
        keep_best_only: bool = True,
        dry_run: bool = False,
        keep_model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        清理模型文件释放磁盘空间

        Args:
            keep_best_only: 是否只保留最佳模型
            dry_run: 仅模拟运行，不实际删除
            keep_model_names: 显式指定需要保留的模型列表

        Returns:
            清理结果信息
        """
        if self._predictor is None:
            logger.warning("模型未训练，无需清理")
            return {"status": "skipped", "reason": "not_trained"}

        import shutil

        result = {
            "dry_run": dry_run,
            "models_before": list(self._predictor.model_names()),
            "best_model": self._predictor.model_best,
        }

        # 获取清理前目录大小
        model_path = self.output_path
        if model_path.exists():
            before_size = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            )
            result["size_before_mb"] = round(before_size / (1024**2), 2)
        else:
            result["size_before_mb"] = 0

        if not dry_run:
            try:
                # 删除非最佳模型
                if keep_model_names:
                    resolved_keep_models = [
                        model_name for model_name in keep_model_names
                        if model_name in self._predictor.model_names()
                    ]
                    logger.info(f"删除非保留模型以释放空间，保留: {resolved_keep_models}")
                    self._predictor.delete_models(models_to_keep=resolved_keep_models, dry_run=False)
                elif keep_best_only:
                    logger.info("删除非最佳模型以释放空间...")
                    self._predictor.delete_models(models_to_keep="best", dry_run=False)

                # 保存空间（清理缓存等）
                self._predictor.save_space()

                # 获取清理后目录大小
                if model_path.exists():
                    after_size = sum(
                        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                    )
                    result["size_after_mb"] = round(after_size / (1024**2), 2)
                    result["freed_mb"] = round((before_size - after_size) / (1024**2), 2)

                result["models_after"] = list(self._predictor.model_names())
                result["status"] = "success"

                logger.info(f"清理完成，释放 {result.get('freed_mb', 0)} MB")

            except Exception as e:
                logger.error(f"清理失败: {e}")
                result["status"] = "error"
                result["error"] = str(e)
        else:
            result["status"] = "dry_run"

        return result

    def get_disk_usage(self) -> Dict[str, Any]:
        """
        获取模型目录磁盘占用

        Returns:
            磁盘占用信息
        """
        model_path = self.output_path

        if not model_path.exists():
            return {"exists": False, "path": str(model_path)}

        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        file_count = sum(1 for f in model_path.rglob("*") if f.is_file())

        return {
            "exists": True,
            "path": str(model_path),
            "total_size_mb": round(total_size / (1024**2), 2),
            "total_size_gb": round(total_size / (1024**3), 3),
            "file_count": file_count,
            "model_count": len(list(self._predictor.model_names())) if self._predictor else 0,
        }


def train_arrive_model(
    train_data: pd.DataFrame,
    label: str = "label_arrive_14d",
    output_path: str = "./outputs/models/arrive_model",
    presets: str = "high_quality",
    time_limit: int = 3600,
    sample_weight: Optional[str] = "balance_weight",
) -> LeadScoringPredictor:
    """
    训练到店预测模型的便捷函数

    Args:
        train_data: 训练数据
        label: 目标变量
        output_path: 模型保存路径
        presets: 模型预设
        time_limit: 训练时间限制
        sample_weight: 样本权重配置，默认 'balance_weight' 自动平衡类别

    Returns:
        训练好的预测器
    """
    predictor = LeadScoringPredictor(
        label=label,
        output_path=output_path,
        eval_metric="roc_auc",
        problem_type="binary",
        sample_weight=sample_weight,
        weight_evaluation=True,
    )

    predictor.train(
        train_data=train_data,
        presets=presets,
        time_limit=time_limit,
    )

    predictor.save()

    return predictor


def train_ohab_model(
    train_data: pd.DataFrame,
    label: str = "clue_level",
    output_path: str = "./outputs/models/ohab_model",
    presets: str = "high_quality",
    time_limit: int = 3600,
    sample_weight: Optional[str] = "balance_weight",
) -> LeadScoringPredictor:
    """
    训练 OHAB 评级模型的便捷函数

    Args:
        train_data: 训练数据
        label: 目标变量
        output_path: 模型保存路径
        presets: 模型预设
        time_limit: 训练时间限制
        sample_weight: 样本权重配置，默认 'balance_weight' 自动平衡类别

    Returns:
        训练好的预测器
    """
    predictor = LeadScoringPredictor(
        label=label,
        output_path=output_path,
        eval_metric="accuracy",
        problem_type="multiclass",
        sample_weight=sample_weight,
        weight_evaluation=True,
    )

    predictor.train(
        train_data=train_data,
        presets=presets,
        time_limit=time_limit,
    )

    predictor.save()

    return predictor
