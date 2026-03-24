"""
AutoGluon 模型封装模块

封装 AutoGluon TabularPredictor 的训练、预测、评估等功能。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
        """
        self.label = label
        self.output_path = Path(output_path)
        self.eval_metric = eval_metric
        self.problem_type = problem_type
        self.sample_weight = sample_weight
        self.weight_evaluation = weight_evaluation

        self._predictor = None
        self._feature_metadata = None
        self._feature_columns: Optional[List[str]] = None

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
            presets: AutoGluon 预设 ('best_quality', 'high_quality', 'good_quality', 'medium_quality')
            time_limit: 训练时间限制（秒）
            excluded_columns: 需要排除的列
            **kwargs: 其他 AutoGluon 参数

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

        # 训练
        self._predictor.fit(train_data, **fit_kwargs)

        logger.info("模型训练完成")

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
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
        return self._predictor.predict(aligned_data)

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
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
        return self._predictor.predict_proba(aligned_data)

    def get_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        获取正类概率（用于二分类 Top-K 排序）

        Args:
            data: 输入数据

        Returns:
            正类概率数组
        """
        proba = self.predict_proba(data)

        if self.problem_type == "multiclass" or proba.shape[1] > 2:
            # 多分类，返回最大概率
            return proba.max(axis=1).values
        else:
            # 二分类，使用 positive_class 属性确定正类位置
            positive_class = self._predictor.positive_class
            return proba[positive_class].values

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

        # 使用 AutoGluon 内置评估
        eval_result = self._predictor.evaluate(aligned_test_data, silent=True)

        # 计算额外指标
        y_true = aligned_test_data[self.label].values
        y_pred = self.predict(aligned_test_data)

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
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

        return self._predictor.leaderboard(test_data, silent=silent)

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

        # 转换为 DataFrame
        importance_df = pd.DataFrame({
            "feature": importance.index,
            "importance": importance.values,
        }).sort_values("importance", ascending=False)

        return importance_df

    def save(self, path: Optional[str] = None):
        """
        保存模型（AutoGluon 自动保存，此方法保存额外元数据）

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
            # 从 AutoGluon predictor 推断
            metadata = {}

        # 加载 AutoGluon predictor（跳过版本检查以避免兼容性问题）
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

    def cleanup(self, keep_best_only: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """
        清理模型文件释放磁盘空间

        Args:
            keep_best_only: 是否只保留最佳模型
            dry_run: 仅模拟运行，不实际删除

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
                if keep_best_only:
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
        presets: AutoGluon 预设
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
        presets: AutoGluon 预设
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
