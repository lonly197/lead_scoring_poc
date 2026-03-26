"""
训练进度监控回调

提供实时训练进度输出，包括：
- 已完成/总模型数
- 当前最佳模型和分数
- 预估剩余时间
- 友好的进度条显示
"""

import logging
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from autogluon.core.callbacks import AbstractCallback

logger = logging.getLogger(__name__)


class TrainingProgressCallback(AbstractCallback):
    """
    训练进度监控回调

    用法:
        from src.training.progress_callback import TrainingProgressCallback

        predictor.fit(
            train_data,
            callbacks=[TrainingProgressCallback()],
        )
    """

    def __init__(
        self,
        show_progress_bar: bool = True,
        log_interval: int = 1,
        time_limit: Optional[float] = None,
    ):
        """
        初始化进度回调

        Args:
            show_progress_bar: 是否显示进度条
            log_interval: 日志输出间隔（每 N 个模型）
            time_limit: 训练时间限制（秒）
        """
        super().__init__()
        self.show_progress_bar = show_progress_bar
        self.log_interval = log_interval
        self.time_limit = time_limit

        # 状态跟踪
        self.models_completed: List[str] = []
        self.model_scores: Dict[str, float] = {}
        self.current_model: Optional[str] = None
        self.train_start_time: Optional[float] = None
        self.model_start_time: Optional[float] = None
        self.total_models_estimate: Optional[int] = None
        self.trainer = None
        self.started_models: List[str] = []
        self.planned_family_counts: Counter[str] = Counter()
        self.completed_family_counts: Counter[str] = Counter()
        self.family_fit_times: dict[str, list[float]] = defaultdict(list)

        # 预设模型列表（根据 preset 推断）
        self._preset_models = {
            "best_quality": 30,    # 估算值
            "high_quality": 20,
            "good_quality": 12,    # L1(5) + L1_ensemble(1) + L2(5) + L2_ensemble(1)
            "medium_quality": 8,
        }
        self._family_time_priors = {
            "lightgbm": 20.0,
            "catboost": 180.0,
            "xgboost": 1800.0,
            "weightedensemble": 5.0,
            "nn_torch": 600.0,
            "fastai": 900.0,
            "rf": 120.0,
            "xt": 120.0,
            "knn": 60.0,
            "other": 60.0,
        }

    def before_trainer_fit(self, trainer, **kwargs):
        """训练开始前初始化状态。"""
        self.trainer = trainer
        self.models_completed = []
        self.model_scores = {}
        self.current_model = None
        self.train_start_time = time.time()
        self.model_start_time = None
        self.started_models = []
        self.completed_family_counts = Counter()
        self.family_fit_times = defaultdict(list)
        self.planned_family_counts = self._estimate_planned_family_counts(
            trainer=trainer,
            trainer_kwargs=kwargs,
        )
        self.total_models_estimate = max(
            sum(self.planned_family_counts.values()),
            self._estimate_total_models(trainer=trainer, trainer_kwargs=kwargs),
        )

    def _before_model_fit(
        self,
        trainer,
        model,
        time_limit: Optional[float] = None,
        stack_name: str = "core",
        level: int = 1,
    ) -> tuple[bool, bool]:
        """模型训练前调用。"""
        model_name = getattr(model, "name", str(model))
        self.current_model = model_name
        self.model_start_time = time.time()
        if model_name not in self.started_models:
            self.started_models.append(model_name)

        if self.train_start_time is None:
            self.train_start_time = time.time()

        self._ensure_capacity_for_model(model_name)

        # 计算进度
        total_models = self.total_models_estimate or self._estimate_total_models(trainer=trainer)
        completed = len(self.models_completed)
        progress_pct = (completed / total_models * 100) if total_models > 0 else 0

        # 计算已用时间
        elapsed = time.time() - self.train_start_time if self.train_start_time else 0
        elapsed_str = self._format_duration(elapsed)

        # 计算预估剩余时间
        estimated_remaining = self._estimate_remaining_seconds(trainer=trainer)
        if estimated_remaining is not None:
            remaining_str = self._format_duration(estimated_remaining)
        elif time_limit is not None:
            remaining_str = self._format_duration(max(float(time_limit), 0.0))
        else:
            remaining_str = "计算中..."

        # 输出进度信息
        self._print_progress(
            current_model=model_name,
            completed=completed,
            total=total_models,
            progress_pct=progress_pct,
            elapsed=elapsed_str,
            remaining=remaining_str,
        )
        return False, False

    def _after_model_fit(
        self,
        trainer,
        model_names: list[str],
        stack_name: str = "core",
        level: int = 1,
    ) -> bool:
        """模型训练后调用。"""
        if not model_names:
            self._print_model_skipped(stack_name=stack_name, level=level)
            return False

        for model_name in model_names:
            self.models_completed.append(model_name)
            family = self._normalize_family_from_model_name(model_name)
            self.completed_family_counts[family] += 1
            score = self._get_model_attribute(trainer, model_name, "val_score")
            fit_time = self._get_model_attribute(trainer, model_name, "fit_time")
            pred_time = self._get_model_attribute(trainer, model_name, "predict_time")

            if score is not None:
                self.model_scores[model_name] = score

            # 若 trainer 未提供 fit_time，退回到本地计时。
            model_time = 0
            if self.model_start_time:
                model_time = time.time() - self.model_start_time
            if fit_time is not None:
                self.family_fit_times[family].append(float(fit_time))
            elif model_time > 0:
                self.family_fit_times[family].append(float(model_time))

            self._print_model_complete(
                model_name=model_name,
                score=score,
                fit_time=fit_time if fit_time is not None else model_time,
                pred_time=pred_time,
            )

        if len(self.models_completed) % self.log_interval == 0:
            self._print_best_model()

        return False

    def _get_model_attribute(self, trainer, model_name: str, attribute: str) -> Optional[float]:
        """尽量从 trainer 或 model artifact 中读取模型属性。"""
        if hasattr(trainer, "get_model_attribute"):
            try:
                value = trainer.get_model_attribute(model_name, attribute, default=None)
            except TypeError:
                try:
                    value = trainer.get_model_attribute(model_name, attribute)
                except Exception:
                    value = None
            except Exception:
                value = None
            if value is not None:
                return value

        if hasattr(trainer, "load_model"):
            try:
                model = trainer.load_model(model_name)
            except Exception:
                return None
            return getattr(model, attribute, None)

        return None

    def _estimate_total_models(self, trainer=None, trainer_kwargs: Optional[Dict[str, Any]] = None) -> int:
        """估算总模型数"""
        planned_family_counts = self._estimate_planned_family_counts(
            trainer=trainer,
            trainer_kwargs=trainer_kwargs,
        )
        if planned_family_counts:
            return sum(planned_family_counts.values())

        # 默认返回 good_quality 的估算值
        return self._preset_models.get("good_quality", 12)

    def _estimate_planned_family_counts(
        self,
        trainer=None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Counter[str]:
        hp = None
        if trainer_kwargs:
            candidate = trainer_kwargs.get("hyperparameters")
            if isinstance(candidate, dict):
                hp = candidate
        if hp is None and trainer is not None:
            candidate = getattr(trainer, "hyperparameters", None)
            if isinstance(candidate, dict):
                hp = candidate

        if not hp:
            return Counter()

        level_start = trainer_kwargs.get("level_start") if trainer_kwargs else None
        level_end = trainer_kwargs.get("level_end") if trainer_kwargs else None
        level_count = None
        if isinstance(level_start, int) and isinstance(level_end, int) and level_end >= level_start:
            level_count = level_end - level_start + 1

        return self._count_planned_families(hp, level_count=level_count)

    def _count_planned_families(self, hp: dict[str, Any], level_count: Optional[int] = None) -> Counter[str]:
        if not isinstance(hp, dict) or not hp:
            return Counter()

        if all(isinstance(key, int) for key in hp):
            counts = Counter()
            for level_key in sorted(hp):
                counts.update(self._count_planned_families(hp[level_key]))
                counts["weightedensemble"] += 1
            return counts

        counts = Counter()
        for family, config in hp.items():
            counts[self._normalize_family_from_hyperparameter(family)] += self._count_model_configs(config)

        if level_count and level_count > 1:
            multiplied = Counter({family: count * level_count for family, count in counts.items()})
            multiplied["weightedensemble"] += level_count
            return multiplied

        return counts

    def _count_model_configs(self, config: Any) -> int:
        if isinstance(config, list):
            return len(config) or 1
        if isinstance(config, dict):
            if config and all(isinstance(key, int) for key in config):
                nested = Counter()
                for level_key in sorted(config):
                    nested.update(self._count_planned_families(config[level_key]))
                    nested["weightedensemble"] += 1
                return sum(nested.values())
            return 1
        return 1

    def _normalize_family_from_hyperparameter(self, family: str) -> str:
        normalized = str(family).upper()
        if normalized == "GBM":
            return "lightgbm"
        if normalized == "CAT":
            return "catboost"
        if normalized == "XGB":
            return "xgboost"
        if normalized == "NN_TORCH":
            return "nn_torch"
        if normalized == "FASTAI":
            return "fastai"
        if normalized == "RF":
            return "rf"
        if normalized == "XT":
            return "xt"
        if normalized == "KNN":
            return "knn"
        return normalized.lower()

    def _normalize_family_from_model_name(self, model_name: str) -> str:
        if model_name.startswith("WeightedEnsemble"):
            return "weightedensemble"
        if model_name.startswith("LightGBM"):
            return "lightgbm"
        if model_name.startswith("CatBoost"):
            return "catboost"
        if model_name.startswith("XGBoost"):
            return "xgboost"
        if model_name.startswith("NeuralNetTorch") or model_name.startswith("NN_TORCH"):
            return "nn_torch"
        if model_name.startswith("NeuralNetFastAI") or model_name.startswith("FASTAI"):
            return "fastai"
        if model_name.startswith("RandomForest"):
            return "rf"
        if model_name.startswith("ExtraTrees"):
            return "xt"
        if model_name.startswith("KNeighbors"):
            return "knn"
        return "other"

    def _ensure_capacity_for_model(self, model_name: str) -> None:
        family = self._normalize_family_from_model_name(model_name)
        observed_count = sum(1 for name in self.started_models if self._normalize_family_from_model_name(name) == family)
        if observed_count > self.planned_family_counts.get(family, 0):
            self.planned_family_counts[family] = observed_count
        planned_total = sum(self.planned_family_counts.values())
        started_total = len(self.started_models)
        self.total_models_estimate = max(
            self.total_models_estimate or 0,
            planned_total,
            started_total,
        )

    def _estimate_remaining_seconds(self, trainer=None) -> Optional[float]:
        remaining_by_plan = 0.0
        has_remaining_plan = False
        for family, planned_count in self.planned_family_counts.items():
            remaining_count = max(planned_count - self.completed_family_counts.get(family, 0), 0)
            if remaining_count <= 0:
                continue
            remaining_by_plan += remaining_count * self._estimate_family_duration(family)
            has_remaining_plan = True

        trainer_remaining = self._trainer_remaining_time(trainer)
        if has_remaining_plan and trainer_remaining is not None:
            return max(0.0, min(remaining_by_plan, trainer_remaining))
        if has_remaining_plan:
            return max(0.0, remaining_by_plan)
        if trainer_remaining is not None:
            return max(0.0, trainer_remaining)
        return None

    def _estimate_family_duration(self, family: str) -> float:
        observed = self.family_fit_times.get(family)
        if observed:
            return sum(observed) / len(observed)
        return self._family_time_priors.get(family, self._family_time_priors["other"])

    def _trainer_remaining_time(self, trainer) -> Optional[float]:
        if trainer is None:
            trainer = self.trainer
        if trainer is None:
            return None

        total_limit = getattr(trainer, "_time_limit", None)
        train_start = getattr(trainer, "_time_train_start", None)
        if total_limit is None or train_start is None:
            return None
        try:
            remaining = float(total_limit) - (time.time() - float(train_start))
        except (TypeError, ValueError):
            return None
        return max(0.0, remaining)

    def _print_progress(
        self,
        current_model: str,
        completed: int,
        total: int,
        progress_pct: float,
        elapsed: str,
        remaining: str,
    ) -> None:
        """打印进度信息"""
        # 进度条
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # 简化模型名
        short_name = self._shorten_model_name(current_model)

        msg = (
            f"\n{'='*60}\n"
            f"📊 训练进度: [{bar}] {progress_pct:.1f}%\n"
            f"   模型: {completed}/{total} | 当前: {short_name}\n"
            f"   已用: {elapsed} | 预估剩余: {remaining}\n"
            f"{'='*60}"
        )
        logger.info(msg)

    def _print_model_complete(
        self,
        model_name: str,
        score: Optional[float],
        fit_time: Optional[float],
        pred_time: Optional[float],
    ) -> None:
        """打印模型完成信息"""
        short_name = self._shorten_model_name(model_name)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        time_str = f"{fit_time:.1f}s" if fit_time else "N/A"

        # 判断是否为当前最佳
        is_best = self._is_best_score(score)
        best_marker = " ⭐ 新最佳!" if is_best else ""

        logger.info(f"✅ 完成: {short_name} | 分数: {score_str} | 耗时: {time_str}{best_marker}")

    def _print_best_model(self) -> None:
        """打印当前最佳模型"""
        if not self.model_scores:
            return

        best_model = max(self.model_scores, key=self.model_scores.get)
        best_score = self.model_scores[best_model]
        short_name = self._shorten_model_name(best_model)

        logger.info(f"🏆 当前最佳: {short_name} | 分数: {best_score:.4f}")

    def _print_model_skipped(self, stack_name: str, level: int) -> None:
        """打印模型跳过/失败信息。"""
        short_name = self._shorten_model_name(self.current_model or "未知模型")
        logger.warning(
            "⚠️ 跳过/失败: %s | stack=%s | level=L%s",
            short_name,
            stack_name,
            level,
        )

    def _is_best_score(self, score: Optional[float]) -> bool:
        """判断是否为最佳分数"""
        if score is None or not self.model_scores:
            return False
        return score >= max(self.model_scores.values())

    def _shorten_model_name(self, name: str) -> str:
        """缩短模型名"""
        # 移除 _BAG_L1/L2 后缀以简化显示
        return name.replace("_BAG_L1", "").replace("_BAG_L2", "")

    def _format_duration(self, seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"

    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        elapsed = 0
        if self.train_start_time:
            elapsed = time.time() - self.train_start_time

        best_model = None
        best_score = None
        if self.model_scores:
            best_model = max(self.model_scores, key=self.model_scores.get)
            best_score = self.model_scores[best_model]

        return {
            "total_models_trained": len(self.models_completed),
            "models": self.models_completed,
            "scores": self.model_scores,
            "best_model": best_model,
            "best_score": best_score,
            "estimated_total_models": self.total_models_estimate,
            "total_time_seconds": elapsed,
            "total_time_formatted": self._format_duration(elapsed),
        }


def create_progress_callback(
    preset: str = "good_quality",
    time_limit: Optional[int] = None,
) -> TrainingProgressCallback:
    """
    创建进度回调的便捷函数

    Args:
        preset: 模型预设名称
        time_limit: 训练时间限制（秒）

    Returns:
        配置好的 TrainingProgressCallback 实例
    """
    callback = TrainingProgressCallback(
        show_progress_bar=True,
        log_interval=2,
        time_limit=time_limit,
    )

    # 根据 preset 设置预估模型数
    preset_models = {
        "best_quality": 30,
        "high_quality": 20,
        "good_quality": 12,
        "medium_quality": 8,
    }

    return callback
