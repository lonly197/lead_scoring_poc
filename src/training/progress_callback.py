"""
AutoGluon 训练进度监控回调

提供实时训练进度输出，包括：
- 已完成/总模型数
- 当前最佳模型和分数
- 预估剩余时间
- 友好的进度条显示
"""

import logging
import time
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

        # 预设模型列表（根据 preset 推断）
        self._preset_models = {
            "best_quality": 30,    # 估算值
            "high_quality": 20,
            "good_quality": 12,    # L1(5) + L1_ensemble(1) + L2(5) + L2_ensemble(1)
            "medium_quality": 8,
        }

    def before_trainer_fit(self, trainer, **kwargs):
        """训练开始前初始化状态。"""
        self.trainer = trainer
        self.models_completed = []
        self.model_scores = {}
        self.current_model = None
        self.train_start_time = time.time()
        self.model_start_time = None
        self.total_models_estimate = self._estimate_total_models(
            trainer=trainer,
            trainer_kwargs=kwargs,
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

        if self.train_start_time is None:
            self.train_start_time = time.time()

        # 计算进度
        total_models = self.total_models_estimate or self._estimate_total_models(trainer=trainer)
        completed = len(self.models_completed)
        progress_pct = (completed / total_models * 100) if total_models > 0 else 0

        # 计算已用时间
        elapsed = time.time() - self.train_start_time if self.train_start_time else 0
        elapsed_str = self._format_duration(elapsed)

        # 计算预估剩余时间
        if completed > 0 and elapsed > 0:
            avg_time_per_model = elapsed / completed
            remaining_models = total_models - completed
            estimated_remaining = avg_time_per_model * remaining_models
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
            score = self._get_model_attribute(trainer, model_name, "val_score")
            fit_time = self._get_model_attribute(trainer, model_name, "fit_time")
            pred_time = self._get_model_attribute(trainer, model_name, "predict_time")

            if score is not None:
                self.model_scores[model_name] = score

            # 若 trainer 未提供 fit_time，退回到本地计时。
            model_time = 0
            if self.model_start_time:
                model_time = time.time() - self.model_start_time

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
        hp = None
        if trainer_kwargs:
            candidate = trainer_kwargs.get("hyperparameters")
            if isinstance(candidate, dict):
                hp = candidate
        if hp is None and trainer is not None:
            candidate = getattr(trainer, "hyperparameters", None)
            if isinstance(candidate, dict):
                hp = candidate

        if hp:
            count = sum(len(v) if isinstance(v, list) else 1 for v in hp.values())
            num_levels = None

            if trainer_kwargs:
                level_start = trainer_kwargs.get("level_start")
                level_end = trainer_kwargs.get("level_end")
                if isinstance(level_start, int) and isinstance(level_end, int) and level_end >= level_start:
                    num_levels = level_end - level_start + 1

            if num_levels is None and trainer is not None:
                stack_levels = getattr(trainer, "num_stack_levels", None)
                if isinstance(stack_levels, int) and stack_levels >= 0:
                    num_levels = stack_levels + 1

            if num_levels:
                # 每层通常会多一个 WeightedEnsemble。
                return count * num_levels + num_levels

        # 默认返回 good_quality 的估算值
        return self._preset_models.get("good_quality", 12)

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
        preset: AutoGluon preset 名称
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
