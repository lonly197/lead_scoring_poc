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

        # 预设模型列表（根据 preset 推断）
        self._preset_models = {
            "best_quality": 30,    # 估算值
            "high_quality": 20,
            "good_quality": 12,    # L1(5) + L1_ensemble(1) + L2(5) + L2_ensemble(1)
            "medium_quality": 8,
        }

    def _before_model_fit(self, model_name: str, **kwargs) -> None:
        """模型训练前调用"""
        self.current_model = model_name
        self.model_start_time = time.time()

        if self.train_start_time is None:
            self.train_start_time = time.time()

        # 计算进度
        total_models = self._estimate_total_models()
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

    def _after_model_fit(
        self,
        model_name: str,
        score: Optional[float] = None,
        fit_time: Optional[float] = None,
        pred_time: Optional[float] = None,
        **kwargs
    ) -> None:
        """模型训练后调用"""
        self.models_completed.append(model_name)
        if score is not None:
            self.model_scores[model_name] = score

        # 计算该模型耗时
        model_time = 0
        if self.model_start_time:
            model_time = time.time() - self.model_start_time

        # 输出模型完成信息
        self._print_model_complete(
            model_name=model_name,
            score=score,
            fit_time=fit_time or model_time,
            pred_time=pred_time,
        )

        # 每隔一定数量输出最佳模型
        if len(self.models_completed) % self.log_interval == 0:
            self._print_best_model()

    def _estimate_total_models(self) -> int:
        """估算总模型数"""
        # 从 trainer 获取（如果可用）
        if hasattr(self, 'trainer') and self.trainer is not None:
            try:
                # 尝试从 hyperparameters 推断
                hp = getattr(self.trainer, 'hyperparameters', {})
                if hp:
                    count = sum(len(v) if isinstance(v, list) else 1 for v in hp.values())
                    # 考虑 stacking 层数
                    stack_levels = getattr(self.trainer, 'num_stack_levels', 1)
                    return count * (stack_levels + 1) + (stack_levels + 1)  # 加上 ensemble 模型
            except Exception:
                pass

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