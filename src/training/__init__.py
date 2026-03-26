"""
训练相关模块
"""

try:
    from .progress_callback import TrainingProgressCallback, create_progress_callback
except ModuleNotFoundError:  # pragma: no cover - 测试环境可能未安装 AutoGluon
    TrainingProgressCallback = None

    def create_progress_callback(*args, **kwargs):
        raise ModuleNotFoundError("AutoGluon 未安装，无法创建训练进度回调")

__all__ = ["TrainingProgressCallback", "create_progress_callback"]
