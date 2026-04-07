"""
训练管道组件

提供可复用的训练管道组件：
- DataPreparer: 数据准备（加载、分割、清洗、特征工程）
- ModelTrainer: 模型训练（单阶段/两阶段管道）
- ArtifactEvaluator: 评估与 artifact 生成
"""

from src.training.pipeline.data_prep import DataPreparer, DataBundle
from src.training.pipeline.trainer import ModelTrainer
from src.training.pipeline.evaluator import ArtifactEvaluator

__all__ = [
    "DataPreparer",
    "DataBundle",
    "ModelTrainer",
    "ArtifactEvaluator",
]