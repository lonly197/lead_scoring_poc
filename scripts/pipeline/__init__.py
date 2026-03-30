# scripts/pipeline/__init__.py
"""
数据管道脚本模块

提供数据处理流水线的独立脚本：
- 01_merge.py: 数据合并
- 02_profile.py: 数据探查
- 03_clean.py: 数据清洗
- 04_desensitize.py: 数据脱敏
- 05_split.py: 数据拆分
- run_pipeline.py: 统一运行器
"""

__all__ = [
    "merge",
    "profile",
    "clean",
    "desensitize",
    "split",
]