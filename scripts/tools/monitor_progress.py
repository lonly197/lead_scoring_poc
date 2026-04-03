#!/usr/bin/env python3
"""
训练进度监控脚本

实时分析训练日志，输出友好的进度信息。

用法:
  # 实时监控最新训练日志
  uv run python scripts/monitor_progress.py

  # 监控特定日志文件
  uv run python scripts/monitor_progress.py outputs/logs/train_ohab_20260325_163217.log

  # 持续监控（类似 tail -f）
  uv run python scripts/monitor_progress.py --follow
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)


@dataclass
class ModelInfo:
    """模型训练信息"""
    name: str
    score: Optional[float] = None
    train_time: Optional[float] = None
    pred_time: Optional[float] = None
    layer: int = 1
    status: str = "pending"  # pending, training, completed


@dataclass
class TrainingProgress:
    """训练进度信息"""
    start_time: Optional[datetime] = None
    time_limit: Optional[int] = None
    current_model: Optional[str] = None
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    preset: str = "unknown"

    @property
    def completed_models(self) -> List[ModelInfo]:
        return [m for m in self.models.values() if m.status == "completed"]

    @property
    def best_model(self) -> Optional[ModelInfo]:
        completed = [m for m in self.completed_models if m.score is not None]
        if not completed:
            return None
        return max(completed, key=lambda m: m.score)

    @property
    def total_models_estimate(self) -> int:
        """估算总模型数"""
        preset_models = {
            "best_quality": 30,
            "high_quality": 20,
            "good_quality": 12,
            "medium_quality": 8,
        }
        return preset_models.get(self.preset, 12)


class LogParser:
    """训练日志解析器"""

    # 正则表达式模式
    PATTERNS = {
        "preset": re.compile(r"Presets specified: \['(\w+)'\]"),
        "time_limit": re.compile(r"Time limit = (\d+)s"),
        "fitting_l1": re.compile(r"Fitting (\d+) L1 models"),
        "fitting_l2": re.compile(r"Fitting (\d+) L2 models"),
        "model_start": re.compile(r"Fitting model: (\w+)"),
        "model_complete": re.compile(r"(-?\d+\.\d+)\s+= Validation score"),
        "train_time": re.compile(r"(\d+\.\d+)s\s+= Training\s+runtime"),
        "pred_time": re.compile(r"(\d+\.\d+)s\s+= Validation runtime"),
        "stack_config": re.compile(r"num_stack_levels=(\d+), num_bag_folds=(\d+)"),
        "excluded": re.compile(r"Excluded models: \[([^\]]+)\]"),
        "remaining_time": re.compile(r"Training model for up to ([\d.]+)s of the ([\d.]+)s of remaining time"),
        "start_time": re.compile(r"训练开始时间: ([\d-]+ [\d:]+[+-]\d+)"),
    }

    def __init__(self):
        self.progress = TrainingProgress()

    def parse_line(self, line: str) -> Optional[str]:
        """解析单行日志，返回需要输出的消息"""
        output = None

        # 解析 preset
        match = self.PATTERNS["preset"].search(line)
        if match:
            self.progress.preset = match.group(1)

        # 解析时间限制
        match = self.PATTERNS["time_limit"].search(line)
        if match:
            self.progress.time_limit = int(match.group(1))

        # 解析开始时间
        match = self.PATTERNS["start_time"].search(line)
        if match:
            try:
                self.progress.start_time = datetime.strptime(
                    match.group(1), "%Y-%m-%d %H:%M:%S%z"
                )
            except ValueError:
                pass

        # 解析模型开始
        match = self.PATTERNS["model_start"].search(line)
        if match:
            model_name = match.group(1)
            self.progress.current_model = model_name
            if model_name not in self.progress.models:
                layer = 2 if "_L2" in model_name else 1
                self.progress.models[model_name] = ModelInfo(
                    name=model_name,
                    layer=layer,
                    status="training",
                )
            else:
                self.progress.models[model_name].status = "training"

            output = self._format_progress_update()

        # 解析模型完成
        match = self.PATTERNS["model_complete"].search(line)
        if match and self.progress.current_model:
            score = float(match.group(1))
            if self.progress.current_model in self.progress.models:
                self.progress.models[self.progress.current_model].score = score
                self.progress.models[self.progress.current_model].status = "completed"

        # 解析训练时间
        match = self.PATTERNS["train_time"].search(line)
        if match and self.progress.current_model:
            train_time = float(match.group(1))
            if self.progress.current_model in self.progress.models:
                self.progress.models[self.progress.current_model].train_time = train_time

        # 解析预测时间
        match = self.PATTERNS["pred_time"].search(line)
        if match and self.progress.current_model:
            pred_time = float(match.group(1))
            if self.progress.current_model in self.progress.models:
                self.progress.models[self.progress.current_model].pred_time = pred_time

            # 模型完成时输出摘要
            output = self._format_model_complete()

        return output

    def _format_progress_update(self) -> str:
        """格式化进度更新"""
        completed = len(self.progress.completed_models)
        total = self.progress.total_models_estimate
        pct = (completed / total * 100) if total > 0 else 0

        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        current = self.progress.current_model or "unknown"
        short_name = current.replace("_BAG_L1", "").replace("_BAG_L2", "")

        # 计算已用时间
        elapsed_str = "N/A"
        if self.progress.start_time:
            elapsed = (datetime.now(self.progress.start_time.tzinfo) - self.progress.start_time).total_seconds()
            elapsed_str = self._format_duration(elapsed)

        # 计算预估剩余时间
        remaining_str = "计算中..."
        if completed > 0 and self.progress.start_time:
            elapsed = (datetime.now(self.progress.start_time.tzinfo) - self.progress.start_time).total_seconds()
            if elapsed > 0:
                avg_time = elapsed / completed
                remaining_models = total - completed
                remaining = avg_time * remaining_models
                remaining_str = self._format_duration(remaining)

        return (
            f"\n{'='*60}\n"
            f"📊 训练进度: [{bar}] {pct:.1f}%\n"
            f"   模型: {completed}/{total} | 当前: {short_name}\n"
            f"   已用: {elapsed_str} | 预估剩余: {remaining_str}\n"
            f"{'='*60}"
        )

    def _format_model_complete(self) -> str:
        """格式化模型完成信息"""
        current = self.progress.current_model
        if not current or current not in self.progress.models:
            return ""

        model = self.progress.models[current]
        short_name = model.name.replace("_BAG_L1", "").replace("_BAG_L2", "")

        score_str = f"{model.score:.4f}" if model.score is not None else "N/A"
        time_str = f"{model.train_time:.1f}s" if model.train_time else "N/A"

        # 判断是否为最佳
        is_best = (self.progress.best_model and
                   self.progress.best_model.name == model.name)
        best_marker = " ⭐ 当前最佳!" if is_best else ""

        return f"✅ 完成: {short_name} | 分数: {score_str} | 耗时: {time_str}{best_marker}"

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

    def get_summary(self) -> str:
        """获取训练摘要"""
        lines = ["\n" + "=" * 60]
        lines.append("📋 训练摘要")
        lines.append("=" * 60)

        # 基本信息
        lines.append(f"Preset: {self.progress.preset}")
        if self.progress.time_limit:
            lines.append(f"时间限制: {self.progress.time_limit}秒")

        # 模型统计
        completed = self.progress.completed_models
        lines.append(f"\n已完成模型: {len(completed)}/{self.progress.total_models_estimate}")

        # 最佳模型
        best = self.progress.best_model
        if best:
            short_name = best.name.replace("_BAG_L1", "").replace("_BAG_L2", "")
            lines.append(f"\n🏆 最佳模型: {short_name}")
            lines.append(f"   分数: {best.score:.4f}")
            if best.train_time:
                lines.append(f"   训练时间: {best.train_time:.1f}s")

        # 模型列表
        lines.append("\n📊 模型排行榜:")
        sorted_models = sorted(
            [m for m in completed if m.score is not None],
            key=lambda m: m.score,
            reverse=True
        )
        for i, m in enumerate(sorted_models[:5], 1):
            short_name = m.name.replace("_BAG_L1", "").replace("_BAG_L2", "")
            lines.append(f"   {i}. {short_name}: {m.score:.4f} ({m.train_time:.1f}s)")

        lines.append("=" * 60)
        return "\n".join(lines)


def find_latest_log(log_dir: str = "./outputs/logs") -> Optional[Path]:
    """查找最新的训练日志"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    log_files = list(log_path.glob("train_*.log"))
    if not log_files:
        return None

    return max(log_files, key=lambda p: p.stat().st_mtime)


def monitor_log(log_path: Path, follow: bool = False) -> None:
    """监控日志文件"""
    parser = LogParser()

    print(f"📄 监控日志: {log_path}")
    print("=" * 60)

    # 先读取已有内容
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            output = parser.parse_line(line)
            if output:
                print(output)

    if follow:
        print("\n👀 持续监控中... (Ctrl+C 退出)")
        import time
        last_size = log_path.stat().st_size

        try:
            while True:
                time.sleep(1)
                current_size = log_path.stat().st_size
                if current_size > last_size:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(last_size)
                        for line in f:
                            output = parser.parse_line(line)
                            if output:
                                print(output)
                    last_size = current_size
        except KeyboardInterrupt:
            print("\n\n" + parser.get_summary())


def main():
    parser = argparse.ArgumentParser(
        description="训练进度监控",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 监控最新训练日志
  uv run python scripts/monitor_progress.py

  # 监控特定日志
  uv run python scripts/monitor_progress.py outputs/logs/train_ohab_20260325.log

  # 持续监控（类似 tail -f）
  uv run python scripts/monitor_progress.py --follow
        """,
    )

    parser.add_argument(
        "log_file",
        type=str,
        nargs="?",
        default=None,
        help="日志文件路径（默认监控最新日志）",
    )
    parser.add_argument(
        "-f", "--follow",
        action="store_true",
        help="持续监控模式",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="仅输出训练摘要",
    )

    args = parser.parse_args()

    # 确定日志文件
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = find_latest_log()

    if not log_path or not log_path.exists():
        print("❌ 未找到训练日志文件")
        print("提示: 请指定日志文件路径，或确保 outputs/logs/ 目录存在")
        sys.exit(1)

    if args.summary:
        # 仅输出摘要
        parser_instance = LogParser()
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parser_instance.parse_line(line)
        print(parser_instance.get_summary())
    else:
        monitor_log(log_path, follow=args.follow)


if __name__ == "__main__":
    main()