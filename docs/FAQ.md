# 常见问题 (FAQ)

本文档整理项目常见问题及解答。

---

## 数据处理

### Q: 数据适配器是什么？

**数据适配器**用于自动适配不同格式的数据文件：

| 数据格式 | 特点 |
|----------|------|
| 原格式 | 逗号分隔，有表头，60列 |
| 新格式 | Tab分隔，无表头，46列 |

```python
from src.data.loader import DataLoader

# 启用适配（默认）
loader = DataLoader("data/202603.csv", auto_adapt=True)
df = loader.load()

# 查看检测到的格式
print(loader.get_data_format())
```

### Q: 支持哪些文件格式？

| 格式 | 扩展名 | 支持状态 |
|------|--------|----------|
| CSV | `.csv` | ✅ |
| TSV | `.tsv` | ✅ |
| Parquet | `.parquet` | ✅ |

### Q: 新数据格式如何生成目标变量？

适配器自动从原始时间字段派生：

| 目标变量 | 计算规则 |
|----------|----------|
| `到店标签_14天` | `到店时间 - 线索创建时间 <= 14天` |
| `试驾标签_14天` | `试驾时间 - 线索创建时间 <= 14天` |
| `线索评级结果` | 直接使用 SQL 原始评级列 |

---

## 训练相关

### Q: 三个训练脚本需要按顺序执行吗？

**不需要。** 各脚本完全独立，没有依赖关系。

### Q: 如何后台运行长时间训练？

```bash
# 后台启动
uv run python scripts/run.py train ohab --daemon

# 查看状态
uv run python scripts/run.py monitor status

# 跟踪日志
uv run python scripts/run.py monitor log train_ohab -f

# 停止任务
uv run python/scripts/run.py monitor stop --all
```

**日志位置**：`outputs/logs/{task_name}_{timestamp}.log`

### Q: 如何使用 OOT 时间切分验证？

```bash
uv run python scripts/run.py train ohab --daemon \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20
```

---

## 模型使用

### Q: 如何使用已训练的模型？

```python
from src.models.predictor import LeadScoringPredictor

# 加载模型
predictor = LeadScoringPredictor.load('./outputs/models/arrive_model')

# 预测概率
y_proba = predictor.get_positive_proba(new_data)

# 查看磁盘占用
disk_usage = predictor.get_disk_usage()
print(f"模型大小: {disk_usage['total_size_mb']:.1f} MB")

# 清理非最佳模型释放空间
result = predictor.cleanup(keep_best_only=True)
print(f"释放空间: {result['freed_mb']:.1f} MB")
```

---

## 故障排查

### Q: 遇到 KeyError: '线索创建时间' 怎么办？

```bash
# 诊断数据格式
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 确保代码已同步
git pull && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

### Q: 磁盘空间不足怎么办？

```bash
# 使用轻量预设
uv run python scripts/run.py train test_drive --preset medium_quality

# 清理残留文件
rm -rf outputs/models/*/
rm -rf ~/.cache/autogluon/ /tmp/ray/
```

### Q: 训练失败后如何排查？

1. 查看日志：`uv run python scripts/run.py monitor log train_ohab`
2. 检查磁盘空间：`df -h`
3. 清理残留文件后重试

更多问题请参考 [故障排查指南](./TROUBLESHOOTING.md)。