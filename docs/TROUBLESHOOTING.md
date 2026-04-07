# 故障排查指南

本文档记录常见问题及解决方案。

---

## 数据适配器问题

### KeyError: '线索创建时间'

**原因**：`adapter.py` 中 `strip()` 处理导致列数计算错误。

**解决**：

```bash
# 诊断数据格式
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 确保代码已同步
git pull && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

---

## 训练问题

### 找不到 `train_ohab_oot.py`

**原因**：OOT 逻辑已统一到 `scripts/training/train_ohab.py`。

**解决**：

```bash
# 正确用法
uv run python scripts/run.py train ohab --train-end 2026-03-15 --valid-end 2026-03-20
```

### AssertionError: X_val, y_val is not None, but bagged mode was specified

**原因**：AutoGluon 1.5 在 bagging 模式下不允许 `tuning_data`。

**解决**：代码已自动注入 `use_bag_holdout=True`，确保代码已同步。

### ValueError: Sample weights cannot be None when weight_evaluation=True

**原因**：`sample_weight="balance_weight"` + `weight_evaluation=True` 参数冲突。

**解决**：移除 `weight_evaluation=True` 参数。

---

## 数据质量问题

### O 级样本不足

**现象**：OHAB 评级中 O 级样本极少。

**解决方案**：

| 方案 | 适用场景 |
|------|----------|
| 降级为三分类 | O 级无预测价值 |
| 合并为二分类 | 只需区分高/低意向 |
| 收集更多数据 | O 级有业务价值 |

---

## 磁盘空间问题

**现象**：磁盘空间不足警告。

**解决**：

```bash
# 使用轻量预设
uv run python scripts/run.py train test_drive --preset medium_quality

# 清理残留文件
rm -rf outputs/models/*/
rm -rf ~/.cache/autogluon/ /tmp/ray/
```

---

## 验证问题

### 模型目录缺少 artifact_status

**原因**：训练未完成或异常退出。

**解决**：

```bash
# 查看训练日志
uv run python scripts/run.py monitor log train_ohab -f

# 重新训练
uv run python scripts/run.py train ohab --daemon
```

---

## 常用诊断命令

```bash
# 诊断数据格式
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 检查训练状态
uv run python scripts/run.py monitor status

# 查看训练日志
uv run python scripts/run.py monitor log train_ohab -f

# 清理缓存后重试
rm -rf outputs/cache/
```