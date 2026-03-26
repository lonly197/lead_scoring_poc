# 常见问题 (FAQ)

本文档整理了销售线索智能评级 POC 项目的常见问题及解答。

---

## 数据处理

### Q: 数据适配器是什么？如何使用？

**数据适配器**用于自动适配不同格式的数据文件：

| 数据格式 | 示例文件 | 特点 |
|----------|----------|------|
| 原格式 | `20260308-v2.csv` | 逗号分隔，有表头，60列 |
| 新格式 | `202603.csv` | Tab分隔，无表头，46列 |

**重要**：项目训练脚本默认使用 `DataLoader(..., auto_adapt=True)`，会自动适配当前两套主数据格式。

```python
from src.data.loader import DataLoader

# 显式关闭适配：直接按文件原始结构加载
loader = DataLoader("data/your_data.csv", auto_adapt=False)
df = loader.load()

# 启用适配：处理新格式数据（如 202603.csv）
loader = DataLoader("data/202603.csv", auto_adapt=True)
df = loader.load()

# 查看检测到的格式
print(loader.get_data_format())
```

**适配器功能**：
1. ✅ 自动检测分隔符（逗号/Tab）
2. ✅ 自动识别有无表头
3. ✅ 为无表头数据提供列名映射
4. ✅ 从原始时间字段计算目标变量（到店标签、试驾标签等）
5. ✅ 自动衍生时间特征（星期几、小时）

**下一阶段**：当数仓开发同事导出新格式数据后，可直接使用 `auto_adapt=False` 或删除适配代码。

### Q: 支持哪些文件格式？

| 格式 | 分隔符 | 扩展名 | 支持状态 |
|------|--------|--------|----------|
| CSV（逗号） | `,` | `.csv` | ✅ 支持 |
| TSV（Tab） | `\t` | `.tsv`, `.csv` | ✅ 支持 |
| Parquet | - | `.parquet` | ✅ 支持 |

**重要**：文件内容格式优先于扩展名。例如，`.csv` 文件如果是 Tab 分隔，适配器会自动识别。

### Q: 如何使用自己的数据文件？

确保数据文件包含配置中定义的字段，然后：

```bash
# 方式 1：命令行指定（推荐）
uv run python scripts/train_arrive.py --data-path /path/to/your_data.csv

# 方式 2：修改 .env 文件
DATA_PATH=/path/to/your_data.csv
```

**数据文件要求**：
- 格式：CSV/TSV（UTF-8 编码）
- 分隔符：逗号或Tab均可（**自动检测**）
- 表头：有无均可（**自动适配**）
- 目标变量：预计算或原始时间字段均可（**自动派生**）
- 建议包含配置中的特征字段，缺失字段会自动处理

### Q: 新数据格式如何生成目标变量？

新数据格式（如 `202603.csv`）缺少预计算的目标变量，适配器会自动从原始时间字段派生：

| 目标变量 | 计算规则 |
|----------|----------|
| `到店标签_7天` | `到店时间 - 线索创建时间 <= 7天` |
| `到店标签_14天` | `到店时间 - 线索创建时间 <= 14天` |
| `到店标签_30天` | `到店时间 - 线索创建时间 <= 30天` |
| `试驾标签_14天` | `试驾时间 - 线索创建时间 <= 14天` |
| `试驾标签_30天` | `试驾时间 - 线索创建时间 <= 30天` |
| `线索评级_试驾前` | 直接使用 `线索评级结果` 列 |
| `成交标签` | `下订时间 IS NOT NULL` |

**验证数据质量**：

```python
from src.data.loader import DataLoader

loader = DataLoader("data/202603.csv", auto_adapt=True)
df = loader.load()

# 检查目标变量
print(f"到店率 (14天): {df['到店标签_14天'].mean()*100:.2f}%")
print(f"试驾率 (14天): {df['试驾标签_14天'].mean()*100:.2f}%")
print(f"OHAB分布: {df['线索评级_试驾前'].value_counts().to_dict()}")
```

---

## 训练相关

### Q: 三个脚本需要按顺序执行吗？

**不需要。** 三个脚本完全独立，没有依赖关系。当前 POC 主流程优先看 `train_ohab.py`，因为仓库默认配置和最新评估口径都围绕 HAB 智能评级。

### Q: 训练需要多长时间？

对通用小中型任务，`high_quality` 通常约 1 小时（7.3 万条数据）。但对当前 16GB 服务器上的 OHAB 训练，不建议默认使用 `high_quality`，而应优先使用 `server_16g_compare`（`good_quality + 3 folds`）。

### Q: 如何选择预设？

| 预设 | 磁盘需求 | 训练时间 | 精度 | 适用场景 |
|------|----------|----------|------|----------|
| `medium_quality` | ~1G | ~15分钟 | 中等 | 快速验证、磁盘紧张 |
| `good_quality` | ~2G | ~30分钟 | 良好 | 16GB 服务器上的 OHAB 正式推荐档基础 |
| `high_quality` | ~4G | ~1小时 | 高 | 更大机器或更长时限下再使用 |
| `best_quality` | ~8G | ~4小时 | 最高 | 最终优化 |

### Q: 如何处理类别不平衡问题？

训练脚本已默认启用 `sample_weight="balance_weight"`，自动平衡类别权重：

```python
# 在训练脚本中已自动配置
predictor = LeadScoringPredictor(
    label="线索评级_试驾前",
    sample_weight="balance_weight",  # 自动平衡 H/A/B 权重
    weight_evaluation=True,
)
```

**效果**：少数类的召回率会得到一定补偿，但核心提升仍依赖更合适的切分方式、特征和标签口径。

### Q: 如何后台运行长时间训练？

```bash
# 后台启动训练
uv run python scripts/run.py train_arrive --daemon

# 查看运行状态
uv run python scripts/monitor.py status

# 持续跟踪日志
uv run python scripts/monitor.py log train_arrive -f

# 一键停止所有任务
uv run python scripts/monitor.py stop --all
```

**日志文件位置**：`outputs/logs/{task_name}_{timestamp}.log`

**进程信息位置**：`outputs/.process/{task_name}_{pid}.json`

### Q: 如何运行 OOT 验证训练？

OOT（Out-of-Time）验证仍然支持，但对当前 HAB 主流程不是默认模式。`train_ohab.py` 当前默认是随机分组切分，只有显式配置才会启用 OOT：

```bash
# 后台运行到店预测（统一入口）
uv run python scripts/run.py train_arrive --daemon \
    --data-path ./data/202603.tsv

# 后台运行 OHAB 评级（统一入口）
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202603.tsv

# 查看训练状态
uv run python scripts/monitor.py status

# 持续跟踪日志
uv run python scripts/monitor.py log train_arrive -f
uv run python scripts/monitor.py log train_ohab -f
```

**当前切分说明**：

- 默认 `OHAB_SPLIT_MODE=random`，按分组键做 `70%训练 / 15%验证 / 15%测试`
- 只有显式传 `--train-end/--valid-end`，或把 `OHAB_SPLIT_MODE` 设为 `auto_oot/manual_oot` 时，才走时间切分
- 自动 OOT 的最小跨度默认是 `90` 天

如果你希望手动控制切分点：

```bash
uv run python scripts/train_ohab.py \
    --data-path ./data/202602~03.tsv \
    --train-end 2026-03-15 \
    --valid-end 2026-03-20
```

对于 `202603.tsv` 这类单月数据，通常更建议继续使用随机分组切分，而不是依赖自动 OOT。

| 数据集 | 时间范围 | 用途 |
|--------|----------|------|
| 训练集 | < 2026-03-11 | 模型训练 |
| 验证集 | 2026-03-11 ~ 2026-03-16 | 超参数调优 |
| 测试集 | >= 2026-03-16 | 最终评估 |

**什么时候再考虑 OOT？**

当你已经确认数据集是“评分时点快照”，且未来标签观察窗完整时，再显式使用 OOT 更合适。

### Q: 如何使用 OOT 时间切分验证？

当有跨时间段数据时，可使用时间切分而非随机切分：

```python
from src.data.loader import split_data_oot

train_df, test_df = split_data_oot(
    df=df,
    target_label="到店标签_14天",
    time_column="线索创建时间",
    cutoff_date="2026-02-01",  # 此日期之前为训练集，之后为测试集
)
```

**优势**：更好地模拟真实预测场景（用历史数据预测未来）。

### Q: 如何验证模型且避免读到旧的 `ohab_oot` 目录？

统一训练脚本默认输出到 `./outputs/models/ohab_model`。验证时建议显式指定模型路径：

```bash
uv run python scripts/validate_model.py \
    --model-path ./outputs/models/ohab_model \
    --data-path ./data/202602~03.tsv
```

如果模型元数据中存在 `split_info`，`validate_model.py` 会自动只评估训练时定义的测试集；如果数据里包含 `is_final_ordered`，还会额外输出“AI 评级 vs 最终下定”的业务转化统计。

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

这是数据适配器列名映射错误导致的问题。详见 [故障排查指南](./TROUBLESHOOTING.md#数据适配器问题)。

**快速验证**：

```bash
# 诊断数据格式
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 确保代码已同步
git pull
```

### Q: --time-limit 参数如何设置？

`--time-limit` 控制模型训练的总时间（秒）。时间越长，模型尝试的算法越多，质量越高。

| time-limit | 推荐预设 | 适用场景 |
|------------|----------|----------|
| 600 | `medium_quality` | 快速验证 |
| 1800 | `good_quality` | 初步评估 |
| 3600 | `good_quality` 或 `high_quality` | 按机器资源选择，16GB OHAB 不建议直接切 `high_quality` |

**重要**：时间应与预设匹配。详见 [训练脚本说明](./TRAINING.md#--time-limit-参数详解)。

### Q: 磁盘空间不足怎么办？

1. 使用更轻量的 preset：`--preset medium_quality`
2. 清理残留文件：`rm -rf outputs/models/ohab_model/`
3. 清理缓存：`rm -rf ~/.cache/autogluon/ /tmp/ray/`

### Q: 训练失败后如何排查？

1. 查看日志：`uv run python scripts/monitor.py log train_arrive`
2. 检查磁盘空间：`df -h`
3. 查看任务状态：`uv run python scripts/monitor.py list`
4. 清理残留文件后重试

更多问题请参考 [故障排查指南](./TROUBLESHOOTING.md)。
