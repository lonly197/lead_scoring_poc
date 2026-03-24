# 故障排查指南

本文档记录了项目中遇到的常见问题及其解决方案。

---

## 数据适配器问题

### KeyError: '线索创建时间'

**问题现象**：

```
KeyError: '线索创建时间'
```

服务器上运行 `train_ohab_oot.py` 时报错，提示找不到 `线索创建时间` 列。

**问题原因**：

1. **根因**：`adapter.py` 中使用 `strip()` 处理首行数据时，删除了末尾的制表符，导致列数计算错误
2. **表现**：46 列数据被误识别为 43 列，列名映射位置偏移

**修复方案**：

```python
# adapter.py 第 231 行
# 修复前（错误）
actual_columns = len(first_line.strip().split(format_config.sep))

# 修复后（正确）
actual_columns = len(first_line.split(format_config.sep))
```

**验证方法**：

```bash
# 使用诊断脚本验证
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 期望输出
✅ 加载成功!
  - 数据量: 286,823 行
  - 列数: 56 (46 原始列 + 10 派生列)
关键列检查:
  ✅ 线索创建时间
  ✅ 线索唯一ID
  ✅ 到店时间
  ✅ 试驾时间
  ✅ 线索评级结果
```

**修复提交**：`88ecb03` - fix: correct column mapping for 202603.csv data

---

### 服务器代码未同步

**问题现象**：

本地已修复代码，但服务器仍报错。

**排查步骤**：

```bash
# 1. 检查服务器代码版本
git log --oneline -3

# 2. 拉取最新代码
git pull

# 3. 清除 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 4. 重新验证
uv run python scripts/diagnose_data.py ./data/202603.tsv
```

---

## 训练参数问题

### Could not convert string 错误

**问题现象**：

```
训练失败: Could not convert string 'AAHHHAAHH...'
```

**问题原因**：

`线索评级结果` 列没有被排除，导致：
1. 数据泄漏（与目标变量 `线索评级_试驾前` 高度相关）
2. JSON 字段（`跟进记录_JSON`、`跟进详情_JSON`）格式复杂，导致 AutoGluon 处理失败

**修复方案**：

更新 `config/config.py` 中的 `leakage_columns`，添加缺失的排除列：

```python
leakage_columns = [
    # ... 原有列 ...
    "线索评级结果",     # 原始评级，会泄漏目标变量
    "线索评级_试驾后",  # 另一个评级
    "线索评级变化时间_备用",
    "跟进记录_JSON",    # JSON 字段
    "跟进详情_JSON",    # JSON 字段
]
```

**修复提交**：`待提交` - fix: add missing leakage columns for OHAB training

---

## 多分类训练问题

### TypeError: Could not convert string to numeric

**问题现象**：

```
TypeError: Could not convert string 'AAHHHAAHH...' to numeric
```

在 `loader.py` 第 457 行报错。

**问题原因**：

`split_data_oot_three_way` 函数设计时只考虑了二分类场景，调用 `mean()` 计算正样本率。但 OHAB 评级是多分类任务，目标变量是字符串类型（H/A/B/O），无法计算均值。

**修复方案**：

修改 `src/data/loader.py` 中的 `split_data_oot_three_way` 函数，检测目标变量类型：

```python
# loader.py 第 454-467 行
# 检测目标变量类型，区分二分类和多分类
is_numeric_target = pd.api.types.is_numeric_dtype(df[target_label])

for name, subset in [("训练集", train_df), ("验证集", valid_df), ("测试集", test_df)]:
    dist = subset[target_label].value_counts(normalize=True)
    if is_numeric_target:
        # 二分类：计算正样本率
        positive_rate = subset[target_label].mean() * 100
        logger.info(f"{name}: 正样本率 {positive_rate:.2f}%, 分布: {dist.to_dict()}")
    else:
        # 多分类：只显示分布
        logger.info(f"{name}: 目标分布: {dist.to_dict()}")
```

**修复提交**：`待提交` - fix: support multiclass targets in split_data_oot_three_way

---

## 训练脚本接口问题

### TypeError: unexpected keyword argument 'preset'

**问题现象**：

```
TypeError: LeadScoringPredictor.__init__() got an unexpected keyword argument 'preset'
```

**问题原因**：

训练脚本 `train_ohab_oot.py` 与 `LeadScoringPredictor` 类接口不匹配：
1. `preset` 参数应在 `train()` 方法中传递，而非 `__init__()`
2. 脚本调用 `fit()` 方法但类只有 `train()` 方法
3. 脚本调用不存在的 `get_model_name()` 方法

**修复方案**：

修改 `scripts/train_ohab_oot.py`：

```python
# 修复前（错误）
predictor = LeadScoringPredictor(
    preset=args.preset,  # 错误：__init__ 不接受 preset
    ...
)
predictor.fit(...)  # 错误：没有 fit 方法

# 修复后（正确）
predictor = LeadScoringPredictor(...)
predictor.train(
    train_data=...,
    presets=args.preset,  # 正确：在 train() 中传递
    ...
)
```

**修复提交**：`待提交` - fix: align train_ohab_oot.py with LeadScoringPredictor interface

### --time-limit 参数详解

**参数含义**：

`--time-limit` 是 AutoGluon 的核心参数，控制模型训练的总时间（秒）。AutoGluon 是"时间驱动"的 AutoML 框架，在时间限制内自动尝试多种模型（XGBoost、LightGBM、CatBoost、神经网络等），时间到后返回最佳模型。

**时间与预设匹配**：

| time-limit | 推荐预设 | 说明 |
|------------|----------|------|
| 300 (5分钟) | `medium_quality` | 快速验证、调试流程 |
| 1800 (30分钟) | `good_quality` | 初步评估模型效果 |
| 3600 (1小时) | `high_quality` | 生产级模型（默认值） |
| 7200+ (2小时+) | `best_quality` | 追求极致性能 |

**重要**：时间应与预设匹配，过短时间配合高质量预设会导致模型训练不充分。

---

### weight_evaluation 与 balance_weight 冲突

**问题现象**：

```
ValueError: Sample weights cannot be None when weight_evaluation=True.
```

训练完成后在 `calibrate_model()` 阶段报错。

**问题原因**：

`sample_weight="balance_weight"` + `weight_evaluation=True` 参数组合不兼容：

1. `sample_weight="balance_weight"` 启用自动类别权重平衡
2. `weight_evaluation=True` 要求评估时显式传入样本权重
3. AutoGluon 在 `calibrate_model()` 内部调用 `score_with_y_pred_proba(weights=None)`
4. 冲突：`weight_evaluation=True` 禁止 weights=None

**AutoGluon 官方警告**：

```
We do not recommend specifying weight_evaluation when sample_weight='balance_weight'
```

**修复方案**：

移除 `weight_evaluation=True` 参数：

```python
# 修复前（错误）
predictor = LeadScoringPredictor(
    label=target_label,
    sample_weight="balance_weight",
    weight_evaluation=True,  # ← 导致冲突
)

# 修复后（正确）
predictor = LeadScoringPredictor(
    label=target_label,
    sample_weight="balance_weight",  # AutoGluon 自动处理训练和评估权重
)
```

**修复提交**：`待提交` - fix: remove conflicting weight_evaluation parameter

---

## 数据质量问题

### O 级样本不足

**问题现象**：

OHAB 评级中 O 级（已订车/已成交）样本极少：

```
线索评级结果 分布:
H      112,012 (53%)
A       87,175 (41%)
B       10,805 (5%)
O            12 (0.006%)  ← 极度不平衡
```

**问题影响**：

- 12 个样本无法学习有效模式
- 多分类模型可能忽略 O 级或过拟合
- 评估指标不可靠

**解决方案**：

| 方案 | 适用场景 | 实现方式 |
|------|----------|----------|
| **降级为三分类** | O 级无预测价值 | 过滤 O 级，预测 H/A/B |
| **合并为二分类** | 只需区分高意向/低意向 | H/A 合并为"高意向"，B/O 为"低意向" |
| **收集更多数据** | O 级有业务价值 | 等待更多成交数据沉淀 |

**代码修改示例（降级为三分类）**：

```python
# 在训练脚本中添加过滤
df = df[df['线索评级_试驾前'].isin(['H', 'A', 'B'])].copy()
```

---

## 磁盘空间问题

### 空间不足警告

**问题现象**：

```
磁盘状态: 剩余 1.2G / 需要 4.0G (high_quality)
WARNING: 磁盘空间不足！
```

**解决方案**：

```bash
# 1. 使用轻量预设
uv run python scripts/train_arrive.py --preset medium_quality

# 2. 清理模型缓存
rm -rf outputs/models/*/
rm -rf ~/.cache/autogluon/
rm -rf /tmp/ray/

# 3. 查看磁盘占用
du -sh outputs/models/*
```

---

## 常用诊断命令

```bash
# 诊断数据格式
uv run python scripts/diagnose_data.py ./data/202603.tsv

# 测试数据适配器
uv run python scripts/test_adapter.py

# 检查训练状态
uv run python scripts/monitor.py status

# 查看训练日志
uv run python scripts/monitor.py log train_ohab_oot -f
```