# 模型验证指南

## 1. 验证模型（在服务器上执行）

```bash
cd /opt/lead_scoring_poc

# 使用验证脚本
uv run python scripts/validate_model.py

# 或使用新数据验证
uv run python scripts/validate_model.py --data-path /path/to/new_data.csv
```

## 2. 模型性能分析

基于训练日志的结果：

### 整体指标
| 指标 | 值 | 解读 |
|------|-----|------|
| Accuracy | 78.6% | 总体预测准确率 |
| Balanced Accuracy | 81.8% | 考虑类别不平衡后的准确率 |
| MCC | 0.657 | 中等相关性 (范围 -1 到 1) |

### 各类别性能

| 类别 | 含义 | 样本数 | Precision | Recall | F1-Score |
|------|------|--------|-----------|--------|----------|
| O | 已订车/成交 | 42 | 95% | 100% | 98% |
| A | 14天内试驾 | 1070 | 80% | 86% | 83% |
| B | 21天内试驾 | 711 | 78% | 71% | 74% |
| H | 7天内试驾 | 373 | 72% | 71% | 71% |

### 混淆矩阵解读

```
实际\预测    A     B     H     O
   A       915    98    55    2
   B       160   506    45    0
   H        62    48   263    0
   O         0     0     0   42
```

**关键发现**：
1. **O类预测最准确** - 已订车客户特征明显，100%召回率
2. **A类表现良好** - 86%召回率，但部分B类被误判为A类
3. **B类和H类有混淆** - 部分相互误判，但整体可接受

## 3. 模型文件结构

```
outputs/models/ohab_model/
├── predictor.pkl      # 预测器对象
├── learner.pkl        # AutoGluon learner
├── metadata.json      # 元数据
├── version.txt        # 版本信息
└── models/            # 模型文件（已清理，仅保留最佳）
```

## 4. 使用模型进行预测

### Python 脚本

```python
from autogluon.tabular import TabularPredictor

# 加载模型
predictor = TabularPredictor.load("outputs/models/ohab_model")

# 预测新数据
import pandas as pd
new_data = pd.read_csv("new_leads.csv")

# 预测类别
predictions = predictor.predict(new_data)

# 预测概率
probabilities = predictor.predict_proba(new_data)

# 查看结果
print(predictions.head())
print(probabilities.head())
```

### 命令行

```bash
# 查看模型信息
uv run python -c "
from autogluon.tabular import TabularPredictor
p = TabularPredictor.load('outputs/models/ohab_model')
print('最佳模型:', p.model_best)
print('评估指标:', p.eval_metric)
print('类别:', p.class_labels)
"

# 预测示例
uv run python -c "
import pandas as pd
from autogluon.tabular import TabularPredictor

p = TabularPredictor.load('outputs/models/ohab_model')
# 读取新数据预测...
"
```

## 5. 后续建议

1. **收集更多 H 类样本** - 当前 H 类仅 17%，样本量不足影响精度
2. **特征优化** - 分析哪些特征对预测贡献最大
3. **业务验证** - 与业务专家一起验证预测结果是否符合业务逻辑
4. **A/B 测试** - 在实际业务中进行小范围测试验证效果