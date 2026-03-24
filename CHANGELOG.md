# 变更日志 (CHANGELOG)

本文件记录项目的所有重要修改。

---

## [2026-03-24] 数据适配器列映射修复

### 背景

执行 OOT 训练时发现 `202603.csv` 数据加载后列映射错误，导致：
- `线索创建时间` 列包含错误数据（"留资"而非时间戳）
- `线索评级结果` 列包含时间戳而非 OHAB 评级
- OOT 时间切分后训练集为空

### 问题根因

1. **`strip()` 删除末尾制表符**
   - `first_line.strip().split('\t')` 将 46 列误判为 43 列
   - 末尾的空制表符被删除，导致列数计算错误

2. **列映射位置错误**
   - 原映射定义 `线索评级结果` 在索引 27
   - 实际数据中 OHAB 评级在索引 26
   - 差了 1 位，导致后续列全部错位

3. **列数定义不完整**
   - 原定义 43 列，实际数据 46 列
   - 缺少末尾 3 列的列名定义

### 修复内容

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `src/data/adapter.py` | 231 | 删除 `strip()` 调用，保留末尾空列 |
| `src/data/adapter.py` | 41-86 | 重新映射 46 列，修正 OHAB 评级位置 |
| `src/data/adapter.py` | 260-264 | 新增必需列验证，早期发现映射错误 |

### 修正后的列映射（关键列）

| 索引 | 列名 | 样本值 |
|------|------|--------|
| 0 | 线索唯一ID | DIS260301001002008 |
| 4 | 线索创建时间 | 2026-03-01 00:10:03 |
| 26 | 线索评级结果 | H/A/B/O |
| 30 | 线索评级_试驾后 | H/A/B/O |
| 31-35 | AI分析字段 | 是/否 |
| 36 | 到店时间 | 时间戳 |
| 39 | 试驾时间 | 时间戳 |

### 验证结果

**到店预测（到店标签_7天）**：

| 数据集 | 行数 | 正样本率 |
|--------|------|----------|
| 训练集 (< 2026-03-11) | 123,373 | 5.89% |
| 验证集 (2026-03-11 ~ 2026-03-16) | 64,292 | 6.22% |
| 测试集 (>= 2026-03-16) | 99,158 | 5.63% |

**OHAB评级（线索评级_试驾前）**：

| 数据集 | 行数 | OHAB 分布 |
|--------|------|-----------|
| 训练集 | 90,609 | H:48,179 / A:37,692 / B:4,733 / O:5 |
| 验证集 | 46,470 | - |
| 测试集 | 72,925 | - |

### 影响范围

- ✅ 不影响已训练模型
- ✅ 不影响原数据格式加载
- ✅ 修复后 OOT 训练可正常执行

---

## [2026-03-23] 新数据格式适配

### 背景

新导出的数据文件 `202603.csv`（仅铂智3X车型，28.6万条线索）格式与原数据不同，需要适配。

### 数据格式对比

| 维度 | 原数据 (20260308-v2.csv) | 新数据 (202603.csv) |
|------|--------------------------|---------------------|
| 分隔符 | 逗号 | Tab |
| 表头 | ✅ 有中文表头 | ❌ 无表头 |
| 列数 | 60 | 46 |
| 目标变量 | ✅ 预计算 | ❌ 需从原始字段派生 |
| 时间范围 | 单日快照 | 23天连续数据 |

### 新增模块

#### `src/data/adapter.py` - 数据适配器

支持自动检测数据格式并适配：

1. **格式检测**：根据分隔符和首行内容自动识别
2. **列名映射**：为新数据（无表头）提供46列中文列名
3. **目标变量计算**：从原始时间字段派生
   - `到店时间` + `线索创建时间` → `到店标签_7天/14天/30天`
   - `试驾时间` + `线索创建时间` → `试驾标签_14天/30天`
   - `线索评级结果` → `线索评级_试驾前`
4. **时间特征衍生**：`线索创建星期几`、`线索创建小时`

### 数据质量统计

基于 `202603.csv` 分析：

| 指标 | 值 |
|------|-----|
| 总线索数 | 286,823 |
| 到店率 (14天) | 6.08% |
| 试驾率 (14天) | 0.68% |
| 成交率 | 1.69% |
| OHAB 分布 | H(43%), A(30%), Unknown(16%), B(10%), O(1.6%) |

### OOT 验证可行性

```
数据时间范围: 2026-03-01 ~ 2026-03-23
14天观察窗口截止: 2026-03-09
训练集（截止日前）: 103,352 条
测试集（截止日后）: 183,471 条
```

⚠️ **注意**：测试集线索观察期不足14天，建议等待数据沉淀后再进行 OOT 验证。

### 使用方法

```python
from src.data.loader import DataLoader

# 默认行为：不启用适配（推荐，向后兼容）
loader = DataLoader("data/your_data.csv")
df = loader.load()

# 启用适配：处理新格式数据（如 202603.csv）
loader = DataLoader("data/202603.csv", auto_adapt=True)
df = loader.load()
```

**重要说明**：
- 适配器**默认关闭**（`auto_adapt=False`）
- 这是临时方案，下一阶段数仓会导出标准格式数据
- 届时可直接使用默认加载方式，无需适配器

### 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `src/data/adapter.py` | ✨ 新建 - 数据格式适配器 |
| `src/data/loader.py` | 🔧 更新 - 集成适配器 |
| `config/config.py` | 🔧 更新 - 新增 data_format 配置，调整特征定义 |
| `scripts/test_adapter.py` | ✨ 新建 - 适配器测试脚本 |

---

## [2026-03-23] 代码质量优化

### 修复的问题

#### P0 - 立即修复

| 问题 | 文件 | 修改内容 |
|------|------|----------|
| **fillna 顺序错误** | `src/data/loader.py:213, 244` | `astype(str).fillna("missing")` → `fillna("missing").astype(str)` |
| **positive_class 硬编码** | `src/models/predictor.py:152-154` | `proba.iloc[:, 1]` → `proba[positive_class]` |

#### P1 - 本周内修复

| 问题 | 文件 | 修改内容 |
|------|------|----------|
| **列名匹配逻辑错误** | `src/data/loader.py:81-90` | `startswith("label_")` → `"标签" in col or "评级" in col` |
| **默认 ID 列名错误** | `scripts/generate_topk.py:70` | `"门店线索编号"` → `"线索唯一ID"` |

### 新增功能

#### 类别不平衡处理

`LeadScoringPredictor` 新增 `sample_weight` 参数，支持 AutoGluon 原生类别平衡：

```python
predictor = LeadScoringPredictor(
    label="线索评级_试驾前",
    output_path="./outputs/models/ohab_model",
    sample_weight="balance_weight",  # 自动平衡 O/H/A/B 类别权重
    weight_evaluation=True,
)
```

#### OOT 时间切分验证

新增 `split_data_oot()` 函数，支持时间切分验证：

```python
from src.data.loader import split_data_oot

train_df, test_df = split_data_oot(
    df=df,
    target_label="到店标签_14天",
    time_column="线索创建时间",
    cutoff_date="2026-02-01",
)
```

### 架构优化

#### FeatureEngineer 简化

移除与 AutoGluon 内置预处理冲突的手动编码，避免双重预处理：

| 移除的功能 | 处理方 |
|------------|--------|
| 类别编码 (`_encode_categories`) | AutoGluon `CategoryFeatureGenerator` |
| 缺失值填充 (`_handle_missing_values`) | AutoGluon `FillNaFeatureGenerator` |
| 类别映射 (`_apply_category_mappings`) | AutoGluon 自动持久化 |

| 保留的功能 | 原因 |
|------------|------|
| 时间特征提取 | AutoGluon 不自动提取 day_of_week、hour 等 |
| 数值类型转换 | 确保数据类型正确 |

**接口变更**：

```python
# 旧接口
feature_engineer = FeatureEngineer(
    time_columns=[...],
    categorical_columns=[...],  # 已移除
    numeric_columns=[...],
)
df_processed, metadata = feature_engineer.process(df, fit=True, category_mappings=None)

# 新接口
feature_engineer = FeatureEngineer(
    time_columns=[...],
    numeric_columns=[...],  # 可选
)
df_processed, metadata = feature_engineer.process(df)
```

### 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `src/models/predictor.py` | 新增 sample_weight 支持，修复 positive_class |
| `src/data/loader.py` | 修复 fillna 顺序，修复列名匹配，简化 FeatureEngineer，新增 OOT 验证 |
| `scripts/train_ohab.py` | 适配新接口，启用类别权重平衡 |
| `scripts/train_arrive.py` | 适配新接口，启用类别权重平衡 |
| `scripts/train_test_drive.py` | 适配新接口，启用类别权重平衡 |
| `scripts/generate_topk.py` | 修复默认 ID 列名，适配新接口 |

---

## [2026-03-21] 初始版本

- 实现 `train_arrive.py` 到店预测训练脚本
- 实现 `train_test_drive.py` 试驾预测训练脚本
- 实现 `train_ohab.py` OHAB 评级训练脚本
- 实现 `generate_topk.py` Top-K 名单生成
- 实现后台运行和监控功能
- 实现磁盘空间检查和自动清理

---

## 版本规划

### 待实现

| 优先级 | 功能 | 状态 |
|--------|------|------|
| P1 | OOT 验证 | ✅ 已支持（需等待观察期数据沉淀） |
| P2 | 超参数调优 | 📋 计划中 |
| P2 | 单元测试覆盖 | 📋 计划中 |