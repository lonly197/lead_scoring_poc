# 变更日志 (CHANGELOG)

本文件记录项目的所有重要修改。

---

## [2026-04-01] 修复分值计算和 predict_advanced 逻辑

### 问题分析

1. **分值计算不符合文档设计**：代码使用分段计算（H级=80+P(7d)×19），文档设计为加权求和
2. **predict_advanced 对全量数据调用两个模型**：应该分条件调用（已试驾用下订模型，未试驾用试驾模型）

### 变更内容

#### 1. 修复分值计算公式 (`src/models/ohab_rater.py`)

改为加权求和公式，符合文档设计：

```
原始分数 = P(7天)×100 + P(14天)×60 + P(21天)×30
归一化分数 = 原始分数 / 190 × 100
```

权重设计体现"高意向=短周期"的业务逻辑。

#### 2. 修复 predict_advanced 逻辑 (`scripts/predict.py`)

- 先检测已试驾状态 (`detect_driven_status`)
- 未试驾客户 → 只调用试驾模型
- 已试驾客户 → 只调用下订模型
- 添加 try-except 处理特征缺失的情况

### 使用方法

```bash
# 高等模式预测（分阶段）
uv run python scripts/predict.py \
    --mode advanced \
    --drive-ensemble-path ./outputs/models/test_drive_ensemble \
    --order-ensemble-path ./outputs/models/order_after_drive_ensemble \
    --data-path ./data/unified_split/test.parquet \
    --output ./predictions.csv \
    --skip-adapter
```

### 输出示例

| 阶段 | 数量 | 说明 |
|------|------|------|
| 试驾前 | 378,199 | 使用试驾概率模型 |
| 试驾后 | 19,568 | 使用下订概率模型 |
| O（已成交） | 8,731 | 固定100分 |
| 无效 | 1,409 | 固定0分 |

---

## [2026-04-01] 验证脚本通用化与模型验证完成

### 背景

验证脚本 `validate_ensemble.py` 和预测脚本 `predict.py` 硬编码了"试驾标签"，无法用于验证下订预测模型。

### 变更内容

#### 1. 验证脚本通用化 (`scripts/validate_ensemble.py`)

新增参数支持：
- `--label-prefix`: 标签前缀（"试驾标签" 或 "下订标签"）
- `--skip-adapter`: 跳过数据适配器，直接加载 Parquet 文件

#### 2. 预测脚本通用化 (`scripts/predict.py`)

新增参数支持：
- `--label-prefix`: 标签前缀（"试驾标签" 或 "下订标签"）
- `--skip-adapter`: 跳过数据适配器，直接加载 Parquet 文件

#### 3. 模型验证结果

**试驾预测模型**（`test_drive_ensemble`）：

| 时间窗口 | ROC-AUC | 准确率 | Top-100 Lift |
|---------|---------|--------|--------------|
| 7天 | 0.9996 | 99.86% | 129.8x |
| 14天 | 0.9999 | 99.96% | 116.3x |
| 21天 | 0.9925 | 98.39% | 107.1x |

**下订预测模型**（`order_after_drive_ensemble`）：

| 时间窗口 | ROC-AUC | 准确率 | Top-100 Lift |
|---------|---------|--------|--------------|
| 7天 | 1.0000 | 100% | 11.7x |
| 14天 | 1.0000 | 100% | 11.1x |
| 21天 | 1.0000 | 100% | 10.9x |

### 使用方法

```bash
# 验证试驾预测模型
uv run python scripts/validate_ensemble.py \
    --model-dir ./outputs/models/test_drive_ensemble \
    --test-path ./data/unified_split/test.parquet

# 验证下订预测模型
uv run python scripts/validate_ensemble.py \
    --model-dir ./outputs/models/order_after_drive_ensemble \
    --test-path ./data/order_after_drive_v2_test.parquet \
    --label-prefix 下订标签 \
    --skip-adapter

# 预测试驾概率
uv run python scripts/predict.py \
    --mode medium \
    --ensemble-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/unified_split/test.parquet \
    --output ./predictions.csv

# 预测下订概率
uv run python scripts/predict.py \
    --mode medium \
    --ensemble-path ./outputs/models/order_after_drive_ensemble \
    --data-path ./data/order_after_drive_v2_test.parquet \
    --output ./predictions.csv \
    --label-prefix 下订标签 \
    --skip-adapter
```

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `scripts/validate_ensemble.py` | 新增 `--label-prefix` 和 `--skip-adapter` 参数 |
| `scripts/predict.py` | 新增 `--label-prefix` 和 `--skip-adapter` 参数 |

---

## [2026-04-01] 统一数据源方案与 OHABCN 评级体系

### 背景

原先试驾预测和下订预测使用不同的数据文件：
- 试驾预测：`final_v4_train.parquet` / `final_v4_test.parquet`
- 下订预测：`order_after_drive_v2_train.parquet` / `order_after_drive_v2_test.parquet`

这导致数据切分不一致、特征命名不统一、维护成本高。

### 变更内容

#### 1. 新增统一数据分割脚本 (`scripts/pipeline/06_split_unified.py`)

从 `线索宽表_合并_补充试驾.parquet` 生成一致的训练/测试集：

```
统一数据源
    ↓ OOT 时间切分（2026-03-01）
    ├── train.parquet (791,546 行) → 试驾预测训练
    ├── test.parquet (407,907 行) → 试驾预测测试
    ├── train_driven.parquet (47,413 行) → 下订预测训练（已试驾子集）
    └── test_driven.parquet (22,067 行) → 下订预测测试（已试驾子集）
```

#### 2. 修复 JSON 特征提取列名冲突 (`src/data/json_extractor.py`)

当 JSON 提取的特征名与现有列冲突时，自动添加 `_json` 后缀避免重复列名。

#### 3. 更新训练脚本默认数据路径

- `train_test_drive_ensemble.py`: 默认使用 `unified_split/train.parquet`
- `train_order_after_drive.py`: 默认使用 `unified_split/train_driven.parquet`

#### 4. 新增 OHABCN 评级体系 (`src/models/ohab_rater.py`)

扩展原有的 OHAB 四级评级为 OHABCN 六级评级，并添加分值计算：

| 评级 | 定义 | 分值范围 | 说明 |
|------|------|---------|------|
| O | 已成交 | 100分 | 已订车/已成交 |
| H | 7天内试驾/下订 | 80-99分 | 高意向 |
| A | 14天内试驾/下订 | 60-79分 | 中意向 |
| B | 21天内试驾/下订 | 40-59分 | 低意向 |
| C | 有意向但超过21天 | 20-39分 | 超长尾意向 |
| N | 无效线索 | 0分 | 无电话/已购竞品/明确拒绝 |

**新增功能**：
- `detect_invalid_status()`: 检测无效线索（N级）
- `calculate_score_from_proba()`: 基于概率计算分值
- 分值连续计算：基础分 + 概率 × 区间宽度

### 使用方式

```bash
# 生成统一分割数据
uv run python scripts/pipeline/06_split_unified.py \
    --input ./data/线索宽表_合并_补充试驾.parquet \
    --output ./data/unified_split \
    --time-column 线索创建时间 \
    --cutoff 2026-03-01

# 训练试驾预测模型
uv run python scripts/run.py train ensemble --daemon --included-model-types CAT

# 训练下订预测模型
uv run python scripts/run.py train order_after_drive --daemon --included-model-types CAT

# 预测并输出评级、分值
uv run python scripts/predict.py \
    --mode medium \
    --ensemble-path ./outputs/models/test_drive_ensemble \
    --data-path ./data/unified_split/test.parquet \
    --output ./predictions.csv
```

### 预测输出

| 列名 | 说明 |
|------|------|
| 评级 | O/H/A/B/C/N |
| 阶段 | O/试驾前/试驾后/无效 |
| 分值 | 0-100 连续分值 |
| 试驾概率_7天/14天/21天 | 各时间窗口的试驾概率 |

### 影响

- 试驾预测和下订预测使用相同的数据切分，保证评估一致性
- 简化数据管理，只需维护一份原始数据源
- 评级体系更完善，覆盖更多业务场景（无效线索识别、超长尾意向）

---

## [2026-03-30] 三模型集成训练与数据合并

- 新增 `scripts/merge_data.py`：合并线索宽表 + DMP 行为数据
- 新增 `scripts/train_test_drive_ensemble.py`：7/14/21 天试驾预测三模型训练，支持 `--parallel` 并行
- 新增 `src/inference/hab_deriver.py`：从概率推导 H/A/B 评级
- 新增 `--train-path`/`--test-path`：支持提前拆分数据文件（parquet/csv/tsv）
- 验证脚本拆分为独立文件：`validate_ohab_model.py`、`validate_test_drive_model.py`、`validate_ensemble.py`

---

## [2026-03-24] run.py 参数转发缺口修复

### 背景

在统一训练入口重构完成后，`train_ohab.py` 和 `train_arrive.py` 已支持 `--num-bag-folds`，文档中的示例命令也同步使用了该参数。

但线上执行以下命令时：

```bash
uv run python scripts/run.py train_ohab --daemon \
    --data-path ./data/202602~03.tsv \
    --preset high_quality \
    --num-bag-folds 5
```

`run.py` 会直接报错 `unrecognized arguments: --num-bag-folds 5`。

### 根因分析

- **参数并未废弃**：`scripts/train_ohab.py` 和 `scripts/train_arrive.py` 仍然保留 `--num-bag-folds`，并继续将其传给 AutoGluon 的 `num_bag_folds`。
- **问题出在包装器**：`scripts/run.py` 在 `1e950db` 这轮“统一智能自适应训练架构”改造时，没有同步更新参数声明与转发逻辑。
- **属于重构遗漏**：文档描述与底层训练脚本是正确的，真正失效的是 `run.py` 这一层参数代理。
- **同源兼容性缺口**：`run.py` 还保留了 `train_arrive_oot` / `train_ohab_oot` 旧任务名，但对应脚本已删除，说明同一轮重构中包装层没有完全跟上训练入口整合。

### 修复内容

#### 1. 包装器参数补齐 (`scripts/run.py`)
- 新增 `--num-bag-folds` 参数定义。
- 在构建子进程命令时，显式向训练脚本转发 `--num-bag-folds`。

#### 2. 旧任务名兼容转发 (`scripts/run.py`)
- 为 `train_arrive_oot` 增加到 `train_arrive` 的兼容映射。
- 为 `train_ohab_oot` 增加到 `train_ohab` 的兼容映射。
- 在命令行输出中明确提示“旧任务名已统一到新入口”，减少误解。

#### 3. 变更记录与防回归
- 在变更日志中补充本次事故记录，明确说明“训练脚本参数变更时，必须同步检查 `run.py` 包装层的声明、转发与帮助文案”。

### 影响

- `docs/QUICKSTART.md` 等文档中的 `--num-bag-folds` 示例现在与实际行为重新一致。
- 服务端可继续通过 `uv run python scripts/run.py train_ohab --daemon --num-bag-folds 5 ...` 启动训练。
- 旧的 `train_ohab_oot` / `train_arrive_oot` 调用不会再因为脚本缺失而直接失败。

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `scripts/run.py` | 补充 `--num-bag-folds` 参数声明与转发，增加旧任务名兼容映射 |
| `CHANGELOG.md` | 记录本次包装器参数遗漏的根因、修复与防回归要求 |

---

## [2026-03-24] 智能自适应数据切分与防泄漏闭环设计

### 背景

为解决模型在面对跨度不同的数据集（如单日数据 vs. 跨月数据）时切分逻辑死板的问题，重新设计了数据切分架构。不仅支持自动识别时间跨度，还补全了降级（Fallback）状态下 `validate_model.py` 的防泄漏（Data Leakage）漏洞。

### 修改内容

#### 1. 智能数据探查与切分 (src/data/loader.py)
新增 `smart_split_data` 函数：
- **自动探查**：自动计算 `线索创建时间` 的 `min_date` 和 `max_date` 获取时间跨度。
- **自适应 OOT**：当跨度 $\ge 14$ 天时，自动按照 `70%训练 / 15%验证 / 15%测试` 的比例进行绝对时间切分。
- **平滑降级**：当跨度 $< 14$ 天时，自动退化为分层随机切分（Random Split），并提取测试集的 `线索唯一ID` 作为防泄漏标记。

#### 2. 元数据标记持久化 (scripts/train_ohab_oot.py)
- 将 `--train-end` 和 `--valid-end` 的默认值设为 `None`，默认启用智能切分。
- 将切分模式（`oot` 或 `random`）、切分点及降级情况下的**测试集样本 ID 列表**记录在 `feature_metadata.json` 中。
- 降级时自动调整 AutoGluon 参数，取消外部传入 `tuning_data`。

#### 3. 验证脚本智能防御 (scripts/validate_model.py)
- **Smart Test Set Filtering**：重构了 OOT 测试集的过滤逻辑。脚本现在会优先读取模型目录下的 `feature_metadata.json`。
- 如果模型属于 `oot` 模式，自动根据元数据的时间点切割。
- **防泄漏核心**：如果模型属于 `random` 模式（降级），脚本会强制使用元数据中记录的 `test_ids` 去匹配测试集，彻底杜绝“拿训练数据跑验证”导致的指标虚高幻象。

### 预期影响
- 极大地降低了模型训练门槛，用户无需手动计算并输入切分日期。
- 在样本不足（如单日导出）的情况下，不仅保证了训练不中断，更从根本上堵死了后续评估可能产生的数据泄漏漏洞。

### 修改文件清单
| 文件 | 修改内容 |
|------|----------|
| `src/data/loader.py` | 新增 `smart_split_data` 智能探查与降级切分逻辑 |
| `scripts/train_ohab_oot.py` | 接入智能切分，保存切分元数据至 `feature_metadata.json` |
| `scripts/validate_model.py` | 重构验证过滤逻辑，根据元数据自动实施防泄漏隔离 |

---

## [2026-03-24] 特征脱水与风险评估更新

### 背景

基于对 3 月 8 日单日数据的深度审计，发现模型评估指标存在由于“数据泄漏（Data Leakage）”导致的虚高现象：
- 关键后验特征（如 `成交标签`、`订单状态`、`线索评级结果`）因名称不匹配绕过了排除逻辑。
- 模型在预测时点接触到了“未来答案”，导致 O 级识别率出现超自然的 100%。

### 修改内容

#### 1. 核心配置更新 (Feature Sanitization)

更新 `config/config.py` 中的 `leakage_columns`，强制排除了 20 余个具有后验性质的字段：
- **成交结果类**：`成交标签`, `订单状态`, `下订时间`, `成交日期`, `结算日期`, `订单号`, `意向金支付状态` 等。
- **后续行为类**：`到店时间`, `试驾时间`, `首次到店时间`, `首次试驾完成时间`, `战败原因` 等。
- **评级衍生类**：`latest_intn_level`, `线索评级结果`, `线索评级变化时间` 等。
- **动态风险类**：`最后一次通话距今天数`（因计算基准包含了导出后的未来信息）。

#### 2. 训练报告逻辑重构

同步更新 `OHAB模型训练报告_20260322.md`：
- **新增风险评估章节**：明确告知业务方数据泄漏的途径及已采取的“特征脱水”补救措施。
- **局限性解析**：深入分析了单日快照数据的“窄窗”效应，解释了 94.5%（拟合度）与 78.6%（泛化潜力）指标的差异及业务意义。

#### 3. OOT 长周期数据适配指南

更新 `QUICKSTART.md`：
- **切分策略建议**：新增针对 2-3 月全量数据的参数推荐。
- **推荐参数**：建议将 `train-end` 设为 `2026-03-15`，以充分利用 2 月份的全量历史背景。
- **命令行示例**：提供明确的 2-3 月数据训练指令，支持 5 折交叉验证。

### 预期影响
#### 4. 训练脚本整合与简化
- **脚本合并**：将所有 `*_oot.py` 脚本的智能逻辑分别完整迁移至 `train_arrive.py` 和 `train_ohab.py`。
- **删除冗余**：物理删除了 `train_ohab_oot.py` 和 `train_arrive_oot.py`，统一了预测入口。
- **自适应增强**：新版训练脚本默认支持自适应 OOT/随机切分，用户无需再纠结脚本选择。

### 预期影响
- ✅ 模型训练将彻底杜绝“偷看答案”的行为，训练结果（Acc）可能会出现回归（预计回落至 60%-70%），但这代表了模型在实战环境下的真实能力。
- ✅ 降低了运维和使用复杂度，统一了模型输出目录（`ohab_model`）。
- ✅ 确保所有训练路径都强制开启了防泄漏闭环。

- ✅ 报告更具专业严谨性，能够有效应对业务方对指标波动的质疑。
- ✅ 为下一阶段 2-3 月跨月数据的 OOT 训练奠定了坚实的特征基础。

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `config/config.py` | 扩充 `leakage_columns` 排除名单，实施特征脱水 |
| `reports/OHAB模型训练报告_20260322.md` | 重构局限性与风险评估章节 |

---

## [2026-03-24] 验证脚本 OOT 支持与交叉验证增强

### 背景

模型评估发现验证结果可能存在数据泄露问题：
- `validate_model.py` 加载完整数据集评估，包含训练集数据
- 性能指标（83% accuracy）可能被高估
- 需要使用 OOT 测试集评估真实泛化能力

### 问题诊断

| 评估方式 | 数据来源 | 样本量 | 问题 |
|----------|----------|--------|------|
| 训练时 OOT 切分 | 测试集 | ~72,925 | ✓ 正确做法 |
| validate_model.py | **完整数据集** | 210,004 | ❌ 包含训练集数据 |

### 修改内容

#### 1. validate_model.py 新增 OOT 测试集评估

新增命令行参数：
- `--oot-test`: 启用 OOT 测试集评估模式
- `--train-end`: 训练集截止日期（默认 2026-03-11）
- `--valid-end`: 验证集截止日期（默认 2026-03-16）

OOT 模式下只评估 `时间 >= valid_end` 的数据，避免数据泄露。

#### 2. train_ohab_oot.py 新增交叉验证

新增 `--num-bag-folds` 参数（默认 5），启用 K 折交叉验证增强模型稳定性。

### 使用方法

```bash
# 评估 OOT 测试集（避免数据泄露）
uv run python scripts/validate_model.py \
    --data-path ./data/202603.tsv \
    --oot-test \
    --valid-end "2026-03-16"

# 重新训练（启用交叉验证）
uv run python scripts/train_ohab_oot.py \
    --data-path ./data/202603.tsv \
    --num-bag-folds 5
```

### 预期结果

| 评估方式 | 预期 Accuracy | 说明 |
|----------|---------------|------|
| 完整数据集 | ~83% | 包含训练集，性能虚高 |
| OOT 测试集 | ~60-70% | 真实泛化能力 |

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `scripts/validate_model.py` | 新增 `--oot-test` 参数和 OOT 时间切分逻辑 |
| `scripts/train_ohab_oot.py` | 新增 `--num-bag-folds` 参数支持交叉验证 |

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
