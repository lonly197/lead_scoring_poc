# Validate Entry Design

目标是将模型验证入口拆分为统一调度层和任务专属实现层。

方案采用单一入口加专属脚本：
- 保留 `scripts/validate_model.py` 作为统一入口。
- 新增 `scripts/validate_ohab_model.py`，承接当前 OHAB 专属验证逻辑。
- 新增 `scripts/validate_arrive_model.py`，负责到店二分类模型验证。

设计约束：
- 保持现有 `validate_model.py` 的后台运行能力和公共 CLI 体验。
- OHAB 现有行为尽量不变，只迁移文件边界，不做业务规则重写。
- 到店验证不复用 H/A/B、scorecard、two-stage pipeline 等 OHAB 专属逻辑。
- 统一入口负责模型类型识别与参数透传，不负责业务评估细节。

分层设计：
- `validate_model.py`
  - 解析公共参数，例如 `--daemon`、`--log-file`、`--model-path`
  - 基于显式参数或模型目录元数据判断模型类型
  - 将命令分发到 `validate_ohab_model.py` 或 `validate_arrive_model.py`
- `validate_ohab_model.py`
  - 继承现有 OHAB 验证逻辑与输出结构
- `validate_arrive_model.py`
  - 加载单模型 arrive 产物
  - 运行二分类评估、Top-K、Lift、特征重要性与基础报告
  - 不承诺 baseline vs best 比较，除非训练侧未来补充基线产物

测试策略：
- 为统一入口新增分发测试，覆盖 `ohab`、`arrive` 和自动识别。
- 为到店验证新增元数据校验测试，明确拒绝 OHAB 产物要求。
- 保留现有 OHAB CLI 测试，避免入口拆分造成回归。
