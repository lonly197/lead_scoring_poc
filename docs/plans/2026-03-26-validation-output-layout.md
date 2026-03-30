# Validation Output Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将各模型评估结果统一收口到 `outputs/validation/<model>_validation/` 目录结构。

**Architecture:** 保持各验证脚本职责不变，只调整默认输出目录、相关说明文案与现有产物目录位置。统一入口继续只做分发，不新增兼容分支。

**Tech Stack:** Python, pytest, pathlib, Markdown

---

### Task 1: 为默认输出目录补回归测试

**Files:**
- Modify: `tests/test_validate_arrive_model.py`
- Modify: `tests/test_validate_test_drive_model.py`
- Modify: `tests/test_validate_model_cli.py`

**Step 1: Write the failing test**

补充三个 validator 默认输出目录断言，目标分别是：
- `outputs/validation/ohab_validation`
- `outputs/validation/arrive_validation`
- `outputs/validation/test_drive_validation`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_model_cli.py tests/test_validate_arrive_model.py tests/test_validate_test_drive_model.py -q`
Expected: FAIL，因为当前 arrive/test_drive/ohab 仍使用旧目录。

**Step 3: Write minimal implementation**

仅修改脚本默认值与测试桩所需断言，不引入兼容分支。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_model_cli.py tests/test_validate_arrive_model.py tests/test_validate_test_drive_model.py -q`
Expected: PASS

### Task 2: 调整脚本和文档到新目录结构

**Files:**
- Modify: `scripts/validate_ohab_model.py`
- Modify: `scripts/validate_arrive_model.py`
- Modify: `scripts/validate_test_drive_model.py`
- Modify: `scripts/generate_local_plots.py`
- Modify: `docs/QUICKSTART.md`

**Step 1: Write the failing test**

依赖 Task 1 中的测试先失败。

**Step 2: Run test to verify it fails**

同 Task 1。

**Step 3: Write minimal implementation**

把默认输出目录和示例说明改为：
- `outputs/validation/ohab_validation`
- `outputs/validation/arrive_validation`
- `outputs/validation/test_drive_validation`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_model_cli.py tests/test_validate_arrive_model.py tests/test_validate_test_drive_model.py -q`
Expected: PASS

### Task 3: 迁移当前 outputs 下已有目录

**Files:**
- Move: `outputs/validation/* -> outputs/validation/ohab_validation/`
- Move: `outputs/validation_arrive -> outputs/validation/arrive_validation`

**Step 1: Perform directory move**

保留 `outputs/validation/` 作为父目录，只移动现有内容，不保留旧 arrive 平级目录。

**Step 2: Verify layout**

Run: `find outputs -maxdepth 2 -type d | sort`
Expected: 出现三类子目录，且 `validation_arrive` 不再存在。

**Step 3: Run focused verification**

Run: `uv run pytest tests/test_validate_model_cli.py tests/test_validate_arrive_model.py tests/test_validate_test_drive_model.py -q`
Expected: PASS
