# Validate Entry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split model validation into a unified entry script plus model-specific validators for OHAB and arrive models.

**Architecture:** Keep `scripts/validate_model.py` as a thin dispatcher, move current OHAB-specific validation into `scripts/validate_ohab_model.py`, and add `scripts/validate_arrive_model.py` for arrive-model evaluation. Lock routing and arrive metadata behavior with tests before moving logic.

**Tech Stack:** Python, pytest, argparse, subprocess, pandas, AutoGluon wrappers

---

### Task 1: Lock Dispatcher Behavior With Tests

**Files:**
- Modify: `tests/test_validate_model_cli.py`

**Step 1: Write the failing test**

Add tests that expect:
- unified entry dispatches to `validate_ohab_model.py` when model type is `ohab`
- unified entry dispatches to `validate_arrive_model.py` when model type is `arrive`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_model_cli.py -q`

**Step 3: Write minimal implementation**

Add dispatcher helpers in `scripts/validate_model.py`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_model_cli.py -q`

### Task 2: Lock Arrive Validator Metadata Expectations

**Files:**
- Create: `tests/test_validate_arrive_model.py`

**Step 1: Write the failing test**

Add tests that expect:
- arrive validator accepts minimal arrive metadata
- arrive validator rejects OHAB-only assumptions such as missing artifact status not being fatal

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_arrive_model.py -q`

**Step 3: Write minimal implementation**

Create `scripts/validate_arrive_model.py` with metadata validation helpers.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_arrive_model.py -q`

### Task 3: Move OHAB Validator Behind Dedicated Script

**Files:**
- Create: `scripts/validate_ohab_model.py`
- Modify: `scripts/validate_model.py`

**Step 1: Write the failing test**

Extend CLI tests so unified entry still supports daemon mode and forwards args unchanged to OHAB validator.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_model_cli.py -q`

**Step 3: Write minimal implementation**

Copy current OHAB logic into `validate_ohab_model.py` and keep `validate_model.py` as a dispatcher.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_model_cli.py -q`

### Task 4: Implement Arrive Validation Flow

**Files:**
- Create: `scripts/validate_arrive_model.py`

**Step 1: Write the failing test**

Add tests for:
- model-type detection for arrive output directory
- arrive validator default target and output behavior

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_arrive_model.py -q`

**Step 3: Write minimal implementation**

Implement arrive-specific loading, evaluation, Top-K and artifact writing.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_arrive_model.py -q`

### Task 5: Run Focused Verification

**Files:**
- Modify: `scripts/validate_model.py`
- Create: `scripts/validate_ohab_model.py`
- Create: `scripts/validate_arrive_model.py`
- Modify: `tests/test_validate_model_cli.py`
- Create: `tests/test_validate_arrive_model.py`

**Step 1: Run focused tests**

Run: `uv run pytest tests/test_validate_model_cli.py tests/test_validate_arrive_model.py -q`

**Step 2: Run adjacent regression tests**

Run: `uv run pytest tests/test_run_cli.py -q`

**Step 3: Commit**

```bash
git add scripts/validate_model.py scripts/validate_ohab_model.py scripts/validate_arrive_model.py tests/test_validate_model_cli.py tests/test_validate_arrive_model.py docs/plans/2026-03-26-validate-entry-design.md docs/plans/2026-03-26-validate-entry.md
git commit -m "feat(validate): split unified entry by model type"
```
