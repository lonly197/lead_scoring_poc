# AI Lead Scoring POC - Project Instructions

This project is an intelligent lead rating (H/A/B) POC system for auto dealers, leveraging AutoML (AutoGluon) and a two-stage classification pipeline.

## Project Overview

- **Goal**: Provide precise lead scoring to prioritize sales resources.
- **Ratings (HAB)**:
  - **H**: High intent (test drive/order within 7 days).
  - **A**: Medium intent (test drive/order within 14 days).
  - **B**: Low intent (test drive/order within 21 days).
- **Core Technology Stack**:
  - **AutoML**: AutoGluon (Tabular).
  - **Data Processing**: Pandas, Polars, DuckDB (for fast vector operations).
  - **Dependency Management**: `uv`.
  - **Infrastructure**: Ray (for parallel training).

## Core Workflows

### 1. Data Pipeline
The pipeline transforms raw data (Excel/DMP) into training-ready Parquet files.
```bash
# Full pipeline run
uv run python scripts/pipeline/run_pipeline.py --excel ./data/leads.xlsx --dmp ./data/dmp.csv --output ./data/final
```

### 2. Unified Data Splitting
Always use unified splitting to ensure consistency across different model tasks.
```bash
uv run python scripts/pipeline/06_split_unified.py --input <merged_parquet> --output ./data/unified_split --cutoff 2026-03-01
```

### 3. Training
Use the central orchestrator `scripts/run.py` for training tasks.
```bash
# Recommended: Train CatBoost only for speed/performance balance
uv run python scripts/run.py train ensemble --daemon --included-model-types CAT

# Tasks available: arrive, test_drive, ohab, ensemble, order_after_drive
```

### 4. Validation & Monitoring
```bash
# Validate a trained model
uv run python scripts/run.py validate --model-path ./outputs/models/test_drive_model

# Monitor background tasks
uv run python scripts/run.py monitor status
uv run python scripts/run.py monitor log <task_name> -f
```

## Project Structure

- `scripts/`: Entry points for pipeline, training, validation, and tools.
  - `run.py`: Primary CLI entry point.
- `src/`: Core library logic.
  - `data/`: Adapters, loaders, and feature screening.
  - `models/`: AutoGluon wrappers (`predictor.py`) and rating logic (`ohab_rater.py`).
  - `evaluation/`: Metrics (ML + Business) and scorecard logic.
  - `training/`: Specialized training pipelines (HAB, Ensemble).
- `config/`: Configuration management (`config.py`, `column_mapping.py`).
- `docs/`: Detailed technical and business documentation.

## Development Conventions

- **Anti-Leakage Protocol**:
  - **Group Splitting**: Data is split by `phone` or `lead_id` to prevent same-customer leakage across train/test sets.
  - **Feature Dehydration**: Strict exclusion of posterior features (e.g., "Is Test Drove", "Days to Test Drive") defined in `config.py`.
- **Two-Stage HAB Pipeline**: Models are trained in two stages: `H vs non-H` followed by `A vs B` to ensure classification stability and business monotonicity (H > A > B).
- **Configuration Priority**: CLI Arguments > `.env` variables > `config/config.py` defaults.
- **Resource Management**: The system automatically detects CPU/Memory to set `num_folds_parallel` and `memory_limit_gb`.

## Key Commands Cheat Sheet

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies |
| `uv run python scripts/run.py monitor status` | Check running training jobs |
| `uv run python scripts/run.py monitor stop --all` | Stop all background jobs |
| `uv run python scripts/diagnose_data.py <file>` | Diagnose data column mapping issues |
| `pytest` | Run project tests |
