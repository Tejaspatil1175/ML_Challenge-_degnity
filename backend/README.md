# Business Entity Resolution — Backend System

Production ML pipeline for resolving business entity records across disparate data sources under strict Macro F0.5 optimization.

## Quickstart

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
# View all available CLI subcommands
python -m src.pipeline --help

# Explore datasets
python -m src.pipeline explore

# Run end-to-end pipeline
python -m src.pipeline run-all
```

### 3. Run Test Suite
```bash
pytest backend/tests/
```
