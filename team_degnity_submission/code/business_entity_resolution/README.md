# Business Entity Resolution Pipeline

## End-to-End Execution Guide

```bash
pip install -r requirements.txt
python -m src.pipeline run-all
```

## Reproduction Steps:
1. Data Normalization: `python -m src.pipeline normalize`
2. Candidate Blocking: `python -m src.pipeline block`
3. Pairwise Features: `python -m src.pipeline features`
4. Model Training: `python -m src.pipeline train`
5. Validation & Evaluation: `python -m src.pipeline evaluate`
6. Test Prediction & Validation: `python -m src.pipeline predict --validate`
