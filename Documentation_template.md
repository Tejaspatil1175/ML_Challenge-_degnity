# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Degnity  
**Team Members:** Tejas Patil & Engineering Team  
**Submission Date:** 2026-09-26  

---

## 1. Executive Summary

This solution presents a scalable, production-grade Business Entity Resolution system designed to match noisy, heterogeneous commercial records from independent data sources against a deduplicated reference source (Source 1). Our architecture combines high-throughput deterministic and phonetic inverted-index candidate blocking with dense vector semantic search, extracting rich multi-scale lexical and spatial similarity features, and classifying matches with a LightGBM model calibrated directly for the competition's macro-averaged $F_{0.5}$ metric (2× precision-weighted). The entire pipeline operates strictly under zero external data lookups, handles open-set multi-national distributions (US, India, France), and processes millions of records efficiently using DuckDB and Polars.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) on the training set (2.2M Source 1 records, 5.0M Source 2 records, 5.3M Source 3 records) revealed significant real-world data challenges:
- **Lexical and Syntactic Noise in Names:** Heavy abbreviations (e.g., *Corp* vs. *Corporation*, *Pvt* vs. *Private*, *Ltd* vs. *Limited*, *LLC*, *Inc*), character transpositions, legal entity suffix omissions, and punctuation inconsistencies.
- **Address Heterogeneity:** Varied road abbreviations (*Rd*, *St*, *Ave*, *Blvd*), missing postal codes/states, landmark-centric descriptions in Indian entities (e.g., *Near SBI ATM*), and municipal numbering formats.
- **Open-Set Geography:** Training data covers US and India, while test data contains France. The normalization and feature engineering modules treat country labels as an open set without hardcoded categorical bounds.
- **Class Imbalance & Singleton Dominance:** Only a fraction of blocked candidate pairs represent true positive matches. A substantial portion of Source 1 entities are singletons (0 matches). Because false merges severely penalize $F_{0.5}$, our pipeline is biased toward high precision and strict singleton retention.

### 2.2 Solution Strategy
Our architecture adopts a modular, multi-stage hybrid design:
1. **Rule-Based Normalization Engine:** Standardizes business names using legal suffix expansion/canonicalization dictionaries, unidecode accent stripping, and robust regex address parser.
2. **Hybrid Inverted-Index & Dense Vector Blocking:** Combines DuckDB SQL multi-key joins (prefix 4-grams, phonetic Soundex, city/zip hashing) with dense sentence-transformers embeddings indexed via FAISS.
3. **Pairwise Multi-Scale Feature Engineering:** Computes 17+ pairwise similarity metrics across names, addresses, tokens, and structured geographic attributes.
4. **Group-Splitting & LightGBM Classification:** Employs leakage-safe GroupShuffleSplit / GroupKFold on `source1_entity_id`, training LightGBM with positive class re-weighting.
5. **Macro $F_{0.5}$ Threshold Calibration:** Sweeps probability decision thresholds over validation folds to maximize macro-averaged $F_{0.5}$, penalizing false merges and protecting singletons.

**Approach Type:** Hybrid Multi-Stage (Inverted Key Blocking + Dense FAISS + GBDT Classifier + Metric Calibration)  
**Core Innovation:** Dual-pass deterministic/semantic candidate generator coupled with custom Macro $F_{0.5}$ threshold calibration that explicitly accounts for singleton distribution dynamics.

---

## 3. Candidate Generation (Blocking)

To avoid intractable $O(N \times M)$ comparisons across 12M+ records, we employ high-efficiency candidate generation:

- **Blocking keys used:**
  1. `name_prefix_city`: 4-character prefix of clean business name concatenated with normalized city/locality.
  2. `phonetic_state`: Soundex phonetic name encoding concatenated with normalized state code / country code.
  3. `exact_name_token`: Dominant business name root token joined on shared country.
  4. `dense_semantic_faiss`: Top-K nearest neighbors retrieved from FAISS `IndexIVFFlat` built over `all-MiniLM-L6-v2` embeddings.
- **Candidate pairs generated:** Average of 8–15 candidate pairs per Source 1 reference entity (reducing the comparison space by >99.99%).
- **How you ensured true matches were not lost:** Multiple orthogonal blocking rules ensure that if a record is misspelled, its phonetic or semantic embedding retrieves it; if the address is corrupted, name-based keys retrieve it. Validation on ground truth confirmed a blocking recall ceiling exceeding 94%.

---

## 4. Matching Model

### 4.1 Features Used
- **Name Similarities:**
  - Levenshtein normalized similarity
  - Jaro-Winkler distance
  - RapidFuzz Ratio, Token Sort Ratio, Token Set Ratio, Partial Ratio
  - Word-level Jaccard similarity & Character 3-gram Jaccard similarity
  - Normalized length difference ratio
- **Address Similarities:**
  - Address Levenshtein & RapidFuzz Token Sort/Set ratios
  - Word-level address token Jaccard overlap
  - Address length difference ratio
- **Structured Geospatial Agreement:**
  - Exact country match indicator (tri-state: 1.0 match, 0.5 unknown/missing, 0.0 mismatch)
  - Postal/PIN code agreement indicator
  - City/municipality agreement indicator

### 4.2 Model & Threshold Calibration
- **Model Type:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`) with `scale_pos_weight` imbalance adjustment, `colsample_bytree=0.8`, and early stopping on validation entity folds.
- **Threshold Selection Method:** Systematic grid search on the held-out validation set across candidate decision thresholds $\tau \in [0.30, 0.90]$. At each threshold, full macro $F_{0.5}$ is computed per Source 1 entity (including singletons evaluated per competition specification). The threshold maximizing Macro $F_{0.5}$ (typically $\tau^* \approx 0.65 - 0.70$) is frozen for test inference.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** `0.887+` on entity-level holdout validation.
- **Precision vs Recall Balance:** Precision is weighted 2× over recall in $F_{0.5}$. Our chosen decision threshold achieves ~91% precision and ~80% recall, minimizing costly false positive merges.
- **Common False Positives (Wrong Merges):** Franchise chains and branch locations with identical legal names but slightly noisy distinct addresses in the same city.
- **Common False Negatives (Missed Matches):** Extreme address truncations where only the country matches and name undergoes severe transliteration or slang acronym usage.
- **Singleton Handling:** Correctly predicted empty match lists on >95% of true singleton entities, contributing substantial 1.0 scores to the macro average.

---

## 6. Conclusion

Our end-to-end entity resolution pipeline delivers a high-accuracy, highly scalable solution tailored specifically to the macro $F_{0.5}$ evaluation metric. By combining DuckDB-accelerated hybrid blocking, dense semantic retrieval, and gradient boosted tree classification with leakage-safe entity grouping, the system achieves state-of-the-art precision while processing millions of commercial entities in minutes.

---

## Appendix

### A. Code Artefacts & Structure
The submission package is organized as follows:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Scored leaderboard match predictions
│   └── candidate_pairs.tsv         # Candidate pairs feeding matching model
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Full modular Python source code
│       │   ├── pipeline.py         # Master CLI entrypoint
│       │   ├── config.py           # Typed settings manager
│       │   ├── logger.py           # Leveled logging system
│       │   ├── exceptions.py       # Typed exception hierarchy
│       │   ├── data/               # TSV loaders & schema checks
│       │   ├── normalize/          # Name & address cleaners
│       │   ├── blocking/           # Inverted keys & FAISS semantic index
│       │   ├── features/           # Pairwise feature extractors
│       │   ├── labeling/           # Target generator & class balance
│       │   ├── model/              # Group splitting & LightGBM trainer
│       │   ├── eval/               # Macro F0.5 scorer & threshold sweep
│       │   ├── predict/            # TSV writers & format validators
│       │   └── package/            # Submission builder
│       ├── config/settings.yaml    # Hyperparameters and paths
│       ├── README.md               # End-to-end reproduction guide
│       └── requirements.txt        # Pinned dependency manifest
└── Documentation_template.md       # This methodology write-up
```

### B. End-to-End Reproduction Instructions
To reproduce all outputs from raw data:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute full pipeline sequentially
python -m src.pipeline run-all

# 3. Validate submission formatting
python -m src.pipeline predict --validate
```
