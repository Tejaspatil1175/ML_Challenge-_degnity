# Business Entity Resolution — Dataset Exploration Report

> Auto-generated report on dataset integrity, distributions, and ground-truth patterns.

> **Note:** Computed on a random sample of 2,000 rows per file.

## 1. Source Files Overview

| Dataset Split | File Name | Total Rows | Empty Name % | Empty Addr % | Empty Country % |
|:---|:---|:---:|:---:|:---:|:---:|
| Train Source 1 | `train_source1.tsv` | 2,000 | 0.0% | 0.0% | 0.0% |
| Train Source 2 | `train_source2.tsv` | 2,000 | 0.0% | 3.4% | 0.0% |
| Train Source 3 | `train_source3.tsv` | 2,000 | 0.0% | 3.85% | 0.0% |
| Test Source 1 | `test_source1.tsv` | 2,000 | 0.0% | 0.0% | 0.0% |
| Test Source 2 | `test_source2.tsv` | 2,000 | 0.0% | 2.55% | 0.0% |
| Test Source 3 | `test_source3.tsv` | 2,000 | 0.0% | 2.55% | 0.0% |

---

## 2. Country Distributions (Top 5 per Source)

- **Train Source 1**: US: 1,173, India: 827
- **Train Source 2**: US: 1,199, India: 801
- **Train Source 3**: US: 1,190, India: 810
- **Test Source 1**: India: 882, US: 816, France: 302
- **Test Source 2**: India: 904, US: 788, France: 308
- **Test Source 3**: India: 934, US: 768, France: 298

---

## 3. Ground Truth Resolution & Singleton Statistics

- **Total Reference S1 Entities:** 2,000
- **Total True Match Pairs ($S_1 \times (S_2 \cup S_3)$):** 6,871
- **Singletons (0 Matches / Standalone):** 122 (6.1%)
- **1 Match Entities:** 122 (6.1%)
- **2 Matches Entities:** 325 (16.25%)
- **3+ Multi-Match Entities:** 1,431 (71.55%)

### Strategic Modeling Takeaway:

1. Precision is paramount: Singletons score 1.0 if predicted empty and 0.0 if any false match is predicted.
2. Multi-match handling: Source 1 entities can match multiple targets across S2 and S3 simultaneously.
