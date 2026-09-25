# Business Entity Resolution — Dataset Exploration Report

> Auto-generated report on dataset integrity, distributions, and ground-truth patterns.

> **Note:** Computed on a random sample of 50,000 rows per file.

## 1. Source Files Overview

| Dataset Split | File Name | Total Rows | Empty Name % | Empty Addr % | Empty Country % |
|:---|:---|:---:|:---:|:---:|:---:|
| Train Source 1 | `train_source1.tsv` | 50,000 | 0.0% | 0.0% | 0.0% |
| Train Source 2 | `train_source2.tsv` | 50,000 | 0.0% | 3.2% | 0.0% |
| Train Source 3 | `train_source3.tsv` | 50,000 | 0.0% | 3.22% | 0.0% |
| Test Source 1 | `test_source1.tsv` | 50,000 | 0.0% | 0.0% | 0.0% |
| Test Source 2 | `test_source2.tsv` | 50,000 | 0.0% | 2.682% | 0.0% |
| Test Source 3 | `test_source3.tsv` | 50,000 | 0.0% | 2.6% | 0.0% |

---

## 2. Country Distributions (Top 5 per Source)

- **Train Source 1**: US: 29,985, India: 20,015
- **Train Source 2**: US: 29,928, India: 20,072
- **Train Source 3**: US: 30,042, India: 19,958
- **Test Source 1**: India: 23,316, US: 19,329, France: 7,355
- **Test Source 2**: India: 23,446, US: 19,316, France: 7,238
- **Test Source 3**: India: 23,796, US: 19,043, France: 7,161

---

## 3. Ground Truth Resolution & Singleton Statistics

- **Total Reference S1 Entities:** 50,000
- **Total True Match Pairs ($S_1 \times (S_2 \cup S_3)$):** 173,397
- **Singletons (0 Matches / Standalone):** 2,786 (5.57%)
- **1 Match Entities:** 2,636 (5.27%)
- **2 Matches Entities:** 8,552 (17.1%)
- **3+ Multi-Match Entities:** 36,026 (72.05%)

### Strategic Modeling Takeaway:

1. Precision is paramount: Singletons score 1.0 if predicted empty and 0.0 if any false match is predicted.
2. Multi-match handling: Source 1 entities can match multiple targets across S2 and S3 simultaneously.
