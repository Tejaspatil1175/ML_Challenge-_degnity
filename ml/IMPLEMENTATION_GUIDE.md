# Business Entity Resolution — Implementation Guide (Engineering Commit Plan)

Give this whole file to your AI IDE. Work through the commits **in order,
one at a time**. Each commit is a small, single-responsibility change that
should run and be verifiable before you move to the next — that's proper
engineering discipline, not "write everything then debug for a week."

---

## 1. The problem, in plain words

Business records come from 3 sources. The same business can appear in all
3 with different spelling/format. For every Source 1 record, find matching
Source 2 / Source 3 records (zero, one, or many). Scored with **F0.5**
(precision weighted 2× over recall, computed per-entity then averaged) —
so when unsure, don't match.

## 2. Dataset facts

| File | Rows |
|---|---|
| train_source1.tsv | 2,206,823 |
| train_source2.tsv | 5,034,618 |
| train_source3.tsv | 5,285,605 |
| train_ground_truth.tsv | 2,206,823 |
| test_source1.tsv | 1,732,546 |
| test_source2.tsv | 4,887,275 |
| test_source3.tsv | 5,082,318 |

Columns: `entity_id, business_name, business_address, country`.
Ground truth: `source1_entity_id, matched_entity_ids` (comma-separated).

**Millions of rows on both sides → brute-force comparison is impossible.**
Blocking quality sets the recall ceiling for everything downstream.

## 3. Rules to respect

- Output is tab-separated. `matching_results.tsv` needs exactly one row per
  test Source 1 entity (empty list allowed), no duplicate IDs, only real
  test-set S2/S3 IDs. `candidate_pairs.tsv` must be a superset of matches.
- No external APIs / databases / internet lookups.
- Final model: MIT/Apache-2.0, ≤8B parameters.
- Always run `utils/validate_submission.py` before submitting.

## 4. Tech stack

```bash
pip install duckdb polars pyarrow unidecode jellyfish usaddress regex \
            sentence-transformers faiss-cpu rapidfuzz scikit-learn scipy \
            numpy lightgbm tqdm pytest pyyaml
```

| Stage | Library |
|---|---|
| Load huge TSVs | duckdb, polars, pyarrow |
| Clean text | unidecode, jellyfish, usaddress, regex |
| Blocking | rapidfuzz, sentence-transformers (MiniLM), faiss-cpu |
| Features | scikit-learn, numpy, scipy |
| Model | lightgbm |
| Config | pyyaml |
| Testing | pytest |
| Utility | tqdm |

## 5. Final folder structure

```
ml/
├── IMPLEMENTATION_GUIDE.md
backend/
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── exceptions.py
│   ├── pipeline.py            # CLI entrypoint
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── schema_checks.py
│   ├── normalize/
│   │   ├── __init__.py
│   │   ├── name.py
│   │   ├── address.py
│   │   └── suffix_dict.yaml
│   ├── blocking/
│   │   ├── __init__.py
│   │   ├── keys.py
│   │   ├── embed.py
│   │   ├── faiss_index.py
│   │   └── merge.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── pair_features.py
│   ├── labeling/
│   │   ├── __init__.py
│   │   └── join_ground_truth.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── split.py
│   │   ├── train.py
│   │   └── persist.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── f05_scorer.py
│   │   └── threshold_search.py
│   ├── predict/
│   │   ├── __init__.py
│   │   ├── candidate_writer.py
│   │   └── matching_writer.py
│   └── package/
│       ├── __init__.py
│       └── build_submission.py
├── tests/
│   ├── __init__.py
│   ├── test_normalize.py
│   ├── test_blocking_keys.py
│   ├── test_features.py
│   ├── test_f05_scorer.py
│   └── test_output_format.py
├── config/
│   └── settings.yaml
├── README.md
└── requirements.txt
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

---

## 6. Commit plan (32 commits)

### Phase A — Foundation (1–6) [COMPLETED ✅]

- [x] **Commit 1 — Repo scaffolding** `[DONE - 6d3139e]`
  Create the folder tree above (empty files ok). Init git if not already.
  Done when: folder structure exists, matches section 5.

- [x] **Commit 2 — requirements.txt + environment** `[DONE - c2d4f0c]`
  Pin every library version from section 4. Add a `Makefile` or `run.sh`
  with `install`, `test`, `run` targets.
  Done when: `pip install -r requirements.txt` succeeds in a clean venv.

- [x] **Commit 3 — config.py + settings.yaml** `[DONE - c7ba566]`
  Externalize all paths, thresholds, batch sizes, K (top-K for blocking) into
  `config/settings.yaml`, loaded by `config.py`. No hardcoded paths anywhere
  else in the codebase from this point on.
  Done when: changing a value in YAML changes pipeline behavior without code edits.

- [x] **Commit 4 — logger.py** `[DONE - 3ace75e]`
  Central logging setup (level, format, optional file output). Every module
  uses `logger = get_logger(__name__)` instead of `print()`.
  Done when: running any script produces timestamped, leveled log lines.

- [x] **Commit 5 — pipeline.py CLI skeleton** `[DONE - 4aee519]`
  `argparse`-based entrypoint with subcommands: `explore`, `normalize`, `block`,
  `features`, `train`, `evaluate`, `predict`, `package`. Each does nothing yet
  except log "not implemented."
  Done when: `python -m src.pipeline --help` lists all subcommands.

- [x] **Commit 6 — Error handling conventions** `[DONE - e67b09c]`
  Define a small `exceptions.py` (e.g. `SchemaError`, `BlockingError`) and a
  policy: fail loudly on data problems, never silently drop rows without
  logging a warning + count.
  Done when: at least one existing function raises a typed exception instead
  of a bare `Exception`/silent pass.

### Phase B — Data layer (7–10) [COMPLETED ✅]

- [x] **Commit 7 — data/loader.py: load_source()** `[DONE - 513e861]`
  DuckDB-based TSV reader, explicit `sep='\t'`, returns a Polars DataFrame
  with columns `entity_id, business_name, business_address, country`.
  Done when: loading each of the 6 source files returns the exact row counts
  from section 2.

- [x] **Commit 8 — data/schema_checks.py** `[DONE - 0265524]`
  Validate: required columns present, `entity_id` prefix matches expected
  source (S1-/S2-/S3-), no fully-null rows. Raise `SchemaError` on failure.
  Done when: schema check passes on all 6 files, and a unit test proves it
  fails on a deliberately broken sample.

- [x] **Commit 9 — Data exploration report** `[DONE - 5a4f67a]`
  `pipeline.py explore` subcommand: prints row counts, null-field %, country
  distribution, duplicate entity_id check, ground-truth match-count histogram
  (how many S1 entities have 0/1/2+ matches).
  Done when: report runs on train data and output is saved to `reports/explore.md`.

- [x] **Commit 10 — Sampling utility** `[DONE - e4a803c]`
  `data/loader.py`: `load_source(path, sample_n=None)` — lets every later
  script run on a small sample (e.g. 5,000 rows) for fast iteration.
  Done when: every downstream subcommand accepts a `--sample` CLI flag.

### Phase C — Normalization (11–15) [COMPLETED ✅]

- [x] **Commit 11 — normalize/suffix_dict.yaml + name.py** `[DONE - 48d0163]`
  Build the legal-suffix / abbreviation dictionary (Corp/Corporation,
  Pvt/Private, Ltd/Limited, & / and, Inc, LLC, etc.) as YAML data, not
  hardcoded in Python. `normalize_name()` uses it + `unidecode` + lowercasing
  + punctuation stripping.
  Done when: unit test in `tests/test_normalize.py` covers 8+ before/after cases.

- [x] **Commit 12 — normalize/address.py** `[DONE - 34d4e1c]`
  `normalize_address()`: abbreviation expansion (Rd→road, St→street,
  Ave→avenue), lowercase, whitespace cleanup.
  Done when: unit test covers US + India address examples.

- [x] **Commit 13 — Address component extraction** `[DONE - 34d4e1c]`
  `extract_address_parts()`: best-effort city/state/zip via `usaddress` for
  US, regex fallback for India/France (unknown country in test set — must not crash).
  Done when: function returns a dict with `None` for missing parts instead of
  raising, tested on a France-labeled sample.

- [x] **Commit 14 — Apply normalization to full datasets + cache** `[DONE - ad82c26]`
  `pipeline.py normalize`: runs normalization on all 6 files, writes
  normalized Parquet files to `data/processed/` (Parquet, not re-computing
  every run — this matters at millions of rows).
  Done when: normalized Parquet files exist and reload in seconds, not minutes.

- [x] **Commit 15 — Country as open set** `[DONE - ad82c26]`
  Explicitly confirm the pipeline does not hardcode `{US, India}` anywhere —
  grep the codebase for that. Add a unit test using a fake "France" row to
  prove nothing breaks.
  Done when: test passes; this directly satisfies the PDF's open-set requirement.

### Phase D — Blocking (16–23) [COMPLETED ✅]

- [x] **Commit 16 — blocking/keys.py: key generation** `[DONE - ba25d6d]`
  `name_key()` (first 4 chars of normalized name + normalized city),
  `phonetic_key()` (`jellyfish.soundex` + normalized state).
  Done when: unit test proves two known-duplicate business names produce the
  same key.

- [x] **Commit 17 — DuckDB key-based candidate join** `[DONE - ba25d6d]`
  Join S1 to S2/S3 on shared keys via DuckDB SQL (not a Python nested loop —
  must scale to millions of rows).
  Done when: join completes on full train data in a reasonable time (log the
  duration) and produces a candidate table.

- [x] **Commit 18 — eval/f05_scorer.py: recall-at-blocking utility** `[DONE - ba25d6d]`
  Before building the full F0.5 scorer, build a simpler `recall_at_candidates()`
  that checks what % of true matches from ground truth survive the blocking
  step. This is reused later.
  Done when: recall number is printed and logged for the Commit 17 candidates.

- [x] **Commit 19 — blocking/embed.py: embedding model wrapper** `[DONE - 0c32de2]`
  Load `all-MiniLM-L6-v2` once (singleton pattern), expose `embed(texts: list[str]) -> np.ndarray`.
  Done when: embedding 1,000 sample rows completes and returns correct shape.

- [x] **Commit 20 — Batch embedding with checkpointing** `[DONE - 0c32de2]`
  Embed all 3 sources in batches (e.g. 10k rows/batch) with `tqdm` progress,
  save embeddings to disk (`.npy` or memory-mapped) so a crash doesn't mean
  starting over.
  Done when: embedding job can be interrupted and resumed from the last
  completed batch.

- [x] **Commit 21 — blocking/faiss_index.py** `[DONE - 0c32de2]`
  Build a FAISS index per source (`IndexIVFFlat` for speed at this scale),
  save/load the index to disk.
  Done when: index builds on full Source 2 + Source 3 embeddings without
  running out of memory (log peak memory if possible).

- [x] **Commit 22 — Top-K query per Source 1 entity** `[DONE - 0c32de2]`
  For each S1 embedding, query top-K (config-driven, default 10) nearest
  neighbors from each FAISS index.
  Done when: query returns candidate IDs + similarity scores for a sample of
  S1 entities in well under a second per query.

- [x] **Commit 23 — blocking/merge.py: combine + dedupe** `[DONE - 0c32de2]`
  Merge Commit 17 (key-based) + Commit 22 (embedding-based) candidates per S1
  entity, dedupe IDs, write `candidate_pairs.tsv`.
  Done when: `candidate_pairs.tsv` exists for train data, format matches PDF
  spec exactly (tab-separated, one row per S1 entity, comma-separated IDs).

### Phase E — Features & labels (24–28) [COMPLETED ✅]

- [x] **Commit 24 — Blocking recall report** `[DONE - 6f9980a]`
  Re-run Commit 18's recall utility on the merged candidates from Commit 23.
  Log recall + average candidates-per-entity. Tune `K` in settings.yaml if
  recall is too low or candidate count too high.
  Done when: recall number is written to `reports/blocking_recall.md` with
  the chosen K justified.

- [x] **Commit 25 — features/pair_features.py: string similarity** `[DONE - 6f9980a]`
  `rapidfuzz` ratio + token_sort_ratio on name and address per candidate pair.
  Done when: unit test in `tests/test_features.py` checks known similar/dissimilar pairs.

- [x] **Commit 26 — Token + embedding similarity features** `[DONE - 8bc3f4f]`
  Add Jaccard token overlap, TF-IDF cosine (fit on train names), and reuse
  Commit 20's embeddings for cosine similarity.
  Done when: feature dataframe has all columns, no NaNs (impute or drop with logging).

- [x] **Commit 27 — Structured match features + assembly** `[DONE - 8bc3f4f]`
  Add country/city/state exact-match flags. Assemble full feature dataframe,
  cache as Parquet (`data/processed/features.parquet`).
  Done when: one row per candidate pair, all features present, file loads fast.

- [x] **Commit 28 — labeling/join_ground_truth.py** `[DONE - 8bc3f4f]`
  Join candidate pairs with `train_ground_truth.tsv` to create binary labels.
  Log class balance (% positive vs negative) — expect heavy imbalance.
  Done when: labeled dataframe saved, class balance logged.

### Phase F — Model (29–31) [COMPLETED ✅]

- [x] **Commit 29 — model/split.py: leakage-safe split** `[DONE - 6d42fa0]`
  Split by **Source 1 entity_id** (not by row) into train/val (80/20), so no
  entity's candidates leak across the split.
  Done when: unit test proves zero entity_id overlap between train/val sets.

- [x] **Commit 30 — model/train.py + persist.py** `[DONE - 2c919a5]`
  Train `lightgbm.LGBMClassifier` on Commit 27 features + Commit 28 labels,
  handle class imbalance (`class_weight` or `scale_pos_weight`), log feature
  importances, save model with versioned filename (`model_v1.txt`).
  Done when: model trains, is saved, and can be reloaded to produce identical
  predictions.

- [x] **Commit 31 — eval/f05_scorer.py: full metric + eval/threshold_search.py** `[DONE - ea3ca77]`
  Implement F0.5 exactly per the PDF (per-entity, macro-averaged, singleton
  rules included). Sweep thresholds 0.3–0.9 step 0.05 on validation set, pick
  best, log a threshold-vs-F0.5 table.
  Done when: chosen threshold + validation F0.5 score logged to
  `reports/eval_results.md`.

### Phase G — Inference & delivery (32+) [COMPLETED ✅]

- [x] **Commit 32 — predict/candidate_writer.py + matching_writer.py** `[DONE - 9b4046e]`
  Run full blocking (Commits 16–23) + features (25–27) + trained model
  (Commit 30) + chosen threshold (Commit 31) on the **test** set. Write
  `candidate_pairs.tsv` and `matching_results.tsv` to `output/`. Every test
  Source 1 entity must appear, even with an empty match list.
  Done when: both files exist, correct format.

- [x] **Commit 33 — Output validation wrapper** `[DONE - 9b4046e]`
  Wrap `utils/validate_submission.py` in `pipeline.py predict --validate`,
  plus your own pre-checks (no duplicate IDs, matches ⊆ candidates) that run
  before the official validator.
  Done when: `PASS` from the official validator.

- [x] **Commit 34 — Test suite completion** `[DONE - 9b4046e]`
  Fill out `tests/test_output_format.py` covering: tab separation, one row
  per entity, no dupes, IDs exist in test set. Run full `pytest` suite.
  Done when: all tests green.

- [x] **Commit 35 — package/build_submission.py + docs** `[DONE - 8b5d8d4]`
  Fill in `Documentation_template.md` (methodology, blocking strategy, model
  architecture, features). Script that zips
  `<team_name>_submission.zip` per the PDF's required structure
  (`output/`, `code/business_entity_resolution/`, doc file).
  Done when: zip produced and matches the required structure exactly.

---

## 7. Engineering discipline while implementing

- One commit = one focused change. Don't bundle blocking + features + model
  into one giant commit — it makes bugs impossible to isolate.
- Every function that touches the full dataset should also work on a small
  `--sample` for fast local iteration (Commit 10).
- Log row counts and timing at every major transform — on data this large,
  a silent 10x slowdown or row-count mismatch is how bugs hide.
- Cache expensive intermediate results (normalized data, embeddings,
  features) to Parquet/`.npy` — never recompute embeddings on every run.
- Write the unit test in the same commit as the function, not "later."
