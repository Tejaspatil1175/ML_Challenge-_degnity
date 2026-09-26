"""Dataset exploration and statistics reporting module.

Generates comprehensive data health and distribution summaries for train and test splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import duckdb
import polars as pl

from backend.src.config import cfg
from backend.src.data.loader import load_source, load_ground_truth
from backend.src.data.schema_checks import validate_source_schema, validate_ground_truth_schema
from backend.src.logger import get_logger

logger = get_logger(__name__)


def compute_source_stats(
    name: str,
    path: Path,
    expected_prefix: str,
    sample_n: Optional[int] = None,
) -> Dict[str, object]:
    """Calculates statistics for a source dataset."""
    logger.info(f"Computing statistics for {name} ({path.name})...")
    df = load_source(path, sample_n=sample_n)
    validate_source_schema(df, expected_source=expected_prefix)

    total_rows = df.height
    empty_name_pct = (df.filter(pl.col("business_name") == "").height / total_rows) * 100
    empty_addr_pct = (df.filter(pl.col("business_address") == "").height / total_rows) * 100
    empty_country_pct = (df.filter(pl.col("country") == "").height / total_rows) * 100

    country_counts = (
        df.group_by("country")
        .len()
        .sort("len", descending=True)
        .head(5)
        .to_dicts()
    )

    return {
        "source": name,
        "filename": path.name,
        "rows": total_rows,
        "empty_name_pct": round(empty_name_pct, 4),
        "empty_addr_pct": round(empty_addr_pct, 4),
        "empty_country_pct": round(empty_country_pct, 4),
        "top_countries": country_counts,
    }


def compute_ground_truth_stats(
    path: Path,
    sample_n: Optional[int] = None,
) -> Dict[str, object]:
    """Calculates ground-truth match distribution and singleton rates."""
    logger.info(f"Computing ground truth statistics for {path.name}...")
    df = load_ground_truth(path, sample_n=sample_n)
    validate_ground_truth_schema(df)

    total_s1 = df.height
    
    # Calculate match count per S1 entity
    def count_matches(m_str: str) -> int:
        return 0 if not m_str.strip() else len(m_str.split(","))

    match_counts = df.select(
        pl.col("matched_entity_ids")
        .map_elements(count_matches, return_dtype=pl.Int64)
        .alias("match_count")
    )

    singletons = match_counts.filter(pl.col("match_count") == 0).height
    one_match = match_counts.filter(pl.col("match_count") == 1).height
    two_matches = match_counts.filter(pl.col("match_count") == 2).height
    multi_matches = match_counts.filter(pl.col("match_count") > 2).height

    total_target_matches = match_counts.select(pl.sum("match_count")).item()

    return {
        "total_s1_entities": total_s1,
        "total_target_pairs": total_target_matches,
        "singletons": singletons,
        "singleton_pct": round((singletons / total_s1) * 100, 2),
        "one_match": one_match,
        "one_match_pct": round((one_match / total_s1) * 100, 2),
        "two_matches": two_matches,
        "two_matches_pct": round((two_matches / total_s1) * 100, 2),
        "multi_matches": multi_matches,
        "multi_matches_pct": round((multi_matches / total_s1) * 100, 2),
    }


def generate_exploration_report(
    sample_n: Optional[int] = None,
    output_report_path: Optional[Path] = None,
) -> Path:
    """Runs complete dataset exploration across train and test sets and writes markdown report."""
    logger.info(f"Starting dataset exploration (sample_n={sample_n})...")
    report_file = output_report_path or (Path(cfg.paths.reports_dir) / "explore.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        ("Train Source 1", cfg.paths.train_s1, "S1"),
        ("Train Source 2", cfg.paths.train_s2, "S2"),
        ("Train Source 3", cfg.paths.train_s3, "S3"),
        ("Test Source 1", cfg.paths.test_s1, "S1"),
        ("Test Source 2", cfg.paths.test_s2, "S2"),
        ("Test Source 3", cfg.paths.test_s3, "S3"),
    ]

    source_results = []
    for name, path, prefix in sources:
        stats = compute_source_stats(name, path, prefix, sample_n=sample_n)
        source_results.append(stats)

    gt_stats = compute_ground_truth_stats(cfg.paths.train_gt, sample_n=sample_n)

    # Build Markdown Content
    md = ["# Business Entity Resolution — Dataset Exploration Report\n"]
    md.append(f"> Auto-generated report on dataset integrity, distributions, and ground-truth patterns.\n")
    if sample_n:
        md.append(f"> **Note:** Computed on a random sample of {sample_n:,} rows per file.\n")

    md.append("## 1. Source Files Overview\n")
    md.append("| Dataset Split | File Name | Total Rows | Empty Name % | Empty Addr % | Empty Country % |")
    md.append("|:---|:---|:---:|:---:|:---:|:---:|")
    for s in source_results:
        md.append(
            f"| {s['source']} | `{s['filename']}` | {s['rows']:,} | {s['empty_name_pct']}% | {s['empty_addr_pct']}% | {s['empty_country_pct']}% |"
        )
    md.append("\n---\n")

    md.append("## 2. Country Distributions (Top 5 per Source)\n")
    for s in source_results:
        top_str = ", ".join([f"{c['country'] or '[Missing]'}: {c['len']:,}" for c in s["top_countries"]])
        md.append(f"- **{s['source']}**: {top_str}")
    md.append("\n---\n")

    md.append("## 3. Ground Truth Resolution & Singleton Statistics\n")
    md.append(f"- **Total Reference S1 Entities:** {gt_stats['total_s1_entities']:,}")
    md.append(f"- **Total True Match Pairs ($S_1 \\times (S_2 \\cup S_3)$):** {gt_stats['total_target_pairs']:,}")
    md.append(f"- **Singletons (0 Matches / Standalone):** {gt_stats['singletons']:,} ({gt_stats['singleton_pct']}%)")
    md.append(f"- **1 Match Entities:** {gt_stats['one_match']:,} ({gt_stats['one_match_pct']}%)")
    md.append(f"- **2 Matches Entities:** {gt_stats['two_matches']:,} ({gt_stats['two_matches_pct']}%)")
    md.append(f"- **3+ Multi-Match Entities:** {gt_stats['multi_matches']:,} ({gt_stats['multi_matches_pct']}%)")
    md.append("\n### Strategic Modeling Takeaway:\n")
    md.append(
        "1. Precision is paramount: Singletons score 1.0 if predicted empty and 0.0 if any false match is predicted.\n"
        "2. Multi-match handling: Source 1 entities can match multiple targets across S2 and S3 simultaneously.\n"
    )

    content = "\n".join(md)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Exploration report successfully written to: {report_file}")
    return report_file
