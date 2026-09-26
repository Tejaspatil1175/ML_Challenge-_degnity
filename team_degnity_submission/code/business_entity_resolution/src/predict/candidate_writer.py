"""Candidate pairs TSV writer formatted for challenge submission.

Formats blocked candidate pools into one row per reference S1 entity with comma-separated candidate_entity_ids.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set
import polars as pl

from backend.src.config import cfg
from backend.src.data.loader import load_source
from backend.src.exceptions import SubmissionValidationError
from backend.src.logger import get_logger

logger = get_logger(__name__)


def write_candidate_pairs_tsv(
    candidate_df: pl.DataFrame,
    all_s1_ids: List[str] | Set[str],
    output_path: Optional[Path] = None,
) -> Path:
    """Formats and writes candidate_pairs.tsv with exact challenge specification.

    Schema:
        source1_entity_id\tcandidate_entity_ids

    Args:
        candidate_df: DataFrame with ['source1_entity_id', 'candidate_entity_id'].
        all_s1_ids: Complete set/list of all required Source 1 entity IDs in test split.
        output_path: Destination TSV file path.

    Returns:
        Path to written candidate TSV file.
    """
    target = output_path or Path(cfg.paths.output_dir) / "candidate_pairs.tsv"
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info(f"Assembling candidate_pairs.tsv for {len(all_s1_ids):,} total S1 entities...")

    # Group candidates by S1 entity while preserving order and removing duplicates
    cand_map: Dict[str, List[str]] = defaultdict(list)
    seen_pairs: Set[tuple[str, str]] = set()

    if candidate_df.height > 0:
        for s1_id, cand_id in zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_id"]):
            if not cand_id or not str(cand_id).strip():
                continue
            cand_id = str(cand_id).strip()
            pair = (s1_id, cand_id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                cand_map[s1_id].append(cand_id)

    # Write TSV line by line to keep memory minimal
    written_count = 0
    non_empty_count = 0

    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = cand_map.get(s1_id, [])
            cand_str = ",".join(cands)
            f.write(f"{s1_id}\t{cand_str}\n")
            written_count += 1
            if cand_str:
                non_empty_count += 1

    elapsed = time.time() - t0
    logger.info(
        f"Saved candidate_pairs.tsv to {target} in {elapsed:.2f}s: "
        f"{written_count:,} total entities ({non_empty_count:,} with candidates, {written_count - non_empty_count:,} empty)."
    )
    return target
