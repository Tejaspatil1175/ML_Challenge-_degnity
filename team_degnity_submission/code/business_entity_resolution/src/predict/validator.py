"""Submission output validator and official validation runner.

Performs strict format, schema, candidate-subset, and singleton integrity checks matching official competition criteria.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from backend.src.config import cfg
from backend.src.exceptions import SubmissionValidationError
from backend.src.logger import get_logger

logger = get_logger(__name__)


def validate_submission_files(
    matching_tsv_path: Path | str,
    candidate_tsv_path: Optional[Path | str] = None,
    test_dir_path: Optional[Path | str] = None,
    check_ids: bool = False,
) -> Tuple[bool, List[str]]:
    """Runs rigorous pre-checks and the official validate_submission.py tool.

    Args:
        matching_tsv_path: Path to matching_results.tsv.
        candidate_tsv_path: Path to candidate_pairs.tsv.
        test_dir_path: Path to dataset/test containing test_source1.tsv.
        check_ids: Whether to enable memory-heavy ID existence check.

    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
    """
    m_path = Path(matching_tsv_path)
    c_path = Path(candidate_tsv_path) if candidate_tsv_path else None
    t_dir = Path(test_dir_path) if test_dir_path else Path(cfg.paths.raw_dataset_dir) / "test"

    issues: List[str] = []
    logger.info("Executing comprehensive submission validation suite...")

    # 1. Check matching file existence
    if not m_path.exists():
        issues.append(f"Matching file not found at: {m_path}")
        return False, issues

    # 2. Local sanity checks on matching_results.tsv
    with open(m_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header or "\t" not in header:
            issues.append(f"matching_results.tsv header is invalid or not tab-separated: {header!r}")
        cols = [c.strip().lower() for c in header.rstrip("\n").split("\t")]
        if cols != ["source1_entity_id", "matched_entity_ids"]:
            issues.append(f"matching_results.tsv header mismatch. Expected ['source1_entity_id', 'matched_entity_ids'], got {cols}")

        s1_seen = set()
        matched_map: Dict[str, Set[str]] = {}
        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                issues.append(f"Row {line_num} in matching_results.tsv does not have exactly 2 tab-separated columns")
                continue
            s1, m_ids_str = parts
            if s1 in s1_seen:
                issues.append(f"Duplicate source1_entity_id '{s1}' at line {line_num}")
            s1_seen.add(s1)

            ids = [x.strip() for x in m_ids_str.split(",") if x.strip()]
            if len(ids) != len(set(ids)):
                issues.append(f"Duplicate IDs inside matched_entity_ids for '{s1}' at line {line_num}")
            
            for mid in ids:
                if not mid.startswith(("S2-", "S3-")):
                    issues.append(f"Invalid entity ID '{mid}' (must start with S2- or S3-) for '{s1}'")
            matched_map[s1] = set(ids)

    # 3. Local sanity checks on candidate_pairs.tsv if provided
    candidate_map: Dict[str, Set[str]] = {}
    if c_path and c_path.exists():
        with open(c_path, "r", encoding="utf-8") as f:
            c_header = f.readline()
            if not c_header or "\t" not in c_header:
                issues.append(f"candidate_pairs.tsv header is invalid or not tab-separated: {c_header!r}")
            c_cols = [c.strip().lower() for c in c_header.rstrip("\n").split("\t")]
            if c_cols != ["source1_entity_id", "candidate_entity_ids"]:
                issues.append(f"candidate_pairs.tsv header mismatch. Expected ['source1_entity_id', 'candidate_entity_ids'], got {c_cols}")

            for line_num, line in enumerate(f, start=2):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 2:
                    issues.append(f"Row {line_num} in candidate_pairs.tsv does not have exactly 2 tab-separated columns")
                    continue
                s1, c_ids_str = parts
                c_ids = [x.strip() for x in c_ids_str.split(",") if x.strip()]
                candidate_map[s1] = set(c_ids)

        # 4. Subset check: matches ⊆ candidates
        for s1, m_set in matched_map.items():
            c_set = candidate_map.get(s1, set())
            extra = m_set - c_set
            if extra:
                issues.append(f"Matches for '{s1}' contain IDs not in candidates list: {extra}")

    # 5. Invoke official validator script if present
    official_validator_candidates = [
        cfg.paths.workspace_root / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py",
        cfg.paths.workspace_root / "utils" / "validate_submission.py",
    ]
    val_script = next((p for p in official_validator_candidates if p.exists()), None)

    if val_script:
        logger.info(f"Invoking official validator at: {val_script}")
        cmd = [
            sys.executable,
            str(val_script),
            "--matching", str(m_path),
            "--test-dir", str(t_dir),
        ]
        if c_path and c_path.exists():
            cmd.extend(["--candidate", str(c_path)])
        if check_ids:
            cmd.append("--check-ids")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            issues.append(f"Official validator reported failures:\n{result.stdout}\n{result.stderr}")
        else:
            logger.info("Official validator returned PASS (exit 0).")

    is_valid = len(issues) == 0
    if not is_valid:
        logger.error(f"Validation FAILED with {len(issues)} issue(s):")
        for iss in issues[:10]:
            logger.error(f"  - {iss}")
    else:
        logger.info("Submission validation PASSED perfectly! All formatting and constraint rules satisfied.")

    return is_valid, issues
