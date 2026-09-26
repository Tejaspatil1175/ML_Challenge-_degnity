"""Unit tests for submission TSV output formatting and integrity validator."""

from pathlib import Path
import polars as pl
import pytest

from backend.src.predict.candidate_writer import write_candidate_pairs_tsv
from backend.src.predict.matching_writer import write_matching_results_tsv
from backend.src.predict.validator import validate_submission_files


def test_candidate_and_matching_writer_and_validator(tmp_path):
    all_s1 = ["S1-001", "S1-002", "S1-003"]
    
    # Create mock test_source1.tsv in tmp_path
    mock_test_s1 = tmp_path / "test_source1.tsv"
    with open(mock_test_s1, "w", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for s1 in all_s1:
            f.write(f"{s1}\tMock Business {s1}\t123 Main St\tUS\n")

    # Candidate DF
    cand_df = pl.DataFrame({
        "source1_entity_id": ["S1-001", "S1-001", "S1-002"],
        "candidate_entity_id": ["S2-100", "S3-200", "S2-300"],
    })
    
    cand_path = tmp_path / "candidate_pairs.tsv"
    write_candidate_pairs_tsv(cand_df, all_s1_ids=all_s1, output_path=cand_path)
    assert cand_path.exists()

    # Scored candidate DF
    scored_df = pl.DataFrame({
        "source1_entity_id": ["S1-001", "S1-001", "S1-002"],
        "candidate_entity_id": ["S2-100", "S3-200", "S2-300"],
        "pred_prob": [0.90, 0.40, 0.85],
    })

    match_path = tmp_path / "matching_results.tsv"
    write_matching_results_tsv(
        scored_candidates_df=scored_df,
        all_s1_ids=all_s1,
        threshold=0.60,
        output_path=match_path,
    )
    assert match_path.exists()

    # Verify content of matching_results.tsv
    with open(match_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f]
    
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert "S1-001\tS2-100" in lines
    assert "S1-002\tS2-300" in lines
    assert "S1-003\t" in lines  # Singleton

    # Validate using validator module with mock test directory
    is_valid, issues = validate_submission_files(
        matching_tsv_path=match_path,
        candidate_tsv_path=cand_path,
        test_dir_path=tmp_path,
    )
    assert is_valid is True
    assert len(issues) == 0


def test_validator_detects_bad_prefix_and_duplicates(tmp_path):
    mock_test_s1 = tmp_path / "test_source1.tsv"
    with open(mock_test_s1, "w", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S1-001\tMock Business\t123 Main St\tUS\n")

    # Broken matching file with duplicate row and S1 self match
    bad_match = tmp_path / "bad_matching.tsv"
    with open(bad_match, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-001\tS1-002,S2-100,S2-100\n")
        f.write("S1-001\tS2-200\n")

    is_valid, issues = validate_submission_files(matching_tsv_path=bad_match, test_dir_path=tmp_path)
    assert is_valid is False
    assert any("Duplicate" in iss for iss in issues)
    assert any("Invalid entity ID" in iss or "self-matches" in iss.lower() or "self match" in iss.lower() for iss in issues)
