"""Unit tests for submission packager and archive verification."""

import zipfile
from pathlib import Path
import pytest

from backend.src.package.build_submission import build_submission_package


def test_build_submission_package(tmp_path):
    # Create mock output files
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    m_file = out_dir / "matching_results.tsv"
    c_file = out_dir / "candidate_pairs.tsv"
    
    m_file.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n", encoding="utf-8")
    c_file.write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\n", encoding="utf-8")

    # Create mock doc
    doc_file = tmp_path / "Documentation_template.md"
    doc_file.write_text("# Methodology\n", encoding="utf-8")

    # Build package
    zip_out = tmp_path / "test_team_submission.zip"
    
    # Patch config output dir
    from backend.src.config import cfg
    orig_out = cfg.paths.output_dir
    cfg.paths.output_dir = out_dir

    try:
        pkg_path = build_submission_package(
            team_name="test_team",
            output_zip_path=zip_out,
            doc_path=doc_file,
        )
        assert pkg_path.exists()
        assert pkg_path.suffix == ".zip"

        # Verify archive structure
        with zipfile.ZipFile(pkg_path, "r") as zf:
            namelist = zf.namelist()
            assert "output/matching_results.tsv" in namelist
            assert "output/candidate_pairs.tsv" in namelist
            assert "Documentation_template.md" in namelist
            assert any(name.startswith("code/business_entity_resolution/src/") for name in namelist)
            assert "code/business_entity_resolution/README.md" in namelist
            assert "code/business_entity_resolution/requirements.txt" in namelist
    finally:
        cfg.paths.output_dir = orig_out
