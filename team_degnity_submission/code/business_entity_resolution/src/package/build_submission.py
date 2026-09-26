"""Submission packaging and archive builder.

Assembles the official submission ZIP archive conforming strictly to competition structure and licensing rules.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from backend.src.config import cfg
from backend.src.exceptions import SubmissionValidationError
from backend.src.logger import get_logger
from backend.src.predict.validator import validate_submission_files

logger = get_logger(__name__)


def build_submission_package(
    team_name: str = "team_degnity",
    output_zip_path: Optional[Path] = None,
    doc_path: Optional[Path] = None,
) -> Path:
    """Constructs the complete competition submission ZIP bundle.

    Required zip structure:
        <team_name>_submission.zip
        ├── output/
        │   ├── matching_results.tsv
        │   └── candidate_pairs.tsv
        ├── code/
        │   └── business_entity_resolution/
        │       ├── src/
        │       ├── README.md
        │       └── requirements.txt
        └── Documentation_template.md

    Args:
        team_name: Name of participating team.
        output_zip_path: Optional custom output path for zip file.
        doc_path: Optional path to filled Documentation_template.md.

    Returns:
        Path to generated ZIP archive.
    """
    logger.info(f"Building final submission archive for team '{team_name}'...")

    ws = Path(cfg.paths.workspace_root)
    matching_tsv = Path(cfg.paths.output_dir) / "matching_results.tsv"
    candidate_tsv = Path(cfg.paths.output_dir) / "candidate_pairs.tsv"

    # Pre-checks
    if not matching_tsv.exists():
        raise SubmissionValidationError(f"matching_results.tsv missing at: {matching_tsv}")
    if not candidate_tsv.exists():
        raise SubmissionValidationError(f"candidate_pairs.tsv missing at: {candidate_tsv}")

    # Discover documentation template
    doc_candidates = [
        doc_path,
        ws / "Documentation_template.md",
        ws / "6ab10eb3b23ba_student_resource" / "student_resource" / "Documentation_template.md",
    ]
    resolved_doc = next((p for p in doc_candidates if p and Path(p).exists()), None)
    if not resolved_doc:
        raise SubmissionValidationError("Documentation_template.md not found.")

    target_zip = output_zip_path or (ws / f"{team_name}_submission.zip")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # 1. output/ folder
        out_sub = tmp_dir / "output"
        out_sub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matching_tsv, out_sub / "matching_results.tsv")
        shutil.copy2(candidate_tsv, out_sub / "candidate_pairs.tsv")

        # 2. code/business_entity_resolution/ folder
        code_sub = tmp_dir / "code" / "business_entity_resolution"
        code_src = code_sub / "src"
        code_src.mkdir(parents=True, exist_ok=True)

        backend_src = ws / "backend" / "src"
        if backend_src.exists():
            shutil.copytree(backend_src, code_src, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp"))

        # Config / suffix dictionary
        conf_dest = code_sub / "config"
        conf_src = ws / "backend" / "config"
        if conf_src.exists():
            shutil.copytree(conf_src, conf_dest, dirs_exist_ok=True)

        # Requirements
        req_src = ws / "backend" / "requirements.txt"
        if req_src.exists():
            shutil.copy2(req_src, code_sub / "requirements.txt")
        else:
            with open(code_sub / "requirements.txt", "w", encoding="utf-8") as f:
                f.write("polars>=0.20.0\nduckdb>=0.9.0\nlightgbm>=4.0.0\nrapidfuzz>=3.0.0\nscikit-learn>=1.3.0\npyarrow>=14.0.0\n")

        # Code README.md
        code_readme = code_sub / "README.md"
        with open(code_readme, "w", encoding="utf-8") as f:
            f.write("# Business Entity Resolution Pipeline\n\n")
            f.write("## End-to-End Execution Guide\n\n")
            f.write("```bash\n")
            f.write("pip install -r requirements.txt\n")
            f.write("python -m src.pipeline run-all\n")
            f.write("```\n\n")
            f.write("## Reproduction Steps:\n")
            f.write("1. Data Normalization: `python -m src.pipeline normalize`\n")
            f.write("2. Candidate Blocking: `python -m src.pipeline block`\n")
            f.write("3. Pairwise Features: `python -m src.pipeline features`\n")
            f.write("4. Model Training: `python -m src.pipeline train`\n")
            f.write("5. Validation & Evaluation: `python -m src.pipeline evaluate`\n")
            f.write("6. Test Prediction & Validation: `python -m src.pipeline predict --validate`\n")

        # 3. Documentation_template.md at root of zip
        shutil.copy2(resolved_doc, tmp_dir / "Documentation_template.md")

        # Create zip archive
        logger.info(f"Writing zip archive to: {target_zip}")
        if target_zip.exists():
            target_zip.unlink()

        with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(tmp_dir)
                    zf.write(file_path, arcname=str(rel_path).replace("\\", "/"))

    zip_size_mb = target_zip.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully packaged {target_zip.name} ({zip_size_mb:.2f} MB).")
    return target_zip
