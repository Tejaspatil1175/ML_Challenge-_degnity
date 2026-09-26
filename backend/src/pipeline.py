"""Business Entity Resolution - Master Command Line Interface.

Orchestrates all pipeline stages from exploration to submission packaging.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from backend.src.blocking.runner import run_blocking_stage
from backend.src.config import cfg
from backend.src.data.explore import generate_exploration_report
from backend.src.features.runner import run_features_stage
from backend.src.logger import get_logger
from backend.src.normalize.batch_normalize import run_normalization_pipeline

logger = get_logger("pipeline")


def cmd_explore(args: argparse.Namespace) -> int:
    """Explores raw dataset distributions, nulls, and statistics."""
    logger.info("Executing subcommand: explore")
    report_path = generate_exploration_report(sample_n=args.sample)
    logger.info(f"Exploration completed. Report saved at: {report_path}")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    """Runs text and address normalization on source datasets."""
    logger.info("Executing subcommand: normalize")
    saved = run_normalization_pipeline(sample_n=args.sample)
    logger.info(f"Normalization completed. Parquet files cached at: {list(saved.values())[0].parent}")
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Executes rule-based and semantic dense vector candidate blocking."""
    logger.info("Executing subcommand: block")
    split = getattr(args, "split", "train")
    cands, stats = run_blocking_stage(split=split, sample_n=args.sample)
    logger.info(f"Candidate blocking completed: {cands.height:,} candidate pairs generated.")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Extracts pairwise string, token, semantic, and spatial similarity features."""
    logger.info("Executing subcommand: features")
    split = getattr(args, "split", "train")
    feat_df = run_features_stage(split=split, sample_n=args.sample)
    logger.info(f"Features extraction completed: {feat_df.height:,} rows generated.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Trains LightGBM classifier using entity-grouped leakage-safe splitting."""
    logger.info("Executing subcommand: train")
    from pathlib import Path
    import polars as pl
    from backend.src.model.train import train_matching_model

    proc_dir = Path(cfg.paths.processed_dir)
    feat_file = proc_dir / "train_features.parquet"
    if not feat_file.exists():
        logger.info(f"Feature table not found at {feat_file}. Running feature extraction first...")
        from backend.src.features.runner import run_features_stage
        run_features_stage(split="train")

    logger.info(f"Loading training features from: {feat_file}")
    features_df = pl.read_parquet(feat_file)

    val_ratio = cfg.training.get("val_ratio", 0.20)
    model, feature_names, val_split, val_probs = train_matching_model(
        train_features_df=features_df,
        val_ratio=val_ratio,
        model_version="v1",
    )
    logger.info("Model training stage completed successfully.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Performs threshold sweep and Macro F0.5 evaluation on validation split."""
    logger.info("Executing subcommand: evaluate")
    from pathlib import Path
    import polars as pl
    from backend.src.data.loader import load_ground_truth
    from backend.src.eval.threshold_search import (
        save_evaluation_report,
        search_optimal_threshold,
    )
    from backend.src.model.persist import load_model
    from backend.src.model.split import split_train_val
    from backend.src.model.train import get_feature_columns, predict_pair_probabilities

    model_path = Path(cfg.paths.models_dir) / "latest_model.joblib"
    if not model_path.exists():
        logger.warning(f"Trained model not found at {model_path}. Running training first...")
        cmd_train(args)

    model, meta = load_model(model_path)
    feat_file = Path(cfg.paths.processed_dir) / "train_features.parquet"
    if not feat_file.exists():
        logger.warning(f"Feature table not found at {feat_file}. Running feature extraction first...")
        from backend.src.features.runner import run_features_stage
        run_features_stage(split="train")

    features_df = pl.read_parquet(feat_file)
    _, val_split = split_train_val(
        features_df,
        val_ratio=cfg.training.get("val_ratio", 0.20),
        random_seed=cfg.execution.get("random_seed", 42),
    )

    feature_cols = meta.get("metadata", {}).get("feature_names") or get_feature_columns(val_split)
    val_probs = predict_pair_probabilities(model, val_split, feature_cols)
    val_scored = val_split.with_columns(pl.Series("pred_prob", val_probs))

    gt_df = load_ground_truth(cfg.paths.train_gt)
    best_thresh, best_f05, sweep_df = search_optimal_threshold(
        val_scored_df=val_scored,
        ground_truth_df=gt_df,
        threshold_min=cfg.evaluation.get("threshold_min", 0.30),
        threshold_max=cfg.evaluation.get("threshold_max", 0.90),
        threshold_step=cfg.evaluation.get("threshold_step", 0.02),
        beta=cfg.evaluation.get("beta", 0.5),
    )

    report_path = save_evaluation_report(best_thresh, best_f05, sweep_df)
    logger.info(f"Evaluation completed. Report saved at: {report_path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Generates candidate_pairs.tsv and matching_results.tsv for test data."""
    logger.info("Executing subcommand: predict")
    from backend.src.predict.matching_writer import run_prediction_pipeline
    from backend.src.predict.validator import validate_submission_files

    cand_path, match_path = run_prediction_pipeline(sample_n=getattr(args, "sample", None))
    logger.info(f"Predictions generated:\n  - Matching: {match_path}\n  - Candidates: {cand_path}")

    if getattr(args, "validate", False):
        is_valid, issues = validate_submission_files(
            matching_tsv_path=match_path,
            candidate_tsv_path=cand_path,
        )
        if not is_valid:
            logger.error("Submission validation failed.")
            return 1
        logger.info("Submission validation succeeded.")

    return 0


def cmd_package(args: argparse.Namespace) -> int:
    """Builds final submission zip archive per official challenge specification."""
    logger.info("Executing subcommand: package")
    logger.warning("Packaging logic will be attached in Phase G.")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """Executes the full pipeline sequentially."""
    logger.info("Starting end-to-end entity resolution pipeline...")
    stages = [cmd_normalize, cmd_block, cmd_features, cmd_train, cmd_evaluate, cmd_predict, cmd_package]
    for stage in stages:
        res = stage(args)
        if res != 0:
            logger.error(f"Stage {stage.__name__} failed with code {res}")
            return res
    logger.info("End-to-end pipeline completed successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the main CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="business_entity_resolution",
        description="Scalable Business Entity Resolution Pipeline (Macro F0.5 Optimized)",
    )
    
    subparsers = parser.add_subparsers(dest="subcommand", title="Pipeline Subcommands")

    # explore
    p_explore = subparsers.add_parser("explore", help="Explore raw datasets and output data summary")
    p_explore.add_argument("--sample", type=int, default=None, help="Number of sample rows to explore")
    p_explore.set_defaults(func=cmd_explore)

    # normalize
    p_norm = subparsers.add_parser("normalize", help="Run text and address normalization")
    p_norm.add_argument("--sample", type=int, default=None, help="Sample limit for fast local iteration")
    p_norm.set_defaults(func=cmd_normalize)

    # block
    p_block = subparsers.add_parser("block", help="Run multi-stage candidate blocking")
    p_block.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to block")
    p_block.add_argument("--sample", type=int, default=None, help="Sample limit for fast local iteration")
    p_block.add_argument("--top-k", type=int, default=cfg.blocking.get("top_k_embed", 10), help="Top-K nearest neighbors")
    p_block.set_defaults(func=cmd_block)

    # features
    p_feat = subparsers.add_parser("features", help="Extract pairwise similarity features")
    p_feat.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split for features")
    p_feat.add_argument("--sample", type=int, default=None, help="Sample limit for feature extraction")
    p_feat.set_defaults(func=cmd_features)

    # train
    p_train = subparsers.add_parser("train", help="Train LightGBM entity matching classifier")
    p_train.add_argument("--splits", type=int, default=cfg.training.get("n_splits", 5), help="Number of GroupKFold splits")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate validation predictions and sweep F0.5 threshold")
    p_eval.set_defaults(func=cmd_evaluate)

    # predict
    p_pred = subparsers.add_parser("predict", help="Generate submission predictions on test set")
    p_pred.add_argument("--validate", action="store_true", help="Validate output format using official validator")
    p_pred.set_defaults(func=cmd_predict)

    # package
    p_pkg = subparsers.add_parser("package", help="Assemble submission package zip")
    p_pkg.set_defaults(func=cmd_package)

    # run-all
    p_all = subparsers.add_parser("run-all", help="Execute entire pipeline end-to-end")
    p_all.add_argument("--sample", type=int, default=None, help="Sample limit for test run")
    p_all.add_argument("--top-k", type=int, default=cfg.blocking.get("top_k_embed", 10))
    p_all.add_argument("--validate", action="store_true", default=True)
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    cfg.paths.ensure_directories()
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if args.subcommand is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
