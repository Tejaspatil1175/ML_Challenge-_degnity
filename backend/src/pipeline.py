"""Business Entity Resolution - Master Command Line Interface.

Orchestrates all pipeline stages from exploration to submission packaging.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from backend.src.config import cfg
from backend.src.data.explore import generate_exploration_report
from backend.src.logger import get_logger

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
    logger.info(f"Sample size: {args.sample}")
    logger.warning("Normalization logic will be attached in Phase C.")
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Executes rule-based and semantic dense vector candidate blocking."""
    logger.info("Executing subcommand: block")
    logger.info(f"Top-K: {args.top_k}, Sample size: {args.sample}")
    logger.warning("Blocking logic will be attached in Phase D.")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Extracts pairwise string, token, semantic, and spatial similarity features."""
    logger.info("Executing subcommand: features")
    logger.warning("Features extraction logic will be attached in Phase E.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Trains LightGBM classifier using GroupKFold leakage-safe splitting."""
    logger.info("Executing subcommand: train")
    logger.warning("Training logic will be attached in Phase F.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Performs threshold sweep and Macro F0.5 evaluation."""
    logger.info("Executing subcommand: evaluate")
    logger.warning("Evaluation logic will be attached in Phase F.")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Generates candidate_pairs.tsv and matching_results.tsv for test data."""
    logger.info("Executing subcommand: predict")
    logger.info(f"Validate output: {args.validate}")
    logger.warning("Prediction logic will be attached in Phase G.")
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
    p_block.add_argument("--sample", type=int, default=None, help="Sample limit for fast local iteration")
    p_block.add_argument("--top-k", type=int, default=cfg.blocking.get("top_k_embed", 10), help="Top-K nearest neighbors")
    p_block.set_defaults(func=cmd_block)

    # features
    p_feat = subparsers.add_parser("features", help="Extract pairwise similarity features")
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
