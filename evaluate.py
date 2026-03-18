import argparse
from grid_pattern_formation.evals import (
    build_eval_context, 
    run_eval_by_name
)

parser = argparse.ArgumentParser(description="evaluation for trained models")

parser.add_argument(
    "--eval-name",
    type=str,
    default="all",
)
parser.add_argument(
    "--checkpoint-path",
    type=str,
    required=True,
    help="path to trained model checkpoint",
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="path to config yaml",
)
parser.add_argument(
    "--results-root",
    type=str,
    default="results",
    help="root results directory",
)

args = parser.parse_args()

eval_context = build_eval_context(
    checkpoint_path=args.checkpoint_path,
    config_path=args.config,
    results_root=args.results_root,
)

print(f"Will save results to: {eval_context.save_dir}")

run_eval_by_name(
    eval_context=eval_context,
    eval_name=args.eval_name,
)