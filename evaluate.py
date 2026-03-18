import argparse

from grid_pattern_formation.evals.core import build_context

from grid_pattern_formation.evals.analysis_connectivity import run_eigenvalues, run_jmean_sorted, run_jmean_unsorted, run_single_neuron_connectivity_sorted, run_single_neuron_connectivity_unsorted, run_sorted_connectivity_eigvs, run_unsorted_connectivity_eigvs
from grid_pattern_formation.evals.analysis_core import run_grid_score_histogram, run_grid_score_panels, run_manifold_distance, run_place_cell_outputs, run_trajectory_decoding
from grid_pattern_formation.evals.analysis_dynamics import run_projection_onto_sliding_mode, run_thetas_plot, run_torus_construction
from grid_pattern_formation.evals.analysis_tables import run_grid_scores_csv, run_sparsities_csv, run_trajectory_decodings_csv

ANALYSIS_RUNNERS = {
    "trajectory_decoding": run_trajectory_decoding,
    "place_cell_outputs": run_place_cell_outputs,
    "grid_score_panels": run_grid_score_panels,
    "grid_score_histogram": run_grid_score_histogram,
    "manifold_distance": run_manifold_distance,
    "eigenvalues": run_eigenvalues,
    "unsorted_connectivity": run_unsorted_connectivity_eigvs,
    "sorted_connectivity": run_sorted_connectivity_eigvs,
    "jmean_unsorted": run_jmean_unsorted,
    "jmean_sorted": run_jmean_sorted,
    "single_neuron_connectivity_unsorted": run_single_neuron_connectivity_unsorted,
    "single_neuron_connectivity_sorted": run_single_neuron_connectivity_sorted,
    "projection_onto_sliding_mode": run_projection_onto_sliding_mode,
    "thetas_plot": run_thetas_plot,
    "torus_construction": run_torus_construction,
    "grid_scores_csv": run_grid_scores_csv,
    "sparsities_csv": run_sparsities_csv,
}


def main():
    parser = argparse.ArgumentParser(description="evaluation for trained models")

    parser.add_argument(
        "--eval-name",
        type=str,
        default="all",
        help=(
            "what to run, you can enter all or a comma-seperated list"
            f"available: {', '.join(ANALYSIS_RUNNERS.keys())}"
        ),
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

    eval_context = build_context(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        results_root=args.results_root,
    )

    print(f"Will save results to: {eval_context.save_dir}")

    outputs = {}

    if args.eval_name == "all":
        selected_runs = list(ANALYSIS_RUNNERS.keys())
    else:
        selected_runs = [args.eval_name]

    for eval_name in selected_runs:
        print(f"Running {eval_name}")
        outputs[eval_name] = ANALYSIS_RUNNERS[eval_name](eval_context)


if __name__ == "__main__":
    main()
