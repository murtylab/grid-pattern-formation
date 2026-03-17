import argparse

from evals.analysis_connectivity import (
    run_eigenvalues,
    run_jmean_sorted,
    run_jmean_unsorted,
    run_sorted_connectivity_eigvs,
    run_single_neuron_connectivity_sorted,
    run_single_neuron_connectivity_unsorted,
    run_unsorted_connectivity_eigvs,
)
from evals.analysis_core import (
    run_grid_score_histogram,
    run_grid_score_panels,
    run_manifold_distance,
    run_place_cell_outputs,
    run_trajectory_decoding,
)
from evals.analysis_dynamics import (
    run_projection_onto_sliding_mode,
    run_thetas_plot,
    run_torus_construction,
)
from evals.analysis_tables import (
    run_grid_scores_csv,
    run_sparsities_csv,
    run_trajectory_decodings_csv,
)
from evals.core import build_context

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
    "trajectory_decodings_csv": run_trajectory_decodings_csv,
}

def _parse_runs(run_arg: str):
    if run_arg == "all":
        return list(ANALYSIS_RUNNERS.keys())
    return [item.strip() for item in run_arg.split(",") if item.strip()]

def main():
    parser = argparse.ArgumentParser(description="evaluation for trained models")

    parser.add_argument(
        "--run",
        type=str,
        default="all",
        help=(
            "what to run, you can enter all or a comma-seperated list"
            f"available: {', '.join(ANALYSIS_RUNNERS.keys())}"
        ),
    )
    parser.add_argument(
        "--model",
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
        "--model_name",
        type=str,
        required=True,
        help="model name used for saving outputs in results/<model_name>",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="root results directory",
    )

    args = parser.parse_args()

    ctx = build_context(
        model_path=args.model,
        config_path=args.config,
        model_name=args.model_name,
        results_root=args.results_root,
    )

    print(f"Results root: {ctx.results_dir}")
    print(f"Model output directory: {ctx.save_dir}")

    selected_runs = _parse_runs(args.run)
    unknown = [name for name in selected_runs if name not in ANALYSIS_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown analysis names: {unknown}")

    outputs = {}
    for run_name in selected_runs:
        print(f"Running {run_name}")
        outputs[run_name] = ANALYSIS_RUNNERS[run_name](ctx)

if __name__ == "__main__":
    main()
