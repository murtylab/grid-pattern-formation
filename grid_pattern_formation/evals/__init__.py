from .analysis_connectivity import (
    run_eigenvalues,
    run_jmean_sorted,
    run_jmean_unsorted,
    run_single_neuron_connectivity_sorted,
    run_single_neuron_connectivity_unsorted,
    run_sorted_connectivity_eigvs,
    run_unsorted_connectivity_eigvs,
)
from .analysis_core import (
    run_grid_score_histogram,
    run_grid_score_panels,
    run_manifold_distance,
    run_place_cell_outputs,
    run_trajectory_decoding,
)
from .analysis_dynamics import (
    run_projection_onto_sliding_mode,
    run_thetas_plot,
    run_torus_construction,
)
from .analysis_tables import (
    run_grid_scores_csv,
    run_sparsities_csv,
)
from .core import EvalContext, build_eval_context

analysis_map = {
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

def run_eval_by_name(
    eval_context: EvalContext,
    eval_name: str = "all",
):
    assert isinstance(eval_context, EvalContext), f"eval_context must be an instance of EvalContext, but got type {type(eval_context)}"

    if eval_name == "all":
        outputs = {}
        for name, func in analysis_map.items():
            outputs[name] = func(eval_context)
        return outputs
    else:
        assert eval_name in analysis_map, f"eval_name must be one of {list(analysis_map.keys())}, but got {eval_name}"
        return analysis_map[eval_name](eval_context)