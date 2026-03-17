import csv
import os
from typing import Dict, List

import numpy as np
import torch

from evals.analysis_core import compute_grid_scores
from evals.core import EvalContext

def _append_row(csv_path: str, fieldnames: List[str], row: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return csv_path

def _get_top25_indices_and_scores(ctx: EvalContext):
    scores = compute_grid_scores(ctx)
    score_60 = np.asarray(scores["score_60"])
    top_25_indices = np.argsort(score_60)[-25:][::-1]
    top_25_scores = score_60[top_25_indices]
    return top_25_indices, top_25_scores, scores

def calculate_firing_sparsity(firing_rates):
    rates_flat = np.asarray(firing_rates).flatten()

    p40 = np.percentile(rates_flat, 40)
    p60 = np.percentile(rates_flat, 60)

    is_between = (rates_flat >= p40) & (rates_flat <= p60)
    sparsity = np.mean(is_between)
    return sparsity

def run_grid_scores_csv(ctx: EvalContext) -> str:
    _top_25_indices, top_25_scores, _scores = _get_top25_indices_and_scores(ctx)

    row = {
        "data_source": ctx.model_name,
        "grid_score_mean": float(np.mean(top_25_scores)),
        "grid_score_sds": float(np.std(top_25_scores, ddof=1)) if len(top_25_scores) > 1 else 0.0,
    }

    csv_path = os.path.join(ctx.results_dir, "grid_scores.csv")
    return _append_row(csv_path, ["data_source", "grid_score_mean", "grid_score_sds"], row)

def run_sparsities_csv(ctx: EvalContext) -> str:
    top_25_indices, _top_25_scores, scores = _get_top25_indices_and_scores(ctx)
    rate_map_lores = np.asarray(scores["rate_map_lores"])

    ratemaps = rate_map_lores.copy()
    ratemaps = ratemaps[top_25_indices]
    sparsities = [calculate_firing_sparsity(ratemaps[i]) for i in range(ratemaps.shape[0])]

    row = {
        "data_source": ctx.model_name,
        "sparsity_mean": float(np.mean(sparsities)),
        "sparsity_sds": float(np.std(sparsities, ddof=1)) if len(sparsities) > 1 else 0.0,
    }

    csv_path = os.path.join(ctx.results_dir, "sparsities.csv")
    return _append_row(csv_path, ["data_source", "sparsity_mean", "sparsity_sds"], row)

def run_trajectory_decodings_csv(ctx: EvalContext) -> str:
    inputs, pos, _pc_outputs = ctx.trajectory_generator.get_test_batch()
    pred_pos = ctx.place_cells.get_nearest_cell_pos(ctx.model.predict(inputs))

    err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()

    row = {
        "data_source": ctx.model_name,
        "error": float(err.detach().cpu().item()),
    }

    csv_path = os.path.join(ctx.results_dir, "trajectory_decodings.csv")
    return _append_row(csv_path, ["data_source", "error"], row)
