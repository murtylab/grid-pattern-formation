import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..utils.visualize import plot_ratemaps

from .core import EvalContext, get_cached_ratemaps
from .grid_scores import GridScorer

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def run_trajectory_decoding(eval_context: EvalContext) -> str:
    inputs, pos, _pc_outputs = eval_context.trajectory_generator.get_test_batch()
    pos = pos.detach().cpu().numpy()
    pred_pos = eval_context.place_cells.get_nearest_cell_pos(eval_context.model.predict(inputs)).detach().cpu().numpy()
    us = eval_context.place_cells.us.detach().cpu().numpy()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    for i in range(5):
        ax.plot(pos[:, i, 0], pos[:, i, 1], c="black", linewidth=2)
        ax.plot(pred_pos[:, i, 0], pred_pos[:, i, 1], ".-", c="C1")
    ax.scatter(us[:, 0], us[:, 1], s=20, alpha=0.5, c="lightgrey")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-eval_context.options.box_width / 2, eval_context.options.box_width / 2])
    ax.set_ylim([-eval_context.options.box_height / 2, eval_context.options.box_height / 2])

    out_path = os.path.join(eval_context.save_dir, "trajectory_decoding.png")
    _savefig(out_path)
    return out_path

def run_place_cell_outputs(eval_context: EvalContext, n_examples: int = 8) -> str:
    inputs, _pos, pc_outputs = eval_context.trajectory_generator.get_test_batch()
    preds = eval_context.model.predict(inputs)

    preds = preds.reshape(-1, eval_context.options.Np).detach().cpu()
    pc_outputs = eval_context.model.softmax(pc_outputs).reshape(-1, eval_context.options.Np).detach().cpu()

    pc_pred = eval_context.place_cells.grid_pc(preds[:100])
    pc_true = eval_context.place_cells.grid_pc(pc_outputs[:100])

    plt.figure(figsize=(16, 4))
    for i in range(n_examples):
        plt.subplot(2, n_examples, i + n_examples + 1)
        plt.imshow(pc_pred[2 * i], cmap="jet")
        if i == 0:
            plt.ylabel("Predicted")
        plt.axis("off")

    for i in range(n_examples):
        plt.subplot(2, n_examples, i + 1)
        plt.imshow(pc_true[2 * i], cmap="jet", interpolation="gaussian")
        if i == 0:
            plt.ylabel("True")
        plt.axis("off")

    plt.suptitle("Place Cell Outputs", fontsize=16)
    out_path = os.path.join(eval_context.save_dir, "place_cell_outputs.png")
    _savefig(out_path)
    return out_path

def compute_grid_scores(eval_context: EvalContext, lo_res: int = 20, n_avg: int = 100) -> Dict[str, np.ndarray]:
    cache_key = ("grid_scores", lo_res, n_avg)
    if cache_key in eval_context.cache:
        return eval_context.cache[cache_key]

    _activations_lores, rate_map_lores, _g, _pos = get_cached_ratemaps(
        eval_context,
        res=lo_res,
        n_avg=n_avg,
        ng=eval_context.options.Ng,
    )

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = (
        (-eval_context.options.box_width / 2, eval_context.options.box_width / 2),
        (-eval_context.options.box_height / 2, eval_context.options.box_height / 2),
    )
    scorer = GridScorer(lo_res, coord_range, zip(starts, ends.tolist()))

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
        *[scorer.get_scores(rm.reshape(lo_res, lo_res)) for rm in tqdm(rate_map_lores, desc="Grid scores")]
    )

    out = {
        "score_60": np.array(score_60),
        "score_90": np.array(score_90),
        "max_60_mask": np.array(max_60_mask, dtype=object),
        "max_90_mask": np.array(max_90_mask, dtype=object),
        "sac": np.array(sac, dtype=object),
        "max_60_ind": np.array(max_60_ind),
        "rate_map_lores": rate_map_lores,
    }
    eval_context.cache[cache_key] = out
    return out

def run_grid_score_panels(eval_context: EvalContext, res: int = 50, n_avg: int = 100, n_plot: int = 25) -> Dict[str, str]:
    from .core import compute_ratemaps
    activations, _rate_map, _g, _pos = compute_ratemaps(
            model=eval_context.model,
            trajectory_generator=eval_context.trajectory_generator,
            options=eval_context.options,
            res=res,
            n_avg=n_avg,
            Ng=eval_context.options.Ng,
        )
    scores = compute_grid_scores(eval_context)
    score_60 = scores["score_60"]

    idxs = np.flip(np.argsort(score_60))
    ng = eval_context.options.Ng

    outputs = {}

    plt.figure(figsize=(8, 8))
    rm_fig = plot_ratemaps(activations[idxs], n_plot, smooth=True, width=5)
    plt.imshow(rm_fig)
    plt.suptitle(
        f"High grid scores: {np.round(score_60[idxs[0]], 2)} to {np.round(score_60[idxs[n_plot]], 2)}",
        fontsize=14,
    )
    plt.axis("off")
    out_high = os.path.join(eval_context.save_dir, "grid_scores_high.png")
    _savefig(out_high)
    outputs["high"] = out_high

    plt.figure(figsize=(8, 8))
    rm_fig = plot_ratemaps(activations[idxs[ng // 4 :]], n_plot, smooth=True, width=5)
    plt.imshow(rm_fig)
    plt.suptitle(
        f"Medium grid scores: {np.round(score_60[idxs[ng//2]], 2)} to {np.round(score_60[idxs[ng//2 + n_plot]], 2)}",
        fontsize=14,
    )
    plt.axis("off")
    out_med = os.path.join(eval_context.save_dir, "grid_scores_medium.png")
    _savefig(out_med)
    outputs["medium"] = out_med

    plt.figure(figsize=(8, 8))
    rm_fig = plot_ratemaps(activations[np.flip(idxs)], n_plot, smooth=True, width=5)
    plt.imshow(rm_fig)
    plt.suptitle(
        f"Low grid scores: {np.round(score_60[idxs[-n_plot]], 2)} to {np.round(score_60[idxs[-1]], 2)}",
        fontsize=14,
    )
    plt.axis("off")
    out_low = os.path.join(eval_context.save_dir, "grid_scores_low.png")
    _savefig(out_low)
    outputs["low"] = out_low

    return outputs

def run_grid_score_histogram(eval_context: EvalContext) -> str:
    scores = compute_grid_scores(eval_context)
    score_60 = scores["score_60"]

    plt.figure(figsize=(6, 4))
    plt.hist(score_60, range=(-1, 2.5), bins=15, color="slategrey")
    plt.xlabel("Grid score")
    plt.ylabel("Count")
    out_path = os.path.join(eval_context.save_dir, "grid_score_histogram.png")
    _savefig(out_path)
    return out_path

def run_manifold_distance(eval_context: EvalContext, res: int = 50, n_avg: int = 100) -> Dict[str, str]:
    _activations, rate_map, _g, _pos = get_cached_ratemaps(eval_context, res=res, n_avg=n_avg, ng=eval_context.options.Ng)
    scores = compute_grid_scores(eval_context)
    score_60 = scores["score_60"]

    outputs = {}

    origins = np.stack(np.mgrid[:3, :3] - 1) * res // 4 + res // 2

    fig = plt.figure(figsize=(8, 8))
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 3 * i + j + 1)
            origin_idx = np.ravel_multi_index((origins[0, i, j], origins[1, i, j]), (res, res))
            r0 = rate_map[:, origin_idx, None]
            dists = np.linalg.norm(r0 - rate_map, axis=0)
            im = plt.imshow(
                dists.reshape(res, res) / np.maximum(np.max(dists), 1e-8),
                cmap="viridis_r",
                interpolation="gaussian",
            )
            plt.axis("off")
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.86, 0.12, 0.02, 0.74])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.locator_params(nbins=3)
    cbar.outline.set_visible(False)
    out_all = os.path.join(eval_context.save_dir, "manifold_distance_all_cells.png")
    plt.savefig(out_all, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs["all_cells"] = out_all

    n_grid_cells = min(500, eval_context.options.Ng)
    grid_sort = np.flip(np.argsort(score_60))

    fig = plt.figure(figsize=(8, 8))
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 3 * i + j + 1)
            origin_idx = np.ravel_multi_index((origins[0, i, j], origins[1, i, j]), (res, res))
            r0 = rate_map[grid_sort[:n_grid_cells], origin_idx, None]
            dists = np.linalg.norm(r0 - rate_map[grid_sort[:n_grid_cells]], axis=0)
            im = plt.imshow(
                dists.reshape(res, res) / np.maximum(np.max(dists), 1e-8),
                cmap="viridis_r",
                interpolation="gaussian",
            )
            plt.axis("off")
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.86, 0.12, 0.02, 0.74])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.locator_params(nbins=3)
    cbar.outline.set_visible(False)
    out_top = os.path.join(eval_context.save_dir, "manifold_distance_top500_cells.png")
    plt.savefig(out_top, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs["top_500"] = out_top

    return outputs
