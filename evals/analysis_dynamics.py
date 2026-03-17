import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy

from evals.core import EvalContext, get_cached_ratemaps
from evals.analysis_connectivity import _compute_phase_order

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
def run_projection_onto_sliding_mode(ctx: EvalContext, res: int = 50, n_avg: int = 100) -> str:
    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx, res=res, n_avg=n_avg, ng=ctx.options.Ng)

    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    M = ctx.model.RNN.weight_ih_l0.detach().cpu().numpy()

    theta = 0
    v = np.stack([np.cos(theta), np.sin(theta)])

    G = (J @ rate_map) > 0
    Mv = M @ v
    GMv = G * Mv[:, None]
    JGMv = J @ GMv

    shift = 5
    SM = rate_map.reshape(ctx.options.Ng, res, res)
    SM = np.roll(SM, shift=-shift, axis=1) - np.roll(SM, shift=shift, axis=1)
    SM = SM.reshape(ctx.options.Ng, -1)

    proj = np.sum(JGMv * SM, axis=0)
    denom = np.sqrt(np.sum(SM**2, axis=0) * np.sum(JGMv**2, axis=0)) + 1e-8
    cos_ang = proj / denom

    crop = 10
    idxs1, idxs2 = np.mgrid[crop : res - crop, crop : res - crop]
    idxs = np.ravel_multi_index((idxs1, idxs2), (res, res)).ravel()

    plt.figure(figsize=(8, 5))
    plt.hist(cos_ang[idxs], bins=20, color="slategrey")
    plt.xlabel("Projection onto sliding mode")
    plt.ylabel("Count")
    out_path = os.path.join(ctx.save_dir, "projection_onto_sliding_mode.png")
    _savefig(out_path)
    return out_path

def run_thetas_plot(ctx: EvalContext, res: int = 50, n_avg: int = 100) -> str:
    n = np.sqrt(ctx.options.Ng).astype(int)
    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx, res=res, n_avg=n_avg, ng=ctx.options.Ng)
    total_order, _phases = _compute_phase_order(ctx, rate_map, res)

    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    M = ctx.model.RNN.weight_ih_l0.detach().cpu().numpy()

    G = (J @ rate_map) > 0

    thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    clock_idxs = np.roll([0, 1, 2, 5, 8, 7, 6, 3], 1)
    idx = np.ravel_multi_index((20, 20), (res, res))

    plt.figure(figsize=(9, 9))
    for i, theta in enumerate(thetas):
        v = np.stack([np.cos(theta), np.sin(theta)])
        Mv = M @ v
        GMv = G[:, idx] * Mv
        JGMv = J @ GMv

        plt.subplot(3, 3, clock_idxs[i] + 1)
        im = JGMv[total_order].reshape(n, n)
        im = scipy.ndimage.gaussian_filter(im, (3, 3))
        plt.imshow(im, cmap="RdBu")
        plt.axis("off")

    out_path = os.path.join(ctx.save_dir, "thetas_plot.png")
    _savefig(out_path)
    return out_path

def run_torus_construction(ctx: EvalContext, res: int = 50, n_avg: int = 100) -> str:
    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx, res=res, n_avg=n_avg, ng=ctx.options.Ng)

    X_centered = rate_map - rate_map.mean(-1, keepdims=True)
    X_centered -= X_centered.mean(-1, keepdims=True)
    Ua, _S, _V = scipy.linalg.svd(X_centered)
    rm_embed = Ua.T @ rate_map

    k1 = [3, 0]
    k2 = [2, 2.5]
    k3 = [-1, 2.3]

    freq = 1
    x = np.mgrid[:res, :res] * 2 * np.pi / res
    x = x.reshape(2, -1)
    k = freq * np.stack([k1, k2, k3])
    X = np.concatenate([np.cos(k.dot(x)), np.sin(k.dot(x))], axis=0)

    cmaps = ["Blues", "Oranges", "Greens"]
    crop = 0
    idxs1, idxs2 = np.mgrid[crop : res - crop, crop : res - crop]
    idxs = np.ravel_multi_index((idxs1, idxs2), (res, res)).ravel()

    RM = rm_embed[:10, idxs]
    X_crop = X[:, idxs]
    R = np.linalg.inv(RM.dot(RM.T)).dot(RM).dot(X_crop.T).T

    U, s, _V = np.linalg.svd(R)
    S2 = U.dot(np.diag(1.0 / s)).dot(U.T)
    R = S2.dot(R)

    plt.figure(figsize=(12, 4))
    for i in range(R.shape[0] // 2):
        plt.subplot(1, 3, i + 1)
        plt.scatter(R.dot(RM)[i], R.dot(RM)[i + 3], c=X[i][idxs], cmap=cmaps[i], s=10)
        plt.axis("off")

    out_path = os.path.join(ctx.save_dir, "ring_fit.png")
    _savefig(out_path)
    return out_path

def run_all_dynamics(ctx: EvalContext) -> Dict[str, str]:
    return {
        "projection_onto_sliding_mode": run_projection_onto_sliding_mode(ctx),
        "thetas_plot": run_thetas_plot(ctx),
        "torus_construction": run_torus_construction(ctx),
    }
