import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy

from .core import EvalContext, get_cached_ratemaps

from ..utils.two_d_sort import get_2d_sort

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def _compute_phase_order(ctx: EvalContext, rate_map: np.ndarray, res: int) -> Tuple[np.ndarray, np.ndarray]:
    cache_key = ("phase_order", res)
    if cache_key in ctx.cache:
        return ctx.cache[cache_key]

    ng = ctx.options.Ng
    n = np.sqrt(ng).astype(int)

    rm_fft_real = np.zeros([ng, res, res])
    rm_fft_imag = np.zeros([ng, res, res])
    for i in range(ng):
        rm_fft_real[i] = np.real(np.fft.fft2(rate_map[i].reshape([res, res])))
        rm_fft_imag[i] = np.imag(np.fft.fft2(rate_map[i].reshape([res, res])))
    rm_fft = rm_fft_real + 1j * rm_fft_imag

    k1 = [3, 0]
    k2 = [2, 3]
    k3 = [-1, 3]
    ks = np.array([k1, k2, k3, k1, k1, k1]).astype(int)
    modes = np.stack([rm_fft[:, k[0], k[1]] for k in ks])
    phases = [np.angle(mode) for mode in modes]

    width = n
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(width))
    x_grid = x_grid * 2 * np.pi / width
    y_grid = y_grid * 2 * np.pi / width

    s1 = np.zeros(phases[0].shape)
    s2 = np.zeros(phases[0].shape)
    fac = np.sqrt(3) / 2

    for i in range(ng):
        penalty_1 = np.cos(x_grid - phases[0][i] / fac)
        penalty_2 = np.cos(y_grid - phases[2][i] / fac)
        penalty_3 = np.cos((x_grid + y_grid) - phases[1][i] / fac)
        ind = np.argmax(penalty_1 + penalty_2 + penalty_3 + np.random.randn() / 100)
        s1[i], s2[i] = np.unravel_index([ind], penalty_1.shape)

    total_order = get_2d_sort(s1, s2)
    out = (total_order, np.array(phases))
    ctx.cache[cache_key] = out
    return out

def _compute_jmean(Jsort: np.ndarray, n: int):
    J_square = np.reshape(Jsort, (n, n, n, n))
    Jmean = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Jmean += np.roll(np.roll(J_square[i, j], -i, axis=0), -j, axis=1)

    Jmean[0, 0] = np.max(Jmean[1:, 1:])
    Jmean = np.roll(np.roll(Jmean, n // 2, axis=0), n // 2, axis=1)

    A = np.asarray([[2, 1], [0, np.sqrt(3)]]) / 2
    Ainv = np.linalg.inv(A)
    im = scipy.ndimage.affine_transform(Jmean, Ainv, mode="wrap")
    imroll = np.roll(np.roll(im, -n // 4, axis=0), 0, axis=1)
    return Jmean, imroll

def run_eigenvalues(ctx: EvalContext) -> str:
    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    eigs, _eigvs = np.linalg.eig(J)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(np.real(eigs), np.imag(eigs), c="black", s=20)
    ax.scatter(np.real(eigs[:9]), np.imag(eigs[:9]), c="C1", s=20)
    circle = plt.Circle((0, 0), 1, color="tan", fill=False, linestyle="dashed", linewidth=2)
    ax.add_artist(circle)
    ax.set_xlim([-1.1, 2.5])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect("equal", adjustable="box")
    ax.locator_params(nbins=4)

    out_path = os.path.join(ctx.save_dir, "final_eigs.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _plot_connectivity_eigvs_panel(eigvs_rot: np.ndarray, n: int, title: str, out_path: str) -> str:
    A = np.asarray([[2, 1], [0, np.sqrt(3)]]) / 2
    Ainv = np.linalg.inv(A)

    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16)
    idxs = [1, 3, 5, 4, 0, 2]
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        im = np.real(eigvs_rot[idxs[i]]).reshape(n, n)
        im = np.roll(np.roll(im, n // 4, axis=1), -n // 4, axis=0)
        im = scipy.ndimage.affine_transform(im, Ainv, mode="wrap")
        if i in (1, 4):
            im = np.roll(im, -n // 3, axis=0)
        im = scipy.ndimage.gaussian_filter(im, sigma=(2, 2))
        plt.imshow(im, cmap="coolwarm")
        plt.axis("off")

    _savefig(out_path)
    return out_path

def run_unsorted_connectivity_eigvs(ctx: EvalContext) -> str:
    n = np.sqrt(ctx.options.Ng).astype(int)

    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    _eigenvalues, eigenvectors = np.linalg.eig(J)
    eigvs_rot = eigenvectors.T

    out_path = os.path.join(ctx.save_dir, "eigvs2_unsorted.png")
    return _plot_connectivity_eigvs_panel(
        eigvs_rot=eigvs_rot,
        n=n,
        title="Unsorted Connectivity",
        out_path=out_path,
    )

def run_sorted_connectivity_eigvs(ctx: EvalContext, res: int = 50, n_avg: int = 100) -> str:
    n = np.sqrt(ctx.options.Ng).astype(int)
    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx, res=res, n_avg=n_avg, ng=ctx.options.Ng)
    total_order, _phases = _compute_phase_order(ctx, rate_map, res)

    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    Jsort = J[total_order][:, total_order]
    _eigenvalues, eigenvectors = np.linalg.eig(Jsort)
    eigvs_rot = eigenvectors.T

    out_path = os.path.join(ctx.save_dir, "eigvs2_sorted.png")
    return _plot_connectivity_eigvs_panel(
        eigvs_rot=eigvs_rot,
        n=n,
        title="Sorted Connectivity",
        out_path=out_path,
    )

def run_jmean_unsorted(ctx: EvalContext) -> str:
    n = np.sqrt(ctx.options.Ng).astype(int)
    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    _Jmean, imroll = _compute_jmean(J, n)

    limit = abs(imroll.min())
    plt.figure(figsize=(5, 5))
    img = plt.imshow(imroll, cmap="coolwarm", vmin=-limit, vmax=limit)
    plt.colorbar(img, fraction=0.046, pad=0.04, extend="max")
    plt.title("J (unsorted)")
    plt.axis("off")

    out_path = os.path.join(ctx.save_dir, "jmean_unsorted.png")
    _savefig(out_path)
    return out_path

def run_jmean_sorted(ctx: EvalContext, res: int = 50, n_avg: int = 100) -> str:
    n = np.sqrt(ctx.options.Ng).astype(int)
    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx, res=res, n_avg=n_avg, ng=ctx.options.Ng)
    total_order, _phases = _compute_phase_order(ctx, rate_map, res)

    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    Jsort = J[total_order][:, total_order]
    _Jmean, imroll = _compute_jmean(Jsort, n)

    limit = abs(imroll.min())
    plt.figure(figsize=(5, 5))
    img = plt.imshow(imroll, cmap="coolwarm", vmin=-limit, vmax=limit)
    plt.colorbar(img, fraction=0.046, pad=0.04, extend="max")
    plt.title("J (sorted)")
    plt.axis("off")

    out_path = os.path.join(ctx.save_dir, "jmean_sorted.png")
    _savefig(out_path)
    return out_path

def _plot_single_neuron_connectivity(J_matrix: np.ndarray, n: int, out_path: str) -> str:
    J_square = np.reshape(J_matrix, (n, n, n, n))

    A = np.asarray([[2, 1], [0, np.sqrt(3)]]) / 2
    Ainv = np.linalg.inv(A)

    example_neurons = [(38, 20), (3, 14), (55, 7)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, (i, j) in enumerate(example_neurons):
        single_neuron = J_square[i, j]
        im_single = scipy.ndimage.affine_transform(single_neuron, Ainv, mode="wrap")
        imroll_single = np.roll(np.roll(im_single, -n // 4, axis=0), 0, axis=1)

        img = axes[idx].imshow(imroll_single, cmap="coolwarm", vmin=imroll_single.min(), vmax=imroll_single.max())
        fig.colorbar(img, ax=axes[idx], fraction=0.046, pad=0.04)

        marker_y = (i - n // 4) % single_neuron.shape[0]
        marker_x = j
        axes[idx].plot(marker_x, marker_y, color="black", marker="x", markersize=10, markeredgewidth=2)
        axes[idx].set_title(f"Neuron ({i},{j})")
        axes[idx].axis("off")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def run_single_neuron_connectivity_unsorted(ctx: EvalContext) -> str:
    n = np.sqrt(ctx.options.Ng).astype(int)
    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    out_path = os.path.join(ctx.save_dir, "single_neuron_connectivity_unsorted.png")
    return _plot_single_neuron_connectivity(J, n, out_path)

def run_single_neuron_connectivity_sorted(ctx: EvalContext, res: int = 50, n_avg: int = 100) -> str:
    n = np.sqrt(ctx.options.Ng).astype(int)
    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx, res=res, n_avg=n_avg, ng=ctx.options.Ng)
    total_order, _phases = _compute_phase_order(ctx, rate_map, res)

    J = ctx.model.RNN.weight_hh_l0.detach().cpu().numpy().T
    Jsort = J[total_order][:, total_order]

    out_path = os.path.join(ctx.save_dir, "single_neuron_connectivity_sorted.png")
    return _plot_single_neuron_connectivity(Jsort, n, out_path)
