import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm
from .core import EvalContext, get_cached_ratemaps
from .analysis_connectivity import _compute_phase_order

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
    

def get_fourier_spectrum(
    eval_context: EvalContext,
    res: int = 50,
    n_avg: int = 100,
):
    # Fourier transform 
    Ng = eval_context.options.Ng
    rm_fft_real = np.zeros([Ng,res,res])
    rm_fft_imag = np.zeros([Ng,res,res])

    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx=eval_context, res=res, n_avg=n_avg, ng=eval_context.options.Ng)

    for i in tqdm(range(Ng)):
        rm_fft_real[i] = np.real(np.fft.fft2(rate_map[i].reshape([res,res])))
        rm_fft_imag[i] = np.imag(np.fft.fft2(rate_map[i].reshape([res,res])))
        
    rm_fft = rm_fft_real + 1j * rm_fft_imag
    
    im = (np.real(rm_fft)**2).mean(0)
    im[0,0] = 0
    return im

def run_neural_sheet(
    eval_context: EvalContext,
    res=50,
    n_avg=100,
):
    im = get_fourier_spectrum(eval_context=eval_context, res=res, n_avg=n_avg)

    width = 6
    idxs = np.arange(-width+1, width)
    x2, y2 = np.meshgrid(np.arange(2*width-1), np.arange(2*width-1))

    plt.scatter(x2,y2,c=im[idxs][:,idxs], s=300, cmap='Oranges')
    plt.axis('equal')
    plt.axis('off')
    plt.title('Mean power')
    out_path = os.path.join(eval_context.save_dir, "neural_sheet.png")
    _savefig(out_path)

    peak_coordinates = find_peak_coordinates(im, n_peaks=3)
    print(f"Found peak coordinates: {peak_coordinates}")
    return out_path

def find_peak_coordinates(
    im: np.ndarray,
    n_peaks: int = 3,
) -> list[list[float, float]]:
    from scipy.ndimage import maximum_filter
    
    res = im.shape[0]
    im = im.copy()
    im[0, 0] = 0
    
    local_max = (im == maximum_filter(im, size=3))
    local_max[0, 0] = False
    
    ys, xs = np.where(local_max)
    powers = im[ys, xs]
    order = np.argsort(powers)[::-1]
    
    def to_freq(idx, res):
        if idx > res // 2:
            return idx - res
        return idx
    
    def refine(im, ix, iy, res):
        if 0 < ix < res - 1:
            left = im[iy, ix - 1]
            center = im[iy, ix]
            right = im[iy, ix + 1]
            denom = left - 2 * center + right
            dx = 0.5 * (left - right) / denom if denom != 0 else 0.0
        else:
            dx = 0.0
            
        if 0 < iy < res - 1:
            above = im[iy - 1, ix]
            center = im[iy, ix]
            below = im[iy + 1, ix]
            denom = above - 2 * center + below
            dy = 0.5 * (above - below) / denom if denom != 0 else 0.0
        else:
            dy = 0.0
        
        kx = to_freq(ix, res) + dx
        ky = to_freq(iy, res) + dy
        return [float(kx), float(ky)]
    
    def normalize_sign(kx, ky):
        """Flip to positive half: prefer kx > 0, or if kx ~ 0, prefer ky > 0."""
        if kx < -0.5 or (abs(kx) < 0.5 and ky < -0.5):
            return -kx, -ky
        return kx, ky
    
    peaks = []
    used_freqs = []
    
    for idx in order:
        if len(peaks) >= n_peaks:
            break
            
        ix, iy = int(xs[idx]), int(ys[idx])
        kx_int = to_freq(ix, res)
        ky_int = to_freq(iy, res)
        
        # Normalize to positive half for dedup
        nkx, nky = normalize_sign(kx_int, ky_int)
        
        is_duplicate = False
        for ukx, uky in used_freqs:
            if nkx == ukx and nky == uky:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        
        peak = refine(im, ix, iy, res)
        # Normalize the refined peak too
        pk_x, pk_y = normalize_sign(peak[0], peak[1])
        peaks.append([pk_x, pk_y])
        used_freqs.append((nkx, nky))
    
    return peaks

def run_torus_construction(ctx: EvalContext, res: int = 50, n_avg: int = 100) -> str:
    _activations, rate_map, _g, _pos = get_cached_ratemaps(ctx, res=res, n_avg=n_avg, ng=ctx.options.Ng)

    im = get_fourier_spectrum(ctx, res=res, n_avg=n_avg)
    peaks = find_peak_coordinates(im, n_peaks=3)
    k1, k2, k3 = peaks

    ## round off the coordinates to the nearest integers to get the main frequencies
    k1 = [int(round(k)) for k in k1]
    k2 = [int(round(k)) for k in k2]
    k3 = [int(round(k)) for k in k3]
    print(f"Main frequency peaks (rounded): {k1}, {k2}, {k3}")

    X_centered = rate_map - rate_map.mean(-1, keepdims=True)
    X_centered -= X_centered.mean(-1, keepdims=True)
    Ua, _S, _V = scipy.linalg.svd(X_centered)
    rm_embed = Ua.T @ rate_map

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
