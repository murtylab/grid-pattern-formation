import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import yaml

from ..models.rnn import RNN
from ..place_cells import PlaceCells
from ..trajectory_generator import TrajectoryGenerator
from ..utils.visualize import compute_ratemaps

@dataclass
class EvalContext:
    options: argparse.Namespace
    model: torch.nn.Module
    place_cells: PlaceCells
    trajectory_generator: TrajectoryGenerator
    model_name: str
    results_dir: str
    save_dir: str
    cache: Dict[Tuple[Any, ...], Any] = field(default_factory=dict)

def load_options(config_path: str) -> argparse.Namespace:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    options = argparse.Namespace(**cfg)
    if hasattr(options, "dtype") and isinstance(options.dtype, str):
        options.dtype = getattr(torch, options.dtype)

    if not torch.cuda.is_available() and "cuda" in str(options.device):
        options.device = "cpu"

    return options

def _resolve_results_root(results_root: str) -> str:
    if os.path.isabs(results_root):
        return results_root

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, results_root)

def build_context(model_path: str, config_path: str, model_name: str, results_root: str) -> EvalContext:
    options = load_options(config_path=config_path)
    place_cells = PlaceCells(options)
    model = RNN.from_pretrained(
        checkpoint_path=model_path,
        device=options.device,
    )
    trajectory_generator = TrajectoryGenerator(options=options, place_cells=place_cells)

    results_dir = os.path.abspath(_resolve_results_root(results_root))
    save_dir = os.path.join(results_dir, model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    return EvalContext(
        options=options,
        model=model,
        place_cells=place_cells,
        trajectory_generator=trajectory_generator,
        model_name=model_name,
        results_dir=results_dir,
        save_dir=save_dir,
    )

def get_cached_ratemaps(
    ctx: EvalContext,
    *,
    res: int,
    n_avg: int,
    ng: int,
):
    key = ("ratemaps", res, n_avg, ng)
    if key not in ctx.cache:
        ctx.cache[key] = compute_ratemaps(
            model=ctx.model,
            trajectory_generator=ctx.trajectory_generator,
            options=ctx.options,
            res=res,
            n_avg=n_avg,
            Ng=ng,
        )
    return ctx.cache[key]
