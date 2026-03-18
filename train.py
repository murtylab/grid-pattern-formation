import argparse
import torch

from grid_pattern_formation.place_cells import PlaceCells
from grid_pattern_formation.trajectory_generator import TrajectoryGenerator
from grid_pattern_formation.models.rnn import RNN
from grid_pattern_formation.models.trainer import Trainer
from grid_pattern_formation.utils.seed import seed_everything
from grid_pattern_formation.utils.config import load_config

seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config yaml")
args = parser.parse_args()

options = load_config(config_path=args.config)

if not torch.cuda.is_available() and "cuda" in str(options.device):
    options.device = "cpu"

place_cells = PlaceCells(options)

model = RNN(
    options=options,
    place_cells=place_cells,
).to(options.device)

trajectory_generator = TrajectoryGenerator(
    options=options,
    place_cells=place_cells,
)

trainer = Trainer(
    options=options,
    model=model,
    trajectory_generator=trajectory_generator,
    restore=False
)
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
