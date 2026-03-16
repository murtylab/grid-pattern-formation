import argparse
import yaml
import torch

from grid_pattern_formation.place_cells import PlaceCells
from grid_pattern_formation.trajectory_generator import TrajectoryGenerator
from grid_pattern_formation.models.rnn import RNN
from grid_pattern_formation.models.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

options = argparse.Namespace(**cfg)
options.dtype = getattr(torch, options.dtype)
if not torch.cuda.is_available() and "cuda" in str(options.device):
    options.device = "cpu"

place_cells = PlaceCells(options)
model = RNN(options, place_cells).to(options.device)
trajectory_generator = TrajectoryGenerator(options, place_cells)
trainer = Trainer(options, model, trajectory_generator)
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
