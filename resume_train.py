import argparse
import torch

from grid_pattern_formation.place_cells import PlaceCells
from grid_pattern_formation.trajectory_generator import TrajectoryGenerator
from grid_pattern_formation.models.rnn import RNN
from grid_pattern_formation.models.trainer import Trainer
from grid_pattern_formation.utils.seed import seed_everything
from grid_pattern_formation.utils.config import load_config
from topoloss import TopoLoss, LaplacianPyramid, PowerSpectrum
from topoloss.scheduler import TauScheduler

seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config yaml")
parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint (.pth)")
parser.add_argument("--start_epoch", type=int, required=True, help="epoch to resume from")
parser.add_argument("--wandb_run_id", type=str, default=None, help="wandb run ID to resume logging to")
args = parser.parse_args()

options = load_config(config_path=args.config)


place_cells = PlaceCells(options)

model = RNN.from_pretrained(
    checkpoint_path=args.checkpoint,
    device=options.device,
    options=options,
    place_cells=place_cells,
).to(options.device)

trajectory_generator = TrajectoryGenerator(
    options=options,
    place_cells=place_cells,
)

topo_loss = None
tau_scheduler = None

if options.topoloss_tau is not None:
    if options.topoloss_type == "laplacian_pyramid":
        loss_config = LaplacianPyramid.from_layer(
            model=model,
            layer=model.RNN,
            factor_h=9,
            factor_w=9,
            scale=options.topoloss_tau,
            custom_weight_attribute_name="weight_hh_l0"
        )
    elif options.topoloss_type == "power_spectrum":
        loss_config = PowerSpectrum.from_layer(
            model=model,
            layer=model.RNN,
            freq_cutoff=5.0,
            scale=options.topoloss_tau,
            custom_weight_attribute_name="weight_hh_l0"
        )
    else:
        raise ValueError(f"Unsupported topo loss type: {options.topoloss_type}")

    topo_loss = TopoLoss(
        losses=[loss_config],
        strict_layer_type=False
    )

    if options.topoloss_tau_scheduler == "linear_decay":
        tau_scheduler = TauScheduler(
            topo_loss=topo_loss,
            start_value=options.topoloss_tau,
            end_value=0.0,
            num_steps=options.n_epochs * options.n_steps,
        )
    elif options.topoloss_tau_scheduler == "linear_warmup":
        tau_scheduler = TauScheduler(
            topo_loss=topo_loss,
            start_value=0.0,
            end_value=options.topoloss_tau,
            num_steps=options.n_epochs * options.n_steps,
        )
    else:
        tau_scheduler = None

    # Move the tau scheduler to the correct position
    if tau_scheduler is not None:
        for _ in range(args.start_epoch - 1):
            tau_scheduler.step()

trainer = Trainer(
    options=options,
    model=model,
    trajectory_generator=trajectory_generator,
    restore=False,
    topo_loss=topo_loss,
    tau_scheduler=tau_scheduler,
    wandb_run_id=args.wandb_run_id,
)

print(f"Resuming training from epoch {args.start_epoch}/{options.n_epochs}")
print(f"Loaded checkpoint: {args.checkpoint}")
if args.wandb_run_id:
    print(f"Resuming wandb run: {args.wandb_run_id}")

trainer.train(
    n_epochs=options.n_epochs,
    n_steps=options.n_steps,
    start_epoch=args.start_epoch,
)
