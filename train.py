import argparse
import torch

from grid_pattern_formation.place_cells import PlaceCells
from grid_pattern_formation.trajectory_generator import TrajectoryGenerator
from grid_pattern_formation.models.rnn import RNN
from grid_pattern_formation.models.trainer import Trainer
from grid_pattern_formation.utils.seed import seed_everything
from grid_pattern_formation.utils.config import load_config
from topoloss import TopoLoss, LaplacianPyramid
from topoloss.scheduler import TauScheduler

seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config yaml")
args = parser.parse_args()

options = load_config(config_path=args.config)

place_cells = PlaceCells(options)

model = RNN(
    options=options,
    place_cells=place_cells,
).to(options.device)

trajectory_generator = TrajectoryGenerator(
    options=options,
    place_cells=place_cells,
)


if options.topoloss_tau is not None:
    if options.topoloss_type == "laplacian_pyramid":
        loss_config = LaplacianPyramid.from_layer(
            model=model,
            layer=model.RNN,
            factor_h=options.topoloss_factor_h,
            factor_w=options.topoloss_factor_w,
            scale=options.topoloss_tau,
        )
        loss_config.custom_weight_attribute_name = "weight_hh_l0"
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
            num_steps=options.n_epochs,
        )
    elif options.topoloss_tau_scheduler == "linear_growth":
        tau_scheduler = TauScheduler(
            topo_loss=topo_loss,
            start_value=0.0,
            end_value=options.topoloss_tau,
            num_steps=options.n_epochs,
        )
    elif options.topoloss_tau_scheduler == "cosine_growth":
        tau_scheduler = TauScheduler(
            topo_loss=topo_loss,
            start_value=0.0,
            end_value=options.topoloss_tau,
            num_steps=options.n_epochs,
            mode="cosine_growth"
        )
    elif options.topoloss_tau_scheduler == "cosine_decay":
        tau_scheduler = TauScheduler(
            topo_loss=topo_loss,
            start_value=options.topoloss_tau,
            end_value=0.0,
            num_steps=options.n_epochs,
            mode="cosine_decay"
        )
    
    else:
        tau_scheduler = None
else:
    pass

trainer = Trainer(
    options=options,
    model=model,
    trajectory_generator=trajectory_generator,
    restore=False,
    topo_loss=topo_loss,
    tau_scheduler=tau_scheduler,
)

trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
