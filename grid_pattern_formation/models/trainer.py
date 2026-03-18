import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from ..trajectory_generator import TrajectoryGenerator
from ..utils.visualize import save_ratemaps

def _to_wandb_config(options):
    config = {}
    for key, value in vars(options).items():
        if isinstance(value, (str, int, float, bool, type(None))):
            config[key] = value
        else:
            config[key] = str(value)
    return config

class Trainer(object):
    def __init__(
        self, 
        options: argparse.Namespace, 
        model: nn.Module, 
        trajectory_generator: TrajectoryGenerator, 
        restore=False
    ):
        
        assert isinstance(trajectory_generator, TrajectoryGenerator), f"trajectory_generator must be an instance of TrajectoryGenerator but got: {type(trajectory_generator)}"

        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        self.topographic_loss = options.use_topographic_loss
        
        
        if self.topographic_loss:
            assert options.tau is not None, "tau value must be specified in options when using topographic loss"
            assert options.topoloss_scheduler_type is not None, "topoloss_scheduler_type must be specified in options when using topographic loss"
            self.tau = options.tau
        else:
            self.tau = None
        
        self.log_to_wandb = self.options.wandb_log
        self.wandb_run = None
        if self.log_to_wandb:

            self.wandb_run = wandb.init(
                project=self.options.wandb_project,
                name=options.run_name,
                config=_to_wandb_config(options),
            )

        assert options.optimizer in [
            "Adam",
            "RMSprop",
        ], "Optimizer must be either Adam or RMSprop"
        if options.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=options.learning_rate)
        elif options.optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=options.learning_rate)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_name)
        ckpt_path = os.path.join(self.ckpt_dir, "most_recent_model.pth")
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            print(f"\033[92mRestoring model from {ckpt_path}\033[0m")
            self.model.load_state_dict(torch.load(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            else:
                print(f"\033[91mCheckpoint directory {self.ckpt_dir} already exists!\033[0m")

    def train_step(self, inputs, pc_outputs, pos, tau=None):
        self.model.zero_grad()

        loss, err = self.model.compute_loss(
            inputs=inputs,
            pc_outputs=pc_outputs,
            pos=pos,
            apply_topoloss=self.topographic_loss,
            tau=tau,
        )

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def topoloss_scheduler(self, epoch_idx: int):
        if self.options.topoloss_scheduler_type == "constant":
            return self.tau
        elif self.options.topoloss_scheduler_type == "cosine_annealing":
            min_lambda = self.tau * 0.1
            ep = min(epoch_idx, self.options.n_epochs)
            progress = ep / self.options.n_epochs
            lambda_val = min_lambda + (
                self.tau - min_lambda
            ) * 0.5 * (1 + np.cos(np.pi * progress))
            return lambda_val
        elif self.options.topoloss_scheduler_type == "reverse_cosine_annealing":
            min_lambda = self.tau * 0.1
            ep = min(epoch_idx, self.options.n_epochs)
            progress = ep / self.options.n_epochs
            lambda_val = self.tau - (
                (self.tau - min_lambda) * 0.5 * (1 + np.cos(np.pi * progress))
            )
            return lambda_val
        else:
            raise ValueError(f"Unknown scheduler type: {self.options.topoloss_scheduler_type}.")

    def train(self, n_epochs: int, n_steps: int, save: bool = True):
        gen = self.trajectory_generator.get_generator()

        # Precompute topo lambdas per epoch if using scheduler
        topo_lambdas_scheduler = None
        if self.topographic_loss:
            topo_lambdas_scheduler = {
                e: self.topoloss_scheduler(e) for e in range(1, n_epochs + 1)
            }

        for epoch_idx in range(1, n_epochs + 1):
            topo_lambda = topo_lambdas_scheduler[epoch_idx] if topo_lambdas_scheduler else None

            step_bar = tqdm(
                range(n_steps),
                desc=f"Epoch {epoch_idx}/{n_epochs}",
                unit="step",
                leave=False,
            )
            for step_idx in step_bar:
                inputs, pc_outputs, pos = next(gen)
                loss, err = self.train_step(
                    inputs, pc_outputs, pos, tau=topo_lambda
                )
                self.loss.append(loss)
                self.err.append(err)
                step_bar.set_postfix(loss=f"{loss:.3f}", err=f"{100*err:.2f}cm")
                
                if self.log_to_wandb:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/err": err,
                        },
                        step=epoch_idx * n_steps + step_idx,
                    )

            if save and (epoch_idx % self.options.save_every_epochs == 0 or epoch_idx == 1):
                ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch_idx:06d}.pth")
                torch.save(obj=self.model, f=ckpt_path)
                save_ratemaps(
                    model=self.model, 
                    trajectory_generator=self.trajectory_generator, 
                    options=self.options, 
                    step=epoch_idx,
                    res=20,
                    n_avg=None
                )                
           

        torch.save(self.model, os.path.join(self.ckpt_dir, "final_model.pth"))
        if self.log_to_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
