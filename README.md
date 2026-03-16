# Grid Pattern Formation in RNNs

This repository is a refactored version of https://github.com/ganguli-lab/grid-pattern-formation. 

Run `python train.py --config config.yaml` to replicate the baseline results from the original paper. Basline config lies in `configs/replicate_baseline.yaml`

Below is a short description of all the args we can manipulate:
### General
| Arg | Default | Description |
|-----|---------|-------------|
| `save_dir` | - | Directory to save model checkpoints |
| `run_name` | - | Unique run ID (checkpoints save to `save_dir/run_name/`) |
| `n_epochs` | `100` | Number of training epochs |
| `save_every_epochs` | `10` | Save checkpoint every N epochs |
| `n_steps` | `1000` | Training steps per epoch |
| `dtype` | `float32` | Torch dtype |
| `workers` | `4` | Data loading workers |
| `device` | `cuda:0` | Torch device |

### Hyperparameters
| Arg | Default | Description |
|-----|---------|-------------|
| `batch_size` | `200` | Batch size |
| `sequence_length` | `20` | Length of trajectory sequences |
| `learning_rate` | `1e-4` | Learning rate |
| `optimizer` | `RMSprop` | `RMSprop` or `Adam` |
| `weight_decay` | `1e-4` | L2 regularization on recurrent weights |

### Topography
| Arg | Default | Description |
|-----|---------|-------------|
| `use_topographic_loss` | `false` | Enable topographic loss |
| `tau` | `1.0` | Peak topographic loss weight |
| `topoloss_scheduler_type`*** | `cosine_annealing` | `cosine_annealing` (high to low), `reverse_cosine_annealing` (low to high), or `constant`.  |

### Model
| Arg | Default | Description |
|-----|---------|-------------|
| `RNN_type` | `RNN` | Recurrent architecture |
| `activation` | `relu` | RNN activation function |
| `Ng` | `4096` | Number of grid cells (hidden units) |
| `Np` | `512` | Number of place cells (output units) |

### Place cells
| Arg | Default | Description |
|-----|---------|-------------|
| `place_cell_rf` | `0.12` | Tuning curve width in meters |
| `surround_scale` | `2` | DoG ratio (σ₂² / σ₁²) |
| `DoG` | `true` | Use difference-of-Gaussians tuning |

### Environment
| Arg | Default | Description |
|-----|---------|-------------|
| `periodic` | `false` | `true` = toroidal wrap, `false` = walled box |
| `box_width` | `2.2` | Arena width in meters |
| `box_height` | `2.2` | Arena height in meters |

### Logging
| Arg | Default | Description |
|-----|---------|-------------|
| `wandb_log` | `true` | Log metrics to W&B |
| `wandb_project` | `grid-pattern-formation` | W&B project name |

*** Note: To add a new topographic scheduler, go to `models/trainer.py` and add a new entry to the `Trainer.topoloss_scheduler` function. 