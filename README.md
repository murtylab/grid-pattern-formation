# Grid Pattern Formation in RNNs

This repository is a refactored version of https://github.com/ganguli-lab/grid-pattern-formation. 

Run `python train.py --config config.yaml` to replicate the baseline results from the original paper. Basline config lies in `configs/replicate_baseline.yaml`


Note: To add a new topographic scheduler, go to `models/trainer.py` and add a new entry to the `Trainer.topoloss_scheduler` function. 
