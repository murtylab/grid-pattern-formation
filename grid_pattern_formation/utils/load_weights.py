import torch

def load_trained_weights(model, trainer, weight_dir):
    """Load weights stored as a .npy file (for github)"""

    # Train for a single step to initialize weights
    # trainer.train(n_epochs=1, n_steps=1, save=False)

    # Load weights from npy array
    device = model.options.device if hasattr(model, "options") else "cpu"
    state_dict = torch.load(weight_dir, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded trained weights.")
