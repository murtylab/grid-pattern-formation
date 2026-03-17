import torch

def load_trained_weights(model, weight_dir):
    """Load weights stored as a .npy file (for github)"""

    device = model.options.device if hasattr(model, "options") else "cpu"
    loaded_model = torch.load(weight_dir, map_location=device, weights_only=False)
    model.load_state_dict(loaded_model.state_dict(), strict=True)
    
    print("Loaded trained weights")
    
    return model
