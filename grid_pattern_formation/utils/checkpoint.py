import numpy as np
import torch
import os

checkpoint_url = "https://github.com/ganguli-lab/grid-pattern-formation/raw/refs/heads/master/models/example_trained_weights.npy"

def download_original_checkpoint(
    save_as: str
):
    command = f"wget {checkpoint_url} -O {save_as}"
    print(f"Downloading original checkpoint from {checkpoint_url} to {save_as}...")
    os.system(command)


def convert_original_checkpoint_to_torch_state_dict(
    path: str,
) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
    data = np.load(path, allow_pickle=True)
    ## validate
    assert len(data) == 4, f"Expected checkpoint file to have 4 items but got: {len(data)}"
    expected_shapes = [(512, 4096), (2, 4096), (4096, 4096), (4096, 512)]

    '''
    ['encoder.weight: (4096, 512)', 'RNN.weight_ih_l0: (4096, 2)', 'RNN.weight_hh_l0: (4096, 4096)', 'decoder.weight: (512, 4096)']
    '''

    keys_in_order = [
        "encoder.weight",
        "RNN.weight_ih_l0",
        "RNN.weight_hh_l0",
        "decoder.weight",
    ]

    for index in len(data):
        assert data[index].shape == expected_shapes[index]
    
    state_dict = {}
    for key, array in zip(keys_in_order, data):
        state_dict[key] = torch.from_numpy(array.T) # need to transpose to match PyTorch's weight shape convention

    return state_dict