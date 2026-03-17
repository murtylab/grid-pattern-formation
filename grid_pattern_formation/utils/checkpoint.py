import numpy as np
import torch
import os

checkpoint_url = "https://github.com/ganguli-lab/grid-pattern-formation/raw/refs/heads/master/models/example_trained_weights.npy"

def download_original_checkpoint(
    save_as: str = "original_checkpoint.pth",
    save_numpy_file_as: str = "original_checkpoint.npy",
):
    command = f"wget {checkpoint_url} -O {save_numpy_file_as}"
    os.system(command)

    convert_original_checkpoint_to_torch_state_dict(
        path=save_numpy_file_as,
        save_as=save_as,
    )

def convert_original_checkpoint_to_torch_state_dict(
    path: str,
    save_as: str = "original_checkpoint.pth",
) -> dict:
    
    data = np.load(path, allow_pickle=True)
    ## validate
    assert len(data) == 4, f"Expected checkpoint file to have 4 items but got: {len(data)}"
    expected_shapes = [(512, 4096), (2, 4096), (4096, 4096), (4096, 512)]

    '''
    ['encoder.weight: (4096, 512)', 'RNN.weight_ih_l0: (4096, 2)', 'RNN.weight_hh_l0: (4096, 4096)', 'decoder.weight: (512, 4096)']
    '''
    keys_in_order = [
    "encoder.weight",      # shape (512, 4096)  → but wait, comment says (4096, 512)
    "RNN.weight_ih_l0",    # shape (2, 4096)    → comment says (4096, 2)
    "RNN.weight_hh_l0",    # shape (4096, 4096)
    "decoder.weight",      # shape (512, 4096)  → comment says (4096, 512)
    ]

    transpose_keys = {"encoder.weight", "RNN.weight_ih_l0", "decoder.weight"}
    state_dict = {}
    for key, array in zip(keys_in_order, data):
        t = torch.from_numpy(array)
        if key in transpose_keys:
            t = t.T
        state_dict[key] = t

    torch.save(state_dict, save_as)
    print(f"Converted checkpoint saved to {save_as}")