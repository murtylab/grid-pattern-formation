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

    assert len(data) == 4, f"Expected 4 items in checkpoint, got {len(data)}"

    expected_shapes = [(512, 4096), (2, 4096), (4096, 4096), (4096, 512)]
    keys_in_order = ["encoder.weight", "RNN.weight_ih_l0", "RNN.weight_hh_l0", "decoder.weight"]

    state_dict = {}
    for key, array, shape in zip(keys_in_order, data, expected_shapes):
        assert array.shape == shape, f"{key}: expected shape {shape}, got {array.shape}"
        state_dict[key] = torch.from_numpy(array.T)

    torch.save(state_dict, save_as)
    print(f"Converted checkpoint saved to {save_as}")
    return state_dict