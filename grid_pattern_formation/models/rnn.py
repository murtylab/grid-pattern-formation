from einops import rearrange
import torch
import torch.nn.functional as F


class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells
        self.device = torch.device(options.device)
        self.dtype = torch.float32 if options.dtype is None else options.dtype

        # Input weights
        self.encoder = torch.nn.Linear(
            self.Np, self.Ng, bias=False, dtype=self.dtype, device=self.device
        )
        self.RNN = torch.nn.RNN(
            input_size=2,
            hidden_size=self.Ng,
            nonlinearity=options.activation,
            bias=False,
            dtype=self.dtype,
            device=self.device,
        )
        self.decoder = torch.nn.Linear(
            self.Ng, self.Np, bias=False, dtype=self.dtype, device=self.device
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        """
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns:
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        """
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g, _ = self.RNN(v, init_state)
        return g

    def predict(self, inputs):
        """
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns:
            place_preds: Predicted place cell activations with shape
                [batch_size, sequence_length, Np].
        """
        place_preds = self.decoder(self.g(inputs))

        return place_preds

    def compute_loss(
        self,
        inputs,
        pc_outputs,
        pos,
        apply_topoloss: bool = False,
        tau: float = None,
    ):
        """
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        """

        preds = self.predict(inputs)
        yhat = torch.clamp(
            self.softmax(preds), min=1e-8, max=1.0
        )  # NOTE: why is the clamp necessary? original authors dont use it
        loss = -(pc_outputs * torch.log(yhat)).sum(-1).mean()

        # L2 Weight regularization
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        if apply_topoloss:
            loss += combined_biological_topo_loss(
                weight_matrix=self.RNN.weight_hh_l0,
                grid_height=64,
                grid_width=64,
                lambda_smoothness=tau,
            )

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()

        return loss, err

    @classmethod
    def from_pretrained(cls, checkpoint_path, device, options=None, place_cells=None):
        model_or_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(model_or_state_dict, torch.nn.Module):
            model = model_or_state_dict
        elif isinstance(model_or_state_dict, dict):
            assert options is not None and place_cells is not None, "Options and place cells must be provided when loading from state dict"
            model = cls(options=options, place_cells=place_cells)  # dummy init, will be overwritten by state dict
            model.load_state_dict(model_or_state_dict, strict=True)
            
        return model


def combined_biological_topo_loss(
    weight_matrix: torch.Tensor,
    grid_height: int,
    grid_width: int,
    factor_h: float = 4.0,
    factor_w: float = 4.0,
    tau: float = 1.0,
):
    """
    Combine TopoLoss smoothness with distance penalty.

    Smoothness: Neighboring neurons have similar connectivity patterns
    """
    n_neurons = weight_matrix.shape[0]
    assert n_neurons == grid_height * grid_width

    # Topoloss smoothness
    cortical_sheet = weight_matrix.reshape(grid_height, grid_width, n_neurons)
    grid = rearrange(cortical_sheet, "h w e -> e h w").unsqueeze(0)

    # Blur operation
    downscaled = F.interpolate(
        grid,
        scale_factor=(1 / factor_h, 1 / factor_w),
        mode="bilinear",
        align_corners=False,
    )
    upscaled = F.interpolate(
        downscaled, size=grid.shape[2:], mode="bilinear", align_corners=False
    )

    # Smoothness loss: neighboring neurons should have similar connectivity
    grid_flat = rearrange(grid.squeeze(0), "e h w -> (h w) e")
    upscaled_flat = rearrange(upscaled.squeeze(0), "e h w -> (h w) e")
    smoothness_loss = 1 - F.cosine_similarity(grid_flat, upscaled_flat, dim=-1).mean()

    # apply the scheduler lambda to the topo loss
    total_topo_loss = tau * smoothness_loss
    return total_topo_loss
