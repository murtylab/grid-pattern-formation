import torch
import torch.nn.functional as F
from topoloss import TopoLoss
from einops import rearrange

class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells
        self.device = torch.device(options.device)
        self.dtype = options.dtype
        self.sorscher_compatible = getattr(options, "sorscher_compatible", False)
        
        self.alive_lambda = getattr(options, "alive_lambda", 0.1)
        self.alive_threshold = getattr(options, "alive_threshold", 0.01)

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
        
        if getattr(options, "activation", "relu").lower() == "relu":
            self._initialize_weights()
            
    def _initialize_weights(self):
        torch.nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        
        for name, param in self.RNN.named_parameters():
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

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
        g = self.g(inputs)
        place_preds = self.decoder(g)

        return place_preds

    def compute_loss(
        self,
        inputs,
        pc_outputs,
        pos,
        topo_loss: TopoLoss = None,
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

        preds_flat = rearrange(preds, "b s p -> (b s) p")
        pc_outputs_flat = rearrange(pc_outputs, "b s p -> (b s) p")
        loss = F.cross_entropy(input=preds_flat, target=pc_outputs_flat)

        # L2 Weight regularization
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        if topo_loss is not None:
            topo_loss_value = topo_loss.compute(model=self)
            print(f"Topo loss value: {topo_loss_value.item():.4f}")
            loss = loss + topo_loss_value

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()

        return loss, err

    @classmethod
    def from_pretrained(cls, checkpoint_path, device, options=None, place_cells=None):
        # Load to CPU first to avoid MPS float64 errors, then move to device
        model_or_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(model_or_state_dict, torch.nn.Module):
            model = model_or_state_dict
            if options is not None:
                model = model.to(dtype=options.dtype)
            if place_cells is not None:
                model.place_cells = place_cells
        elif isinstance(model_or_state_dict, dict):
            assert options is not None and place_cells is not None, "Options and place cells must be provided when loading from state dict"
            model = cls(options=options, place_cells=place_cells)
            model.load_state_dict(model_or_state_dict, strict=True)

        return model
