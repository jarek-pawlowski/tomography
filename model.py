import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, layers: int = 2, hidden_size: int = 128):
        super(Regressor, self).__init__()
        self.mlp = self._get_mlp(layers, input_dim, hidden_size, output_dim)

    def _get_mlp(self, layers: int, input_dim: int, hidden_size: int, output_dim: int):
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_size))
        mlp.append(nn.ReLU())
        for i in range(layers - 1):
            mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_size, output_dim))
        return nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
