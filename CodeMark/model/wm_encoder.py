import torch
import torch.nn as nn


class WMLinearEncoder(nn.Module):
    def __init__(self, n_bits: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(n_bits, embedding_dim)

    def forward(self, x: torch.Tensor):
        return self.linear(x)