import mindspore
import mindspore.nn as nn
from mindspore import Tensor


class WMLinearEncoder(nn.Cell):
    def __init__(self, n_bits: int, embedding_dim: int = 64) -> None:
        super(WMLinearEncoder, self).__init__()
        self.linear = nn.Dense(n_bits, embedding_dim)

    def construct(self, x: Tensor):
        return self.linear(x)
