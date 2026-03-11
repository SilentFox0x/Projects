import paddle


class WMLinearEncoder(paddle.nn.Layer):
    def __init__(self, n_bits: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=n_bits, out_features=embedding_dim)

    def forward(self, x: paddle.Tensor):
        return self.linear(x)
