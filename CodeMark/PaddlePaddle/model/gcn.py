import paddle
from dgl.nn.pytorch import GraphConv


class GCNLayer(paddle.nn.Layer):
    def __init__(self, config, is_last_layer=False):
        super(GCNLayer, self).__init__()
        self.config = config
        self.conv = GraphConv(
            self.config.gnn_in_dim, self.config.gnn_out_dim, allow_zero_in_degree=True
        )
        self.activation = paddle.nn.LeakyReLU()
        self.is_last_layer = is_last_layer
        if self.is_last_layer:
            self.dropout = paddle.nn.Dropout(p=0.1)

    def forward(self, g, h):
        h = self.conv(g, h)
        h = self.activation(h)
        if self.is_last_layer:
            h = self.dropout(h)
        return h


def build_gcn_layers(config, num_layers: int) -> paddle.nn.LayerList:
    if num_layers < 1:
        raise ValueError
    layers = paddle.nn.LayerList(
        sublayers=[GCNLayer(config, False) for _ in range(num_layers - 1)]
    )
    layers.append(GCNLayer(config, True))
    return layers
