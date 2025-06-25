import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class GCNLayer(nn.Cell):
    def __init__(self, config, is_last_layer=False):
        super(GCNLayer, self).__init__()
        self.config = config
        self.is_last_layer = is_last_layer

        self.weight = nn.Dense(self.config.gnn_in_dim, self.config.gnn_out_dim, has_bias=False)
        self.activation = nn.LeakyReLU()
        if self.is_last_layer:
            self.dropout = nn.Dropout(keep_prob=0.9)

    def construct(self, adj, h):
        """
        adj: 邻接矩阵，形状为 [N, N]
        h:   节点特征矩阵，形状为 [N, in_dim]
        """
        h = ops.MatMul()(adj, h)
        h = self.weight(h)
        h = self.activation(h)
        if self.is_last_layer:
            h = self.dropout(h)
        return h
