import sys

sys.path.append("/home/liwei/paddlepaddle")
import paddle
from dgl.nn.pytorch import GATConv, GATv2Conv
from paddle_utils import *


class MyGatModel(paddle.nn.Layer):
    def __init__(
        self,
        layer_num: int,
        first_in_feats,
        first_out_feats,
        first_heads,
        second_out_dim,
        second_heads,
    ):
        super(MyGatModel, self).__init__()
        self.layer_num = layer_num
        if self.layer_num == 1:
            self.conv = GATv2Conv(
                in_feats=first_in_feats,
                out_feats=first_out_feats,
                num_heads=first_heads,
                allow_zero_in_degree=True,
            )
        elif self.layer_num == 2:
            self.conv1 = GATv2Conv(
                in_feats=first_in_feats,
                out_feats=first_out_feats,
                num_heads=first_heads,
                allow_zero_in_degree=True,
            )
            temp = first_out_feats * first_heads
            self.conv2 = GATv2Conv(
                in_feats=temp,
                out_feats=second_out_dim,
                num_heads=second_heads,
                allow_zero_in_degree=True,
            )
        self.activation = paddle.nn.LeakyReLU()

    def forward(self, g, h):
        if self.layer_num == 1:
            h = self.conv(g, h)
            h = self.activation(h)
        elif self.layer_num == 2:
            h = self.conv1(g, h)
            h = self.activation(h)
            h = h.reshape(tuple(h.shape)[0], -1)
            h = self.conv2(g, h)
            h = self.activation(h)
        return h
