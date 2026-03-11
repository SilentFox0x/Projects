import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class GraphAttentionLayer(nn.Cell):
    def __init__(self, in_feats, out_feats, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.linear = nn.Dense(in_feats, out_feats * num_heads, has_bias=False)
        self.attn_l = nn.Dense(out_feats, 1, has_bias=False)
        self.attn_r = nn.Dense(out_feats, 1, has_bias=False)
        self.leaky_relu = nn.LeakyReLU()

    def construct(self, g, features):
        # 这里 g 应该是邻接信息，features 是节点特征 (N, F)
        h = self.linear(features)
        h = h.reshape((h.shape[0], self.num_heads, self.out_feats))  # [N, H, F]

        el = self.attn_l(h)  # [N, H, 1]
        er = self.attn_r(h)  # [N, H, 1]

        # 模拟边 attention（没有 DGL 的边聚合函数，需要自定义或外部提供邻接矩阵）
        # 这里只能提供一个近似框架，具体取决于图数据的表示方式
        # 实际应用建议使用 MindSpore Geometric 或自定义聚合逻辑

        # 返回的是线性转换后的特征
        return h.reshape((h.shape[0], -1))


class MyGatModel(nn.Cell):
    def __init__(self, layer_num: int, first_in_feats, first_out_feats, first_heads,
                 second_out_dim, second_heads):
        super(MyGatModel, self).__init__()
        self.layer_num = layer_num
        self.activation = nn.LeakyReLU()

        if self.layer_num == 1:
            self.conv = GraphAttentionLayer(first_in_feats, first_out_feats, first_heads)
        elif self.layer_num == 2:
            self.conv1 = GraphAttentionLayer(first_in_feats, first_out_feats, first_heads)
            temp = first_out_feats * first_heads
            self.conv2 = GraphAttentionLayer(temp, second_out_dim, second_heads)

    def construct(self, g, h):
        if self.layer_num == 1:
            h = self.conv(g, h)
            h = self.activation(h)
        elif self.layer_num == 2:
            h = self.conv1(g, h)
            h = self.activation(h)
            h = h.reshape((h.shape[0], -1))
            h = self.conv2(g, h)
            h = self.activation(h)
        return h
