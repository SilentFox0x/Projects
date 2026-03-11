import torch.nn as nn
from dgl.nn.pytorch import GATConv, GATv2Conv


# class GATLayer(nn.Module):
#     def __init__(self, config, is_last_layer=False):
#         super(GATLayer, self).__init__()
#         self.config = config
#         # self.conv = GATConv(in_feats=self.config.gnn_in_dim, out_feats=self.config.gnn_out_dim,
#         self.conv = GATv2Conv(in_feats=self.config.gnn_in_dim, out_feats=self.config.gnn_out_dim,
#                             num_heads=self.config.gat['num_heads'],
#                             feat_drop=self.config.gat['feat_drop'],
#                             attn_drop=self.config.gat['attn_drop'],
#                             allow_zero_in_degree=True)
#         self.activation = nn.LeakyReLU()
#         self.is_last_layer = is_last_layer
#         if self.is_last_layer:
#             self.dropout = nn.Dropout(0.1)
#
#     def forward(self, g, h):
#         h = self.conv(g, h)
#         h = self.activation(h)
#         # if self.is_last_layer:
#         #     h = self.dropout(h)
#         return h


class MyGatModel(nn.Module):
    def __init__(self, layer_num: int, first_in_feats, first_out_feats, first_heads,
                 second_out_dim, second_heads):
        super(MyGatModel, self).__init__()
        self.layer_num = layer_num
        if self.layer_num == 1:
            self.conv = GATv2Conv(in_feats=first_in_feats, out_feats=first_out_feats, num_heads=first_heads,
                                  allow_zero_in_degree=True)
        elif self.layer_num == 2:
            self.conv1 = GATv2Conv(in_feats=first_in_feats, out_feats=first_out_feats, num_heads=first_heads,
                                   allow_zero_in_degree=True)
            temp = first_out_feats * first_heads
            self.conv2 = GATv2Conv(in_feats=temp, out_feats=second_out_dim, num_heads=second_heads,
                                   allow_zero_in_degree=True)
        self.activation = nn.LeakyReLU()
        # self.dropout = nn.Dropout(0.1)

    def forward(self, g, h):
        if self.layer_num == 1:
            h = self.conv(g, h)
            h = self.activation(h)
        elif self.layer_num == 2:
            h = self.conv1(g, h)
            h = self.activation(h)
            h = h.reshape(h.shape[0], -1)
            h = self.conv2(g, h)
            h = self.activation(h)
        return h
