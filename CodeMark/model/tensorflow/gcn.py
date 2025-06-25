import tensorflow as tf
from tensorflow.keras import layers


class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, activation=None, dropout_rate=0.0):
        super(GraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation or tf.keras.activations.leaky_relu
        self.dropout_rate = dropout_rate

        self.linear = tf.keras.layers.Dense(out_dim)

    def call(self, inputs, training=False):
        X, A = inputs  # X: [N, F], A: [N, N]
        AX = tf.linalg.matmul(A, X)  # [N, F]
        out = self.linear(AX)
        if self.activation:
            out = self.activation(out)
        if training and self.dropout_rate > 0:
            out = tf.keras.layers.Dropout(self.dropout_rate)(out, training=training)
        return out


def build_gcn_layers(config, num_layers: int):
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    layers_list = []
    for i in range(num_layers):
        is_last = (i == num_layers - 1)
        dropout = 0.1 if is_last else 0.0
        gcn_layer = GraphConvLayer(
            in_dim=config.gnn_in_dim,
            out_dim=config.gnn_out_dim,
            activation=tf.keras.activations.leaky_relu,
            dropout_rate=dropout
        )
        layers_list.append(gcn_layer)
    return layers_list
