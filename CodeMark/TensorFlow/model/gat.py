import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# 可选：自定义 GATv2 层（简化版，支持两层 GATv2Conv）
class GATv2Layer(tf.keras.layers.Layer):
    def __init__(self, out_dim, num_heads, activation=None, **kwargs):
        super(GATv2Layer, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.activation = activation or tf.keras.activations.leaky_relu

    def build(self, input_shape):
        feature_dim = int(input_shape[0][-1])
        self.W = self.add_weight(shape=(feature_dim, self.out_dim * self.num_heads),
                                 initializer='glorot_uniform', trainable=True)
        self.attn_kernel = self.add_weight(shape=(2 * self.out_dim, 1),
                                           initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        X, A = inputs  # A is adjacency matrix (dense or sparse)

        # Linear transformation
        h = tf.matmul(X, self.W)  # shape: [N, out_dim * num_heads]
        h = tf.reshape(h, (-1, self.num_heads, self.out_dim))  # [N, heads, out_dim]

        # Compute attention scores
        N = tf.shape(h)[0]
        h1 = tf.tile(tf.expand_dims(h, axis=1), [1, N, 1, 1])
        h2 = tf.tile(tf.expand_dims(h, axis=0), [N, 1, 1, 1])
        a_input = tf.concat([h1, h2], axis=-1)  # [N, N, heads, 2*out_dim]

        e = tf.nn.leaky_relu(tf.tensordot(a_input, self.attn_kernel, axes=[[3], [0]]))  # [N, N, heads, 1]
        e = tf.squeeze(e, axis=-1)  # [N, N, heads]

        # Mask out non-neighbors
        mask = tf.cast(A > 0, tf.float32)
        mask = tf.expand_dims(mask, -1)  # [N, N, 1]
        e = e * mask - 1e9 * (1.0 - mask)

        # Softmax
        attention = tf.nn.softmax(e, axis=1)  # [N, N, heads]

        # Apply attention
        h_prime = tf.einsum('nij,njh->nih', attention, h)  # [N, heads, out_dim]
        h_prime = tf.reshape(h_prime, (-1, self.out_dim * self.num_heads))  # [N, heads*out_dim]

        if self.activation:
            h_prime = self.activation(h_prime)

        return h_prime


class MyGatModel(tf.keras.Model):
    def __init__(self, layer_num, first_in_feats, first_out_feats, first_heads,
                 second_out_dim, second_heads):
        super(MyGatModel, self).__init__()
        self.layer_num = layer_num
        if self.layer_num == 1:
            self.gat = GATv2Layer(out_dim=first_out_feats, num_heads=first_heads)
        elif self.layer_num == 2:
            self.gat1 = GATv2Layer(out_dim=first_out_feats, num_heads=first_heads)
            self.gat2 = GATv2Layer(out_dim=second_out_dim, num_heads=second_heads)

    def call(self, inputs):
        X, A = inputs  # Node features and adjacency matrix
        if self.layer_num == 1:
            X = self.gat([X, A])
        elif self.layer_num == 2:
            X = self.gat1([X, A])
            X = self.gat2([X, A])  # 或使用更新后的 X, A 结构
        return X
