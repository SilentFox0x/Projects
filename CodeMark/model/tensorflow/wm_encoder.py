import tensorflow as tf
from tensorflow.keras import layers

class WMLinearEncoder(tf.keras.Model):
    def __init__(self, n_bits: int, embedding_dim: int = 64):
        super(WMLinearEncoder, self).__init__()
        self.linear = layers.Dense(embedding_dim, input_shape=(n_bits,))

    def call(self, x):
        return self.linear(x)
