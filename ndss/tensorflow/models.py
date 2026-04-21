# Converted from PyTorch to TensorFlow with same structure and naming
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_addons as tfa

class GanGenerator(tf.Module):
    def __init__(self, in_dim=100, dim=64):
        super(GanGenerator, self).__init__()

        def dconv_bn_relu(in_channels, out_channels):
            return tf.keras.Sequential([
                layers.Conv2DTranspose(out_channels, 5, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()])

        self.l1 = tf.keras.Sequential([
            layers.Dense(dim * 8 * 4 * 4, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()])

        self.l2_5 = tf.keras.Sequential([
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation='sigmoid')])

    def forward(self, x):
        y = self.l1(x)
        y = tf.reshape(y, [-1, 4, 4, y.shape[-1] // (4 * 4)])
        y = self.l2_5(y)
        return y


class newGanGenerator(tf.Module):
    def __init__(self, in_dim=150, dim=64):
        super(newGanGenerator, self).__init__()

        def dconv_bn_relu(in_channels, out_channels):
            return tf.keras.Sequential([
                layers.Conv2DTranspose(out_channels, 5, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()])

        self.l1 = tf.keras.Sequential([
            layers.Dense(dim * 4 * 8 * 8, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()])

        self.l2_5 = tf.keras.Sequential([
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation='sigmoid')])

    def forward(self, x):
        y = self.l1(x)
        y = tf.reshape(y, [-1, 8, 8, y.shape[-1] // (8 * 8)])
        y = self.l2_5(y)
        return y


class GanDiscriminator(tf.Module):
    def __init__(self, in_dim=3, dim=64):
        super(GanDiscriminator, self).__init__()

        def conv_ln_lrelu(in_channels, out_channels):
            return tf.keras.Sequential([
                layers.Conv2D(out_channels, 5, strides=2, padding='same'),
                tfa.layers.InstanceNormalization(),
                layers.LeakyReLU(0.2)])

        self.ls = tf.keras.Sequential([
            layers.Conv2D(dim, 5, strides=2, padding='same'), layers.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            layers.Conv2D(1, 4)])

    def forward(self, x):
        y = self.ls(x)
        return tf.reshape(y, [-1])


class Encoder(tf.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        self.model = tf.keras.Model(inputs=base.input, outputs=base.get_layer("conv2_block3_out").output)

    def forward(self, x):
        return self.model(x)


class Classifier(tf.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        x = layers.GlobalAveragePooling2D()(base.output)
        output = layers.Dense(40)(x)
        self.model = tf.keras.Model(inputs=base.input, outputs=output)

    def forward(self, x):
        return self.model(x)


class Eval(tf.Module):
    def __init__(self, n_classes):
        super(Eval, self).__init__()
        base = tf.keras.applications.ResNet152(include_top=False, weights='imagenet')
        self.model = base
        self.flatten = layers.Flatten()
        self.fc_layer_1 = layers.Dense(300)
        self.fc_layer_2 = layers.Dense(200)
        self.fc_layer_3 = layers.Dense(n_classes)

    def forward(self, x, toEnd=True):
        x = self.model(x)
        x = self.flatten(x)
        if toEnd:
            x = self.fc_layer_1(x)
            x = self.fc_layer_2(x)
            x = self.fc_layer_3(x)
        return x


class Decoder(tf.Module):
    def __init__(self, in_dim=64 * 16 * 16, dim=64):
        super(Decoder, self).__init__()

        def dconv_bn_relu(in_channels, out_channels):
            return tf.keras.Sequential([
                layers.Conv2DTranspose(out_channels, 5, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()])

        self.l1 = tf.keras.Sequential([
            layers.Dense(dim * 8 * 4 * 4, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()])

        self.l2_5 = tf.keras.Sequential([
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation='sigmoid')])

    def forward(self, x):
        y = tf.reshape(x, [tf.shape(x)[0], -1])
        y = self.l1(y)
        y = tf.reshape(y, [-1, 4, 4, y.shape[-1] // (4 * 4)])
        y = self.l2_5(y)
        return y


def conv_ln_lrelu(in_dim, out_dim):
    return tf.keras.Sequential([
        layers.Conv2D(out_dim, 5, strides=2, padding='same'),
        tfa.layers.InstanceNormalization(),
        layers.LeakyReLU(0.2)])


class Amortizer(tf.Module):
    def __init__(self, nz=500):
        super().__init__()
        self.main = tf.keras.Sequential([
            layers.Conv2D(128, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1024, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])
        self.output = tf.keras.Sequential([
            layers.Conv2D(512, 2, strides=1, padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(nz, 1, strides=1, padding='valid')
        ])

    def forward(self, x):
        out = self.main(x)
        out2 = self.output(out)
        out2 = tf.reshape(out2, [tf.shape(x)[0], -1])
        return out2