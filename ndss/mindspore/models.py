import mindspore as ms
from mindspore import nn, ops
from mindspore.vision.models import resnet18, resnet152


def conv_ln_lrelu_cell(in_dim, out_dim):
    return nn.SequentialCell([
        nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=5, stride=2, pad_mode='pad', padding=2),
        # 使用 InstanceNorm2d 替代 LayerNorm
        nn.InstanceNorm2d(num_features=out_dim, affine=True),
        nn.LeakyReLU(alpha=0.2)
    ])

class GanGenerator(nn.Cell):  # in_dim=100
    def __init__(self, in_dim=100, dim=64):
        super(GanGenerator, self).__init__()
        def dconv_bn_relu(in_c, out_c):
            return nn.SequentialCell([
                nn.Conv2dTranspose(in_channels=in_c, out_channels=out_c,
                                   kernel_size=5, stride=2,
                                   pad_mode='pad', padding=2, output_padding=1, has_bias=False),
                nn.BatchNorm2d(num_features=out_c),
                nn.ReLU()
            ])

        self.l1 = nn.SequentialCell([
            nn.Dense(in_channels=in_dim, out_channels=dim * 8 * 4 * 4, has_bias=False),
            nn.BatchNorm1d(num_features=dim * 8 * 4 * 4),
            nn.ReLU()
        ])
        self.l2_5 = nn.SequentialCell([
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.Conv2dTranspose(in_channels=dim, out_channels=3,
                               kernel_size=5, stride=2,
                               pad_mode='pad', padding=2, output_padding=1, has_bias=False),
            nn.Sigmoid()
        ])
        self.reshape = ops.Reshape()

    def construct(self, x):
        y = self.l1(x)
        y = self.reshape(y, (y.shape[0], -1, 4, 4))
        y = self.l2_5(y)
        return y

class newGanGenerator(nn.Cell):  # in_dim=150
    def __init__(self, in_dim=150, dim=64):
        super(newGanGenerator, self).__init__()
        def dconv_bn_relu(in_c, out_c):
            return nn.SequentialCell([
                nn.Conv2dTranspose(in_channels=in_c, out_channels=out_c,
                                   kernel_size=5, stride=2,
                                   pad_mode='pad', padding=2, output_padding=1, has_bias=False),
                nn.BatchNorm2d(num_features=out_c),
                nn.ReLU()
            ])

        self.l1 = nn.SequentialCell([
            nn.Dense(in_channels=in_dim, out_channels=dim * 4 * 8 * 8, has_bias=False),
            nn.BatchNorm1d(num_features=dim * 4 * 8 * 8),
            nn.ReLU()
        ])
        self.l2_5 = nn.SequentialCell([
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.Conv2dTranspose(in_channels=dim, out_channels=3,
                               kernel_size=5, stride=2,
                               pad_mode='pad', padding=2, output_padding=1, has_bias=False),
            nn.Sigmoid()
        ])
        self.reshape = ops.Reshape()

    def construct(self, x):
        y = self.l1(x)
        y = self.reshape(y, (y.shape[0], -1, 8, 8))
        y = self.l2_5(y)
        return y

class GanDiscriminator(nn.Cell):
    def __init__(self, in_dim=3, dim=64):
        super(GanDiscriminator, self).__init__()
        self.ls = nn.SequentialCell([
            nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=5, stride=2, pad_mode='pad', padding=2),
            nn.LeakyReLU(alpha=0.2),
            conv_ln_lrelu_cell(dim, dim * 2),
            conv_ln_lrelu_cell(dim * 2, dim * 4),
            conv_ln_lrelu_cell(dim * 4, dim * 8),
            nn.Conv2d(in_channels=dim * 8, out_channels=1, kernel_size=4)
        ])
        self.flatten = ops.Flatten()

    def construct(self, x):
        y = self.ls(x)
        y = self.flatten(y)
        return y

class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        # 需确保在 MindSpore Vision 中存在对应的预训练模型
        self.model = resnet18(pretrained=True)

    def construct(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        return x

class Classifier(nn.Cell):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = resnet18(pretrained=True)
        self.layers = nn.SequentialCell([
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool
        ])
        self.flatten = ops.Flatten()

    def construct(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.model.fc(x)
        return x

class Eval(nn.Cell):
    def __init__(self, n_classes):
        super(Eval, self).__init__()
        self.n_classes = n_classes
        self.model = resnet152(pretrained=True)
        self.fc_layer_1 = nn.Dense(in_channels=2048 * 2 * 2, out_channels=300)
        self.fc_layer_2 = nn.Dense(in_channels=300, out_channels=200)
        self.fc_layer_3 = nn.Dense(in_channels=200, out_channels=self.n_classes)
        self.reshape = ops.Reshape()

    def construct(self, x, toEnd=True):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.reshape(x, (x.shape[0], -1))
        if toEnd:
            x = self.fc_layer_1(x)
            x = self.fc_layer_2(x)
            x = self.fc_layer_3(x)
        return x

class Decoder(nn.Cell):
    def __init__(self, in_dim=64 * 16 * 16, dim=64):
        super(Decoder, self).__init__()
        def dconv_bn_relu(in_c, out_c):
            return nn.SequentialCell([
                nn.Conv2dTranspose(in_channels=in_c, out_channels=out_c,
                                   kernel_size=5, stride=2,
                                   pad_mode='pad', padding=2, output_padding=1, has_bias=False),
                nn.BatchNorm2d(num_features=out_c),
                nn.ReLU()
            ])

        self.l1 = nn.SequentialCell([
            nn.Dense(in_channels=in_dim, out_channels=dim * 8 * 4 * 4, has_bias=False),
            nn.BatchNorm1d(num_features=dim * 8 * 4 * 4),
            nn.ReLU()
        ])
        self.l2_5 = nn.SequentialCell([
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.Conv2dTranspose(in_channels=dim, out_channels=3,
                               kernel_size=5, stride=2,
                               pad_mode='pad', padding=2, output_padding=1, has_bias=False),
            nn.Sigmoid()
        ])
        self.reshape = ops.Reshape()

    def construct(self, x):
        y = self.reshape(x, (x.shape[0], -1))
        y = self.l1(y)
        y = self.reshape(y, (y.shape[0], -1, 4, 4))
        y = self.l2_5(y)
        return y

class Amortizer(nn.Cell):
    def __init__(self, nz=500):
        super(Amortizer, self).__init__()
        self.main = nn.SequentialCell([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(alpha=0.2),

            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=4, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(alpha=0.2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(alpha=0.2)
        ])
        self.output = nn.SequentialCell([
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2, stride=1, pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(in_channels=512, out_channels=nz, kernel_size=1)
        ])
        self.flatten = ops.Flatten()

    def construct(self, x):
        out = self.main(x)
        out2 = self.output(out)
        out2 = self.flatten(out2)
        return out2
