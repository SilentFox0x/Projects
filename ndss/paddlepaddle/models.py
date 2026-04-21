import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import paddle
from paddle_utils import *


class GanGenerator(paddle.nn.Layer):
    def __init__(self, in_dim=100, dim=64):
        super(GanGenerator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return paddle.nn.Sequential(
                paddle.nn.Conv2DTranspose(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                    bias_attr=False,
                ),
                paddle.nn.BatchNorm2D(num_features=out_dim),
                paddle.nn.ReLU(),
            )

        self.l1 = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=in_dim, out_features=dim * 8 * 4 * 4, bias_attr=False
            ),
            paddle.nn.BatchNorm1D(num_features=dim * 8 * 4 * 4),
            paddle.nn.ReLU(),
        )
        self.l2_5 = paddle.nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            paddle.nn.Conv2DTranspose(
                in_channels=dim,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            paddle.nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.shape[0], -1, 4, 4)
        y = self.l2_5(y)
        return y


class newGanGenerator(paddle.nn.Layer):
    def __init__(self, in_dim=150, dim=64):
        super(newGanGenerator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return paddle.nn.Sequential(
                paddle.nn.Conv2DTranspose(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                    bias_attr=False,
                ),
                paddle.nn.BatchNorm2D(num_features=out_dim),
                paddle.nn.ReLU(),
            )

        self.l1 = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=in_dim, out_features=dim * 4 * 8 * 8, bias_attr=False
            ),
            paddle.nn.BatchNorm1D(num_features=dim * 4 * 8 * 8),
            paddle.nn.ReLU(),
        )
        self.l2_5 = paddle.nn.Sequential(
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            paddle.nn.Conv2DTranspose(
                in_channels=dim,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            paddle.nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.shape[0], -1, 8, 8)
        y = self.l2_5(y)
        return y


class GanDiscriminator(paddle.nn.Layer):
    def __init__(self, in_dim=3, dim=64):
        super(GanDiscriminator, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                paddle.nn.InstanceNorm2D(
                    num_features=out_dim, weight_attr=True, bias_attr=True
                ),
                paddle.nn.LeakyReLU(negative_slope=0.2),
            )

        self.ls = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_dim, out_channels=dim, kernel_size=5, stride=2, padding=2
            ),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            paddle.nn.Conv2D(in_channels=dim * 8, out_channels=1, kernel_size=4),
        )

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


class Encoder(paddle.nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = paddle.vision.models.resnet18(pretrained=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        return x


class Classifier(paddle.nn.Layer):
    def __init__(self):
        super(F, self).__init__()
        self.model = paddle.vision.models.resnet18(pretrained=True)
        self.layers = paddle.nn.Sequential(
            self.model.layer2, self.model.layer3, self.model.layer4, self.model.avgpool
        )

    def forward(self, x):
        x = self.layers(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.model.fc(x)
        return x


class Eval(paddle.nn.Layer):
    def __init__(self, n_classes):
        super(Eval, self).__init__()
        self.n_classes = n_classes
        self.model = paddle.vision.models.resnet152(pretrained=True)
        self.fc_layer_1 = paddle.nn.Linear(in_features=2048 * 2 * 2, out_features=300)
        self.fc_layer_2 = paddle.nn.Linear(in_features=300, out_features=200)
        self.fc_layer_3 = paddle.nn.Linear(in_features=200, out_features=self.n_classes)

    def forward(self, x, toEnd=True):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.shape[0], -1)
        if toEnd:
            x = self.fc_layer_1(x)
            x = self.fc_layer_2(x)
            x = self.fc_layer_3(x)
        return x


class Decoder(paddle.nn.Layer):
    def __init__(self, in_dim=64 * 16 * 16, dim=64):
        super(Decoder, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return paddle.nn.Sequential(
                paddle.nn.Conv2DTranspose(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                    bias_attr=False,
                ),
                paddle.nn.BatchNorm2D(num_features=out_dim),
                paddle.nn.ReLU(),
            )

        self.l1 = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=in_dim, out_features=dim * 8 * 4 * 4, bias_attr=False
            ),
            paddle.nn.BatchNorm1D(num_features=dim * 8 * 4 * 4),
            paddle.nn.ReLU(),
        )
        self.l2_5 = paddle.nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            paddle.nn.Conv2DTranspose(
                in_channels=dim,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            paddle.nn.Sigmoid(),
        )

    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = self.l1(y)
        y = y.view(y.shape[0], -1, 4, 4)
        y = self.l2_5(y)
        return y


def conv_ln_lrelu(in_dim, out_dim):
    return paddle.nn.Sequential(
        paddle.nn.Conv2D(
            in_channels=in_dim, out_channels=out_dim, kernel_size=5, stride=2, padding=2
        ),
        paddle.nn.InstanceNorm2D(
            num_features=out_dim, weight_attr=True, bias_attr=True
        ),
        paddle.nn.LeakyReLU(negative_slope=0.2),
    )


class Amortizer(paddle.nn.Layer):
    def __init__(self, nz=500):
        super().__init__()
        self.main = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=128),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Conv2D(
                in_channels=128, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=512),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Conv2D(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            ),
            paddle.nn.BatchNorm2D(num_features=1024),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.output = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=1024,
                out_channels=512,
                kernel_size=2,
                stride=1,
                padding=0,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=512),
            paddle.nn.LeakyReLU(negative_slope=0.2),
            paddle.nn.Conv2D(in_channels=512, out_channels=nz, kernel_size=1),
        )

    def forward(self, x):
        out = self.main(x)
        out2 = self.output(out)
        out2 = out2.view(len(x), -1)
        return out2
