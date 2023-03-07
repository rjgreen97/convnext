import torch
import torch.nn as nn


class InvertedBottleneck(nn.Module):
    def __init__(self, channels):
        super(InvertedBottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=channels,
            ),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=channels * 4,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return x


class ConvNeXt(nn.Module):
    def __init__(
        self, channels=[96, 192, 384, 768], layer_depths=[3, 3, 9, 3], num_classes=10
    ):
        super(ConvNeXt, self).__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=channels[0],
                kernel_size=4,
                stride=4,
                padding=0,
            ),
            nn.GroupNorm(num_groups=1, num_channels=channels[0]),
        )
        self.layer1 = self._make_conv_layer(
            channels=channels[0], layer_depth=layer_depths[0]
        )
        self.downsample1 = self._make_downsample_layer(channels=channels[0])
        self.layer2 = self._make_conv_layer(
            channels=channels[1], layer_depth=layer_depths[1]
        )
        self.downsample2 = self._make_downsample_layer(channels=channels[1])
        self.layer3 = self._make_conv_layer(
            channels=channels[2], layer_depth=layer_depths[2]
        )
        self.downsample3 = self._make_downsample_layer(channels=channels[2])
        self.layer4 = self._make_conv_layer(
            channels=channels[3], layer_depth=layer_depths[3]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.GroupNorm(num_groups=1, num_channels=channels[3]),
            nn.Linear(channels[3], num_classes),
        )

    def _make_conv_layer(self, channels, layer_depth):
        layers = []
        for _i in range(1, layer_depth):
            layers.append(InvertedBottleneck(channels=channels))
        return nn.Sequential(*layers)

    def _make_downsample_layer(self, channels):
        return nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        assert x.shape == (batch_size, 3, 256, 256)
        x = self.patchify(x)
        assert x.shape == (batch_size, 96, 64, 64)
        x = self.layer1(x)
        assert x.shape == (batch_size, 96, 64, 64)
        x = self.downsample1(x)
        assert x.shape == (batch_size, 192, 32, 32)
        x = self.layer2(x)
        assert x.shape == (batch_size, 192, 32, 32)
        x = self.downsample2(x)
        assert x.shape == (batch_size, 384, 16, 16)
        x = self.layer3(x)
        assert x.shape == (batch_size, 384, 16, 16)
        x = self.downsample3(x)
        assert x.shape == (batch_size, 768, 8, 8)
        x = self.layer4(x)
        assert x.shape == (batch_size, 768, 8, 8)
        x = self.head(x)
        assert x.shape == (batch_size, 10)

        return x


if __name__ == "__main__":
    model = ConvNeXt()
    x = torch.FloatTensor(1, 3, 256, 256)
    model.forward(x)
