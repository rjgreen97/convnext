import torch
import torch.nn as nn


class InvertedBottleneck(nn.Module):
    def __init__(self, channels, skip_connection=None):
        super(InvertedBottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                stride=3,
                padding=0,
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
        self.skip_connection = skip_connection

    def forward(self, x):
        residual = x
        x = self.block(x)
        if self.skip_connection:
            residual = self.skip_connection(residual)
        return x


class ConvNeXt(nn.Module):
    def __init__(
        self, channels=[96, 192, 384, 768], layer_depths=[3, 3, 9, 3], num_classes=10
    ):
        super(ConvNeXt, self).__init__()
        self.patchify = nn.Conv2d(
            in_channels=3,
            out_channels=channels[0],
            kernel_size=4,
            stride=4,
            padding=0,
        )
        self.layer1 = self._make_layer(
            channels=channels[0], layer_depth=layer_depths[0]
        )
        self.layer2 = self._make_layer(
            channels=channels[1], layer_depth=layer_depths[1]
        )
        self.layer3 = self._make_layer(
            channels=channels[2], layer_depth=layer_depths[2]
        )
        self.layer4 = self._make_layer(
            channels=channels[3], layer_depth=layer_depths[3]
        )
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.LayerNorm(normalized_shape=channels[3]),
            nn.Linear(channels[3], num_classes),
        )

    # method that creates a layer of inverted bottleneck blocks with skip connections and downsampling once layer depth is reached
    def _make_layer(self, channels, layer_depth):
        layers = []
        for i in range(layer_depth):
            if i == 0:
                layers.append(
                    InvertedBottleneck(
                        channels=channels,
                        skip_connection=nn.Conv2d(channels, channels, 1, 2),
                    )
                )
            else:
                layers.append(InvertedBottleneck(channels=channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]

        assert x.shape == (batch_size, 3, 256, 256)
        x = self.patchify(x)
        assert x.shape == (batch_size, 96, 64, 64)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.classification_head(x)
        # return x


if __name__ == "__main__":
    model = ConvNeXt()
    x = torch.FloatTensor(4096, 3, 256, 256)
    model.forward(x)
