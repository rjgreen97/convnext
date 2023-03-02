import torch
import torch.nn as nn


class InvertedBottleneck(nn.Module):
    def __init__(self, channels, skip_connection=None, hidden_layer_multiplier=4):
        super(InvertedBottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                stride=3,
                padding=0,
            ),
            nn.LayerNorm(normalized_shape=channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * hidden_layer_multiplier,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
        )
        self.conv3 = nn.Conv2d(
            in_channels=channels * hidden_layer_multiplier,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.skip_connection = skip_connection

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.skip_connection:
            residual = self.skip_connection(residual)
        out += residual
        return out


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()

    def _make_layer(self):
        pass

    def forward(self, x):
        pass
