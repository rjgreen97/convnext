import torch
import torch.nn as nn

class InvertedBottleneck(nn.Module):
    """
    Inverted Bottleneck Block: https://arxiv.org/abs/1801.04381
    """
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
