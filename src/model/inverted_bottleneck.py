import torch
import torch.nn as nn

class InvertedBottleneck(nn.Module):

    """
    An Inverted Bottleneck Block begins with a 7x7 depthwise convolution followed by layer normalization, 
    which is then followed by a 1x1 convolution. A depthwise convolution in conjunction with a pointwise
    convolution is known as a depthwise seperable convolution. Depthwise seperable convolutions have become popular 
    in recent years due to their far superior computational efficiency and fewer parameters. The output of this depthwise 
    seperable convolution is then passed through a GELU activation function, and then through another 1x1 pointwise 
    convolution. The output of the 1x1 pointwise convolution is then added to the input of the block (forming a residual 
    connection), and the output of the block is returned.
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
