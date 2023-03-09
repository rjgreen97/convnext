import torch
import torch.nn as nn

from src.model.inverted_bottleneck import InvertedBottleneck


class ConvNeXt(nn.Module):

    """
    My ConvNeXt model takes in a list of channels, a list of layer depths, and a number of classes. 
    The list of channels is used to define the number of feature maps in each layer of the model. The list of
    layer depths is used to define the number of inverted bottleneck blocks in each layer of the model. The number
    of classes is used to define the number of output classes to predict for in the final fully connected layer. 
    The stem of the model is a 4x4 convolution with a stride of 4, followed by a layer normalization.
    By using a kernel size of 4 and a stride of 4, we effectivly patchify the input image, much like a vision transformer.
    The downsampling is performed by the injection a 2x2 convolution with a stride of 2 between layers. Finally, the head of 
    the model consist of global average pooling, followed by flatten, followed by layer normalization into a fully 
    connected layer with the number of output nodes equal to the number of classes.
    """

    def __init__(
        self, channels=[96, 192, 384, 768], layer_depths=[3, 3, 9, 3], num_classes=100
    ):
        super(ConvNeXt, self).__init__()
        self.stem = nn.Sequential(
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
        self.layer2 = self._make_conv_layer(
            channels=channels[1], layer_depth=layer_depths[1]
        )
        self.layer3 = self._make_conv_layer(
            channels=channels[2], layer_depth=layer_depths[2]
        )
        self.layer4 = self._make_conv_layer(
            channels=channels[3], layer_depth=layer_depths[3], add_downsample=False
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.GroupNorm(num_groups=1, num_channels=channels[3]),
            nn.Linear(channels[3], num_classes),
        )

    def _make_conv_layer(self, channels, layer_depth, add_downsample=True):
        layers = []
        for _i in range(1, layer_depth):
            layers.append(InvertedBottleneck(channels=channels))
        if add_downsample:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=channels))
            layers.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, 256, 256)
        x = self.stem(x)
        assert x.shape == (batch_size, 96, 64, 64)
        x = self.layer1(x)
        assert x.shape == (batch_size, 192, 32, 32)
        x = self.layer2(x)
        assert x.shape == (batch_size, 384, 16, 16)
        x = self.layer3(x)
        assert x.shape == (batch_size, 768, 8, 8)
        x = self.layer4(x)
        assert x.shape == (batch_size, 768, 8, 8)
        x = self.head(x)
        assert x.shape == (batch_size, 100)

        return x


if __name__ == "__main__":
    model = ConvNeXt()
    x = torch.FloatTensor(1, 3, 256, 256)
    model.forward(x)
