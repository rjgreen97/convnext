import torch
import torch.nn as nn
from timm.models.layers import DropPath


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


class ResNeXt(nn.Module):
    def __init__(self, block, depth, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_feature_maps = 96
        self.patchify = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=4, stride=4, padding=0
        )
        self.layer1 = self._make_layer(
            block=block, num_blocks=depth[0], feature_maps=96
        )
        self.layer2 = self._make_layer(
            block=block, num_blocks=depth[1], feature_maps=192
        )
        self.layer3 = self._make_layer(
            block=block, num_blocks=depth[2], feature_maps=384
        )
        self.layer4 = self._make_layer(
            block=block, num_blocks=depth[3], feature_maps=768
        )
        # self.classification_head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #   nn.GroupNorm(num_groups=1, num_channels=channels),
        #     nn.Linear(in_features=, out_features=num_classes)
        # )

    def forward(self, x):
        batch_size = x.shape[0]

        assert x.shape == (batch_size, 3, 256, 256)
        x = self.patchify(x)
        assert x.shape == (batch_size, 96, 64, 64)
        x = self.layer1(x)
        print("\n=========================")
        print(f"x.shape: {x.shape}")
        print("=========================")
        # x = self.layer2(x)

        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.classification_head(x)
        # return x


if __name__ == "__main__":
    model = ResNeXt(block=InvertedBottleneck, depth=[3, 3, 9, 3])
    x = torch.FloatTensor(4096, 3, 256, 256)
    model.forward(x)
