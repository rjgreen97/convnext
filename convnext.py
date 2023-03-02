import torch
import torch.nn as nn


class InvertedBottleneck(nn.Module):
    def __init__(self, channels, skip_connection=None):
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
                out_channels=channels * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
        )
        self.conv3 = nn.Conv2d(
            in_channels=channels * 4,
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
    def __init__(self, block, layer_list, num_classes=10):
        super(ResNeXt, self).__init__()
        self.patchify = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=4, stride=4, padding=0
        )

    def _make_layer(self):
        pass

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patchify(x)


if __name__ == "__main__":
    model = ResNeXt(InvertedBottleneck, [3, 3, 9, 3])
    # x = torch.tensor(, 3, , ,)
    # model.forward(x)
