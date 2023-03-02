import torch
import torch.nn as nn


class InvertedBottleneck(nn.Module):
    def __init__(self, channels, expansion=4, skip_connection=None):
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
                out_channels=channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
        )
        self.conv3 = nn.Conv2d(
            in_channels=channels * expansion,
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
        self.layer1 = self._make_layer()
        self.layer2 = self._make_layer()
        self.layer3 = self._make_layer()
        self.layer4 = self._make_layer()
        self.fc1 = nn.Linear(in_features=, out_features=num_classes)

    def _make_layer(self):
        pass

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patchify(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    model = ResNeXt(InvertedBottleneck, [3, 3, 9, 3])
    x = torch.tensor(4096, 3, 224, 224)
    model.forward(x)
