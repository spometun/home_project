import torch
from torch import nn
from livenet.backend.core import Context
from livenet.v2.smart_conv import SmartConv2d

def conv_block_resnet9(in_channels, out_channels, pool=False):
    layers = [SmartConv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU6(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def pick_block(channels: int):
    conv = SmartConv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=channels, bias=False)
    # torch.nn.init.constant_(conv.weight, 1.0)
    return conv


class ResNet9SmallSmart(nn.Module):
    def __init__(self, in_channels, num_classes, device):
        super().__init__()
        self.device = device
        self.context = Context(self)
        self._alpha = 0.05

        self.conv1 = conv_block_resnet9(in_channels, 16)
        self.conv2 = conv_block_resnet9(16, 32, pool=True)
        self.res1 = nn.Sequential(conv_block_resnet9(32, 32), conv_block_resnet9(32, 32))

        self.conv3 = conv_block_resnet9(32, 64, pool=True)
        self.conv4 = conv_block_resnet9(64, 128, pool=True)
        self.res2 = nn.Sequential(conv_block_resnet9(128, 128), conv_block_resnet9(128, 128))

        self.pick2 = pick_block(32)
        self.pick4 = pick_block(128)

        self.classifier = nn.Sequential(nn.MaxPool2d(4), SmartConv2d(128, 10, 1), nn.Flatten())
        self.to(self.device)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + self.pick2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + self.pick4(out)
        out = self.classifier(out)
        return out

    def internal_loss(self):
        loss = torch.tensor(0., device=self.device)
        return loss
