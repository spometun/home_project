import torch
import torch.nn as nn
import torch.nn.functional as F
from livenet.backend.core import Context


class ResBlockClassic(nn.Module):
    def __init__(self, input_channels: int, internal_channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(input_channels, internal_channels, 1)
        self.c2 = nn.Conv2d(internal_channels, internal_channels, 3, groups=internal_channels, padding="same")
        self.c3 = nn.Conv2d(internal_channels, input_channels, 1)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.bn3 = nn.BatchNorm2d(input_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.c1(x)))
        y = F.relu(self.bn2(self.c2(y)))
        y = F.relu(self.bn3(self.c3(y)))
        return x + y


class EffNet(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self._alpha = 0.01
        self.context = Context(self)
        self.blocks = dict()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.av_pool = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3, groups=16)
        self.conv3 = nn.Conv2d(16, 16, 1)
        self.conv4 = nn.Conv2d(16, 32, 1)
        self.r1 = ResBlockClassic(32, 64)
        self.conv5 = nn.Conv2d(32, 64, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, groups=64, stride=2)
        self.conv7 = nn.Conv2d(64, 40, 1)
        self.r2 = ResBlockClassic(40, 128)
        self.conv8 = nn.Conv2d(40, 128, 1)
        self.conv9 = nn.Conv2d(128, 128, 3, groups=128, stride=2)
        self.conv10 = nn.Conv2d(128, 64, 1)

        self.conv11 = nn.Conv2d(64, 128, 1)
        self.conv12 = nn.Conv2d(128, 128, 3, groups=128, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1b = nn.Linear(64, 10)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)

        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(40)

        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(64)

        self.bn11 = nn.BatchNorm2d(128)
        self.bn12 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, 10)
        self.to(self.device)


    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        x = F.relu6(self.bn4(self.conv4(x)))
        x = self.r1(x)
        x = F.relu6(self.bn5(self.conv5(x)))
        x = F.relu6(self.bn6(self.conv6(x)))
        x = F.relu6(self.bn7(self.conv7(x)))
        x = self.r2(x)
        x = F.relu6(self.bn8(self.conv8(x)))
        x = F.relu6(self.bn9(self.conv9(x)))
        x = F.relu6(self.bn10(self.conv10(x)))

        x = F.relu6(self.bn11(self.conv11(x)))
        # x = self.dr1(x)
        x = F.relu6(self.bn12(self.conv12(x)))

        x = self.av_pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x=self.fc(x)
        return x

    def internal_loss(self):
        loss = torch.tensor(0., device=self.device)
        for param in self.parameters():
            if len(param.data.shape) > 1:
                # loss += self._alpha * torch.sum(torch.abs(param)) / param.data.numel()
                loss += self._alpha * torch.sum(torch.square(param)) / param.data.numel()
        return loss


def conv_block_resnet9(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU6(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, device):
        super().__init__()
        self.device = device
        self.context = Context(self)
        self._alpha = 0.05

        self.conv1 = conv_block_resnet9(in_channels, 64)
        self.conv2 = conv_block_resnet9(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block_resnet9(128, 128), conv_block_resnet9(128, 128))

        self.conv3 = conv_block_resnet9(128, 256, pool=True)
        self.conv4 = conv_block_resnet9(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block_resnet9(512, 512), conv_block_resnet9(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))
        self.to(self.device)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    def internal_loss(self):
        loss = torch.tensor(0., device=self.device)
        size={0: 32, 1: 32, 2: 16, 3: 16, 4: 16, 5: 8, 6: 4, 7:4}
        params = list(self.parameters())
        for i in range(8):
            mult = size[i] * size[i]
            ind = i * 4
            p = params[ind]
            summa = torch.sum(torch.abs(p))
            loss += self._alpha * mult * summa
        # for param in self.parameters():
        #     if len(param.data.shape) > 1:
        #         loss += self._alpha * torch.sum(torch.abs(param)) / param.data.numel()
        #         loss += self._alpha * torch.sum(torch.square(param)) / param.data.numel()
        return loss


def pick_block(channels: int):
    conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=channels, bias=False)
    return conv
    # torch.nn.init.constant_(conv.weight, 1.0)
    # return nn.Sequential(*layers)


class ResNet9Small(nn.Module):
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

        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Conv2d(128, 10, 1), nn.Flatten())
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
        size={0: 32, 1: 32, 2: 16, 3: 16, 4: 16, 5: 8, 6: 4, 7:4, 8:1}
        params = list(self.parameters())
        for i in range(9):
            mult = size[i] * size[i]
            ind = i * 4
            p = params[ind]
            summa = torch.sum(torch.abs(p))
            loss += self._alpha * mult * summa
        # for param in self.parameters():
        #     if len(param.data.shape) > 1:
        #         loss += self._alpha * torch.sum(torch.abs(param)) / param.data.numel()
        #         loss += self._alpha * torch.sum(torch.square(param)) / param.data.numel()
        return loss
