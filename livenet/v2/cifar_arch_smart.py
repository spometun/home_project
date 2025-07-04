import torch
from torch import nn
from livenet.backend.core import Context
from livenet.v2.smart_conv import SmartConv2d

def conv_block_resnet9(context, in_channels, out_channels, pool=False):
    layers = [SmartConv2d(context, in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU6(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def pick_block(context, channels: int):
    conv = SmartConv2d(context, in_channels=channels, out_channels=channels, kernel_size=1, groups=channels, bias=False)
    # torch.nn.init.constant_(conv.weight, 1.0)
    return conv


class ResNet9SmallSmart(nn.Module):
    def __init__(self, context, in_channels, num_classes, device):
        super().__init__()
        self.device = device
        self.context = context

        self.conv1 = conv_block_resnet9(context, in_channels, 16)
        self.conv2 = conv_block_resnet9(context, 16, 32, pool=True)
        self.res1 = nn.Sequential(conv_block_resnet9(context, 32, 32), conv_block_resnet9(context, 32, 32))

        self.conv3 = conv_block_resnet9(context, 32, 64, pool=True)
        self.conv4 = conv_block_resnet9(context, 64, 128, pool=True)
        self.res2 = nn.Sequential(conv_block_resnet9(context, 128, 128), conv_block_resnet9(context, 128, 128))

        self.pick2 = pick_block(context, 32)
        self.pick4 = pick_block(context, 128)

        self.classifier = nn.Sequential(nn.MaxPool2d(4), SmartConv2d(context, 128, num_classes, 1), nn.Flatten())
        self.to(self.device)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + self.pick2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) # + self.pick4(out)
        out = self.classifier(out)
        return out

    def internal_loss(self):
        loss = torch.tensor(0., device=self.device)
        leaf_modules = [m for m in self.modules() if len(list(m.children())) == 0]
        for leaf in leaf_modules:
            try:
                leaf_loss = leaf.internal_loss()
                loss += leaf_loss
            except AttributeError:
                pass
        return loss

    def get_stats_strs(self, clear: bool):
        strs = ""
        for full_name, module in self.named_modules():
            if len(list(module.children())) != 0:
                continue
            try:
                m = module.get_stats_str()
                strs += f"{full_name}: {m}\n"
                if clear:
                    module.clear_stats()
            except AttributeError:
                pass
        return strs
