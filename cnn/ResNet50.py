from torch import nn
import torch.nn.functional as F
from deep_rl.util.device import DEVICE


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bypass = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.bypass = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=stride, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        t = self.bn1(self.conv1(x))
        t = F.relu(t)
        t = self.bn2(self.conv2(t))
        t += self.bypass(x)
        return F.relu(t)


FIRST_CHANNEL = 64
#BLOCK_PER_LAYER_LIST = [2, 3, 6, 2]
BLOCK_PER_LAYER_LIST = [2, 2, 2, 2]


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=FIRST_CHANNEL,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(FIRST_CHANNEL)
        )
        self.layer1 = self.__make_layer(
            FIRST_CHANNEL, FIRST_CHANNEL, BLOCK_PER_LAYER_LIST[0], stride=1
        )
        self.layer2 = self.__make_layer(
            FIRST_CHANNEL, 128, BLOCK_PER_LAYER_LIST[1], stride=2
        )
        self.layer3 = self.__make_layer(
            128, 256, BLOCK_PER_LAYER_LIST[2], stride=2
        )
        self.layer4 = self.__make_layer(
            256, 512, BLOCK_PER_LAYER_LIST[3], stride=2
        )
        self.linear = nn.Linear(512 * 1, num_classes)
        self.to(DEVICE)

    def forward(self, x):
        t = F.relu(self.layer0(x))
        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)
        t = F.avg_pool2d(t, kernel_size=4)
        print(t.shape)
        t = t.reshape(-1, 512)
        t = self.linear(t)
        return t

    def __make_layer(self, in_channels, out_channels, num_blocks, stride):
        block_in_channel = in_channels
        stride_list = [stride] + [1] * (num_blocks - 1)
        block_list = []
        for s in stride_list:
            block_list.append(
                BasicBlock(block_in_channel, out_channels, s)
            )
            block_in_channel = out_channels
        return nn.Sequential(*block_list)