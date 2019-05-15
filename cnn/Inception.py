from torch import nn
import torch.nn.functional as F
import torch
from deep_rl.util.device import DEVICE


def inception_k5():
    conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
    return [conv1, pool, conv2]

def inception_k3():
    conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
    return [conv1, pool, conv2]

def forward(conv1, pool, conv2, images):
    t = F.relu(conv1(images))
    t = pool(t)
    return F.relu(conv2(t))

class SimpleInception(nn.Module):
    def __init__(self):
        super(SimpleInception, self).__init__()
        self.conv1_5, self.pool_5, self.conv2_5 = inception_k5()
        self.conv1_3, self.pool_3, self.conv2_3 = inception_k3()
        self.fc1 = nn.Linear(32*16*16, 16*16)
        self.fc2 = nn.Linear(16*16, 10)
        self.to(DEVICE)

    def forward(self, images):
        o5 = forward(self.conv1_5, self.pool_5, self.conv2_5, images)
        o3 = forward(self.conv1_3, self.pool_3, self.conv2_3, images)
        t = torch.cat([o3, o5], dim=1)
        t = t.reshape(-1, 32*16*16)
        t = self.fc1.forward(t)
        t = F.relu(t)
        t = self.fc2.forward(t)
        return t