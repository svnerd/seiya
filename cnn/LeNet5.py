from torch import nn
import torch.nn.functional as F
from deep_rl.util.device import DEVICE

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1) # out: 28x28x6
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2) # out 14x14x16
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1) # out: 10x10x16
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2) # out 5x5x16
        self.fc_1 = nn.Linear(5*5*16, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, 10)
        self.to(DEVICE)

    def forward(self, x):

        x = F.relu(self.conv_1(x))
        x = self.maxpool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.maxpool_2(x)
        x = x.reshape(-1, 400)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return self.fc_3(x)