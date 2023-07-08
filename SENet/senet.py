import torch
import torch.nn as nn
import math

class SeNet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SeNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        print("-----",y.size())
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

senet = SeNet(512)
input = torch.ones([2, 512, 26, 26])
print(senet(input))