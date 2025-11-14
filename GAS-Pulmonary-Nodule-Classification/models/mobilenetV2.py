

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchstat import stat
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()

        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for t, c, n, s in self.cfgs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(Block(in_planes, c, t, stride))
                in_planes = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(device)
        out = F.relu(self.bn1(self.conv1(x))).to(device)
        out = self.layers(out).to(device)
        out = F.relu(self.bn2(self.conv2(out))).to(device)
        out = F.avg_pool2d(out, 4).to(device)
        out = out.view(out.size(0), -1).to(device)
        out = self.linear(out).to(device)
        return out

model = MobileNetV2(num_classes=2)
# x = torch.randn(1, 3, 32, 32)
# output = model(x)
# print(output.shape)
summary(model, (3, 32, 32))
from thop import profile

dummy_input = torch.randn(1, 3, 32, 32).to("cuda")
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


import torch

iterations = 300   # 重复计算的轮次
device = torch.device("cuda")
model.to(device)

random_input = torch.randn(1, 3, 32, 32).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
# 作者：孙海滨
# 日期：2024/6/27
