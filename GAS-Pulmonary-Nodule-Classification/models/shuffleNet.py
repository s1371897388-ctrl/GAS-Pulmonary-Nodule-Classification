import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape: b, num_channels, h, w  -->  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channelshuffle
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class shuffleNet_unit(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups):
        super(shuffleNet_unit, self).__init__()

        mid_channels = out_channels//4
        self.stride = stride
        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups
        self.GConv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        self.GConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.GConv1(x)
        out = channel_shuffle(out, groups=self.groups)
        out = self.DWConv(out)
        out = self.GConv2(out)
        short = self.shortcut(x)
        if self.stride == 2:
            out = F.relu(torch.cat([out, short], dim=1))
        else:
            out = F.relu(out + short)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, groups, num_layers, num_channels, num_classes=1000):
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1, bias=False),  # 修改第一层卷积的stride为1
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self.make_layers(24, num_channels[0], num_layers[0], groups)
        self.stage3 = self.make_layers(num_channels[0], num_channels[1], num_layers[1], groups)
        self.stage4 = self.make_layers(num_channels[1], num_channels[2], num_layers[2], groups)

        self.globalpool = nn.AvgPool2d(kernel_size=2, stride=1)  # 修改全局池化的kernel_size和stride
        self.fc = nn.Linear(num_channels[2], num_classes)

    def make_layers(self, in_channels, out_channels, num_layers, groups):
        layers = []
        layers.append(shuffleNet_unit(in_channels, out_channels - in_channels, 2, groups))
        in_channels = out_channels
        for i in range(num_layers - 1):
            layers.append(shuffleNet_unit(in_channels, out_channels, 1, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试网络结构
#ShuffleNet_g3
model = ShuffleNet(groups=4, num_layers=[4, 8, 4], num_channels=[144, 288, 576], num_classes=2).to(device)  # 假设分类数为10

summary(model, (3, 32, 32))  # 打印模型结构及参数量





# def ShuffleNet_g1(**kwargs):
#     num_layers = [4, 8, 4]
#     num_channels = [144, 288, 576]
#     model = ShuffleNet(1, num_layers, num_channels, **kwargs)
#     return model
#
#
# def ShuffleNet_g2(**kwargs):
#     num_layers = [4, 8, 4]
#     num_channels = [200, 400, 800]
#     model = ShuffleNet(2, num_layers, num_channels, **kwargs)
#     return model
#
#
# def ShuffleNet_g3(**kwargs):
#     num_layers = [4, 8, 4]
#     num_channels = [240, 480, 960]
#     model = ShuffleNet(3, num_layers, num_channels, **kwargs)
#     return model
#
#
# def ShuffleNet_g4(**kwargs):
#     num_layers = [4, 8, 4]
#     num_channels = [272, 544, 1088]
#     model = ShuffleNet(4, num_layers, num_channels, **kwargs)
#     return model
#
#
# def ShuffleNet_g8(**kwargs):
#     num_layers = [4, 8, 4]
#     num_channels = [384, 768, 1536]
#     model = ShuffleNet(8, num_layers, num_channels, **kwargs)
#     return model

from thop import profile


dummy_input = torch.randn(1, 3, 32, 32).to("cuda")
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


import torch

iterations = 300   # 重复计算的轮次
device = torch.device("cuda:0")
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
