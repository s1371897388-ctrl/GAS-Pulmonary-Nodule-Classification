import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


import torch
import torch.nn as nn
import torch.nn.functional as F

def Hswish(x, inplace=True):
    return x * F.relu6(x + 3., inplace=inplace) / 6.

def Hsigmoid(x, inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.

# Squeeze-And-Excite模块
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y)
        y = Hsigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_channels, stride, se=True, nl='HS'):
        super(Bottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride = stride
        if se:
            self.se = SEModule(exp_channels)
        else:
            self.se = None
        self.conv1 = nn.Conv2d(in_channels, exp_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2 = nn.Conv2d(exp_channels, exp_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=exp_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_channels)
        self.conv3 = nn.Conv2d(exp_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out = self.bn2(self.conv2(out))
            out = self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class MobileNetV3_large(nn.Module):
    # (out_channels, kernel_size, exp_channels, stride, se, nl)
    cfg = [
        (16, 3, 16, 1, False, 'RE'),
        (24, 3, 64, 2, False, 'RE'),
        (24, 3, 72, 1, False, 'RE'),
        (40, 5, 72, 2, True, 'RE'),
        (40, 5, 120, 1, True, 'RE'),
        (40, 5, 120, 1, True, 'RE'),
        (80, 3, 240, 2, False, 'HS'),
        (80, 3, 200, 1, False, 'HS'),
        (80, 3, 184, 1, False, 'HS'),
        (80, 3, 184, 1, False, 'HS'),
        (112, 3, 480, 1, True, 'HS'),
        (112, 3, 672, 1, True, 'HS'),
        (160, 5, 672, 2, True, 'HS'),
        (160, 5, 960, 1, True, 'HS'),
        (160, 5, 960, 1, True, 'HS')
    ]

    def __init__(self, num_classes=17):
        super(MobileNetV3_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_channels=16)
        self.conv2 = nn.Conv2d(160, 960, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.conv3 = nn.Conv2d(960, 1280, 1, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(1280, num_classes, 1, stride=1, padding=0, bias=True)

    def _make_layers(self, in_channels):
        layers = []
        for out_channels, kernel_size, exp_channels, stride, se, nl in self.cfg:
            layers.append(
                Bottleneck(in_channels, out_channels, kernel_size, exp_channels, stride, se, nl)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = Hswish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = Hswish(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, out.size(2))  # 全局平均池化，使用特征图的大小
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        return out


# class MobileNetV3_small(nn.Module):
#     # (out_channels,kernel_size,exp_channels,stride,se,nl)
#     cfg = [
#         (16,3,16,2,True,'RE'),
#         (24,3,72,2,False,'RE'),
#         (24,3,88,1,False,'RE'),
#         (40,5,96,2,True,'HS'),
#         (40,5,240,1,True,'HS'),
#         (40,5,240,1,True,'HS'),
#         (48,5,120,1,True,'HS'),
#         (48,5,144,1,True,'HS'),
#         (96,5,288,2,True,'HS'),
#         (96,5,576,1,True,'HS'),
#         (96,5,576,1,True,'HS')
#     ]
#     def __init__(self,num_classes=17):
#         super(MobileNetV3_small,self).__init__()
#         self.conv1=nn.Conv2d(3,16,3,2,padding=1,bias=False)
#         self.bn1=nn.BatchNorm2d(16)
#         # 根据cfg数组自动生成所有的Bottleneck层
#         self.layers = self._make_layers(in_channels=16)
#         self.conv2=nn.Conv2d(96,576,1,stride=1,bias=False)
#         self.bn2=nn.BatchNorm2d(576)
#         # 卷积后不跟BN，就应该把bias设置为True
#         self.conv3=nn.Conv2d(576,1280,1,1,padding=0,bias=True)
#         self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)
#
#     def _make_layers(self,in_channels):
#         layers=[]
#         for out_channels,kernel_size,exp_channels,stride,se,nl in self.cfg:
#             layers.append(
#                 Bottleneck(in_channels,out_channels,kernel_size,exp_channels,stride,se,nl)
#             )
#             in_channels=out_channels
#         return nn.Sequential(*layers)
#
#     def forward(self,x):
#         out=Hswish(self.bn1(self.conv1(x)))
#         out=self.layers(out)
#         out=self.bn2(self.conv2(out))
#         se=SEModule(out.size(1))
#         out=Hswish(se(out))
#         out = F.avg_pool2d(out, 7)
#         out = Hswish(self.conv3(out))
#         out = self.conv4(out)
#         # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
#         a, b = out.size(0), out.size(1)
#         out = out.view(a, b)
#         return out

# 测试代码，跑通证明网络结构没问题
# def test():
#     net=MobileNetV3_small()
#     x=torch.randn(2,3,224,224)
#     y=net(x)
#     print(y.size())
#     print(y)
#
# if __name__=="__main__":
#     test()


# 示例测试



# 示例测试


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = MobileNetV3_large(num_classes=2).to(device)  # 例如设定 num_classes=10
summary(model, (3, 32, 32))   # 打印模型结构及参数量










from thop import profile


dummy_input = torch.randn(1, 3, 32, 32).to("cuda")
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置推理参数和设备
iterations = 300   # 重复计算的轮次
device = torch.device("cuda:0")

# 创建模型实例并移动到 GPU
model = MobileNetV3_large(num_classes=10)  # 替换成您的模型
model.to(device)
model.eval()  # 将模型设置为评估模式

# 准备随机输入张量并移动到 GPU
random_input = torch.randn(1, 3, 32, 32).to(device)

# 使用CUDA事件测量时间
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# 预热，执行几次来预热GPU
for _ in range(10):
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
        curr_time = starter.elapsed_time(ender)  # 计算时间
        times[iter] = curr_time

mean_time = times.mean().item()
fps = 1000 / mean_time  # 计算FPS
print("Inference time: {:.6f} ms, FPS: {:.2f}".format(mean_time, fps))


# 作者：孙海滨
# 日期：2024/7/5
