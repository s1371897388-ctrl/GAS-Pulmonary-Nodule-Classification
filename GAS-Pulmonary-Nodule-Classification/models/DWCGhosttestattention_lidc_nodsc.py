import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from models.fusion_module import TransFusionModule
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
#                                    bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

class DWConv3x3BNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, stride, groups):
        super(DWConv3x3BNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channel, out_channel, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channel = in_channel // divide
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=mid_channel),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=out_channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = torch.flatten(out, start_dim=1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x

class GhostModule(nn.Module):
    def __init__(self, in_channel, out_channel, s=2, kernel_size=1, stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channel = out_channel // s
        ghost_channel = intrinsic_channel * (s - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intrinsic_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(intrinsic_channel),
            nn.ReLU(inplace=True) if use_relu else nn.Sequential()
        )
        # self.primary_conv = nn.Sequential(
        #     DepthwiseSeparableConv(in_channels=in_channel, out_channels=intrinsic_channel, kernel_size=kernel_size,
        #                            stride=stride, padding=(kernel_size - 1) // 2, bias=False),
        #     nn.BatchNorm2d(intrinsic_channel),
        #     nn.ReLU(inplace=True) if use_relu else nn.Sequential()
        # )

        self.cheap_op = DWConv3x3BNReLU(in_channel=intrinsic_channel, out_channel=ghost_channel, stride=stride, groups=intrinsic_channel)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class GhostBottleneck(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        self.bottleneck = nn.Sequential(
            GhostModule(in_channel=in_channel, out_channel=mid_channel, use_relu=True),
            DWConv3x3BNReLU(in_channel=mid_channel, out_channel=mid_channel, stride=stride, groups=mid_channel) if self.stride > 1 else nn.Sequential(),
            SqueezeAndExcite(in_channel=mid_channel, out_channel=mid_channel) if use_se else nn.Sequential(),
            GhostModule(in_channel=mid_channel, out_channel=out_channel, use_relu=False)
        )

        if self.stride > 1:
            self.shortcut = DWConv3x3BNReLU(in_channel=in_channel, out_channel=out_channel, stride=stride, groups=1)
        else:
            self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = x1.view(b, c, -1).permute(2, 0, 1)  # [seq_len, batch_size, embed_dim]
        x2 = x2.view(b, c, -1).permute(2, 0, 1)
        attn_output, _ = self.cross_attention(x1, x2, x2)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
        return attn_output

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttentionBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)  # [seq_len, batch_size, embed_dim]
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
        return attn_output

class DWCGhost(nn.Module):
    def __init__(self, num_classes=2, leader=False, trans_fusion_info=None):
        super(DWCGhost, self).__init__()

        self.first_conv = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        ghost_model_setting = [
            # in, mid, out, kernel, stride, use_se
            # [16, 16, 16, 3, 1, True],
            # [16, 48, 24, 3, 2, True],
            # [24, 72, 24, 3, 1, True],
            [32, 32, 32, 3, 1, False],
            [32, 64, 64, 3, 2, True],
            [64, 128, 64, 3, 1, True],
        ]

        layers = []
        for in_channel, mid_channel, out_channel, kernel_size, stride, use_se in ghost_model_setting:
            layers.append(GhostBottleneck(in_channel=in_channel, mid_channel=mid_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, use_se=use_se))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.total_feature_maps = {}

        self.features = nn.Sequential(*layers)



        self.match_channel_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.match_channel_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False)
        self.match_channel_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, bias=False)

        self.self_attention_1 = SelfAttentionBlock(d_model=32, nhead=2)
        self.cross_attention_1 = CrossAttentionBlock(d_model=32, nhead=2)
        self.self_attention_2 = SelfAttentionBlock(d_model=64, nhead=2)
        self.cross_attention_2 = CrossAttentionBlock(d_model=64, nhead=2)

        self.last_stage = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),

            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),

            nn.ReLU6(inplace=True),
        )

        self.leader = leader

        if self.leader:
            if trans_fusion_info is not None:
                self.trans_fusion_module = TransFusionModule(trans_fusion_info[0], 10, model_num=trans_fusion_info[1])
            else:
                self.trans_fusion_module = TransFusionModule(256, 10)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.fc = nn.Linear(256, num_classes).to(self.device)

        self.register_hook()

    def register_hook(self):
        # self.extract_layers = ['features[0]', 'features[1]', 'features[2]']
        self.extract_layers = ['features.0', 'features.1', 'last_stage']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)

    def forward(self, x):
        x = self.first_conv(x)

        x1 = x  # Save input for first residual connection
        x = self.features[0](x)
        # print("x", x.shape)
        x = self.match_channel_1(x)
        x = self.self_attention_1(x)
        x1 = self.match_channel_1(x1)  # Match channel dimensions
        x1_copy = x1.clone().detach()
        x1_attn = self.cross_attention_1(x1_copy, x)
        x += x1  # First residual connection
        # print("x", x.shape)

        x2 = x  # Save output of second GhostBottleneck for second residual connection
        # print("x2", x2.shape)
        x = self.features[1](x)
        # print("x", x.shape)
        x = self.match_channel_2(x)
        # print("x", x.shape)
        x = self.self_attention_2(x)

        # print("x", x.shape)
        x2 = self.match_channel_3(x2)  # Match channel dimensions
        x2_copy = x2.clone().detach()
        x2_attn = self.cross_attention_2(x2_copy, x)

        x2_downsampled = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=1)  # Downsample x2

        # print("x2_downsampled", x2_downsampled.shape)
        x += x2_downsampled  # Second residual connection

        x = self.features[2](x)
        # print("x", x.shape)
        x = self.last_stage(x)







        if self.leader:
            trans_fusion_output = self.trans_fusion_module(x)

        x3 = self.global_avg_pool(x)
        x3 = x3.view(x3.size(0), -1)
        # print("x3", x3.shape)
        x3 = self.fc(x3)

        if self.leader:
            return x3, trans_fusion_output
        else:
            return x3

        # x = torch.flatten(x, start_dim=1)
        # x = self.classifier(x)
        # return x

# Testing the model with a sample input
model = DWCGhost(num_classes=2).to(device)
# x = torch.randn(1, 3, 32, 32)
# output = model(x)
# print(output.shape)
summary(model, (3, 32, 32))


current_dir = os.path.dirname(os.path.abspath(__file__))

# 将标准输出重定向到一个文件
original_stdout = sys.stdout
with open(os.path.join(current_dir, 'DWCGhost_lidc_summary.txt'), 'w') as f:
    sys.stdout = f  # 将标准输出重定向到文件
    print(f"Model device: {device}")  # 打印设备信息到文件
    summary(model, (3, 32, 32))  # 打印summary到文件中
    sys.stdout = original_stdout  # 恢复标准输出

# 打印成功保存的消息
print("Model summary saved to 'model_summary.txt'")







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
# 日期：2024/5/29
