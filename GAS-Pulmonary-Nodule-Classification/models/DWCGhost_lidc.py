import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from models.fusion_module import TransFusionModule
# import torchvision.models as models
# from torchstat import stat
# from thop import profile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#加入了深度可分离卷积
##no attention
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias, groups=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
# class DepthwiseSeparableConv1(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
#                                    bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias, groups=1)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
# class DWConv3x3BNReLU(nn.Sequential):
#     def __init__(self, in_channel, out_channel, stride, groups):
#         super(DWConv3x3BNReLU, self).__init__(
#             nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU6(inplace=True),
#         )
class DWConv3x3BNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, stride, groups):
        super(DWConv3x3BNReLU, self).__init__(
            DepthwiseSeparableConv(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
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

        # self.primary_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channel, out_channels=intrinsic_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        #     nn.BatchNorm2d(intrinsic_channel),
        #     nn.ReLU(inplace=True) if use_relu else nn.Sequential()
        # )
        self.primary_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channel, out_channels=intrinsic_channel, kernel_size=kernel_size,
                                   stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(intrinsic_channel),
            nn.ReLU(inplace=True) if use_relu else nn.Sequential()
        )

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
            # self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)

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
            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            DepthwiseSeparableConv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),

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
        # self.device = torch.device("cpu" if torch.cuda.is_available() else "cuda:0")
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
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),

            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=5, stride=1),

            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),

            nn.ReLU6(inplace=True),
        )

        self.leader = leader

        if self.leader:
            if trans_fusion_info is not None:
                self.trans_fusion_module = TransFusionModule(trans_fusion_info[0], 10, model_num=trans_fusion_info[1])
            else:
                self.trans_fusion_module = TransFusionModule(256, 10)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

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
model = DWCGhost(num_classes=2).to("cuda")
# # x = torch.randn(1, 3, 32, 32)
# # output = model(x)
# # print(output.shape)
summary(model, (3, 32, 32))
# print(stat(model, (3, 32, 32)))

# 将模型及其所有子模块递归地移动到 CUDA 设备上
# device = torch.device("cpu")
# model = model.to(device)
#

# # 检查每个模块的设备属性
# for name, module in model.named_modules():
#     print(f"{name}: {module.device if hasattr(module, 'device') else 'cpu'}")
#
# # 假设这是你的输入数据
# input_data = torch.randn(1, 3, 32, 32)
#
# # 将输入数据移动到相同的 CUDA 设备上
# input_data = input_data.to(device)
#
# # 检查张量的设备属性
# print(f"Model device: {next(model.parameters()).device}")  # 检查模型参数的设备属性
# print(f"Input data device: {input_data.device}")  # 检查输入数据的设备属性
#
# print(stat(model, (3, 32, 32)))






#计算所占显存
# def calculate_parameters_memory(model):
#     param_memory = 0
#     for param in model.parameters():
#         param_memory += param.numel() * param.element_size()
#     return param_memory / (1024 * 1024)  # Convert to MB
#
# model = DWCGhost()
# memory_usage = calculate_parameters_memory(model)
# print(f"Model parameters memory usage: {memory_usage:.2f} MB")
#
#
#
#
#
#
# #Max RAM Requirement
# def calculate_memory_usage(model, input_size):
#     # 初始化输入张量
#     input_data = torch.randn(*input_size).to('cuda')
#
#     # 计算参数内存
#     param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
#
#     # 前向传播内存（激活值内存）
#     activations_memory = 0
#     hooks = []
#
#     def forward_hook(module, input, output):
#         nonlocal activations_memory
#         activations_memory += output.numel() * output.element_size()
#
#     # 注册前向钩子
#     for layer in model.children():
#         hooks.append(layer.register_forward_hook(forward_hook))
#
#     # 运行前向传播
#     model.to('cuda')
#     output = model(input_data)
#
#     # 反向传播内存（梯度内存）
#     gradients_memory = 0
#     for param in model.parameters():
#         gradients_memory += param.numel() * param.element_size() * 2  # 梯度 + 临时变量
#
#     # 计算总内存需求
#     total_memory = (param_memory + activations_memory + gradients_memory) / (1024 * 1024)  # 转换为MB
#
#     # 清理钩子
#     for hook in hooks:
#         hook.remove()
#
#     return total_memory
#
# input_size = (1, 3, 32, 32)  # 批大小为1，输入为3通道32x32图像
# max_ram = calculate_memory_usage(model, input_size)
# print(f"Max RAM Requirement: {max_ram:.2f} MB")
#
#
#
# dummy_input = torch.randn(1, 3, 32, 32).to("cuda")
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
#
#
#
#
#
#
# import torch
# import torchvision.models as models
#
# # 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 创建虚拟输入并将其移动到GPU
# dummy_input = torch.randn(1, 3, 32, 32).to(device)
#
# # 前向传播
# output = model(dummy_input)
#
# # 查看显存使用情况
# print('当前显存使用量: ', torch.cuda.memory_allocated(device) / 1024**2, 'MB')
# print('最大显存使用量: ', torch.cuda.max_memory_allocated(device) / 1024**2, 'MB')
#
#
# #FLOPs  Total
# #Self-distillation 14.3M*2  Feature Fusion 0.05*1   DWCGhost 27.2*3
#
# # 计算模型参数内存占用
# params_memory = sum(p.numel() * p.element_size() for p in model.parameters())
# print(f"Total memory used by parameters: {params_memory / (1024**2):.2f} MB")
#
#
# from ptflops import get_model_complexity_info
# flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,print_per_layer_stat=True)
# print("%s %s" % (flops, params))
#
# import torch
# import torchvision.models as models
#
# # 加载模型
# # 创建虚拟输入
# input_data = torch.randn(1, 3, 32, 32).to('cuda')
#
# # 将模型移动到 GPU
# model = model.to('cuda')
#
# # 前向传播
# with torch.cuda.device('cuda'):
#     output = model(input_data)
#     torch.cuda.synchronize()  # 等待 GPU 完成所有任务
#
# # 获取当前显存使用量（以字节为单位）
# current_mem_allocated = torch.cuda.memory_allocated()
# print(f"当前显存使用量: {current_mem_allocated / 1024**2} MB")
#
# # 获取最大显存使用量（以字节为单位）
# max_mem_allocated = torch.cuda.max_memory_allocated()
# print(f"最大显存使用量: {max_mem_allocated / 1024**2} MB")
#
# # 计算推断过程中的内存消耗（包括参数、激活值等）
# memory_usage_bytes = max_mem_allocated - current_mem_allocated
# print(f"推断过程中的内存消耗: {memory_usage_bytes / 1024**2} MB")
#
# # 估算 Total MemR+W
# # 这里的估算可以基于模型参数加载、激活值计算等内存操作的总量来进行
# # 可以根据实际需求进一步细化和精确估算
# total_mem_rw = memory_usage_bytes  # 简单示例，实际应根据具体情况进行精确估算
# print(f"估算的 Total MemR+W: {total_mem_rw / 1024**2} MB")
import torch

iterations = 300   # 重复计算的轮次
device = torch.device("cuda:0")
model.to(device)

random_input = torch.randn(1, 3, 32, 32).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
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
