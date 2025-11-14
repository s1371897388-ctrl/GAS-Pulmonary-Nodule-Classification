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

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out

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

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = x1.view(b, c, -1).permute(2, 0, 1)
        x2 = x2.view(b, c, -1).permute(2, 0, 1)
        attn_output, _ = self.cross_attention(x1, x2, x2)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
        return attn_output

class DWCGhost_NoResidual(nn.Module):
    """
    无残差连接消融实验模型
    移除所有残差连接（第一个和第二个都移除）
    保留：Self-Attention + Cross-Attention + Enhanced DWSGhost
    """
    def __init__(self, num_classes=2, leader=False, trans_fusion_info=None):
        super(DWCGhost_NoResidual, self).__init__()

        self.first_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        ghost_model_setting = [
            # in, mid, out, kernel, stride, use_se
            [32, 32, 32, 3, 1, False],
            [32, 64, 64, 3, 2, True],
            [64, 128, 64, 3, 1, True],
        ]

        layers = []
        for in_channel, mid_channel, out_channel, kernel_size, stride, use_se in ghost_model_setting:
            layers.append(GhostBottleneck(in_channel=in_channel, mid_channel=mid_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, use_se=use_se))
        self.total_feature_maps = {}

        self.features = nn.Sequential(*layers)

        # 注意：虽然不使用残差连接，但为了保持代码结构，我们仍然定义这些层
        # 但实际上在forward中不会使用它们
        self.match_channel_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.match_channel_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False)
        self.match_channel_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, bias=False)

        self.self_attention_1 = SelfAttentionBlock(d_model=32, nhead=2)
        self.cross_attention_1 = CrossAttentionBlock(d_model=32, nhead=2)
        self.self_attention_2 = SelfAttentionBlock(d_model=64, nhead=2)
        self.cross_attention_2 = CrossAttentionBlock(d_model=64, nhead=2)

        self.last_stage = nn.Sequential(
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU6(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        self.leader = leader
        if self.leader:
            if trans_fusion_info is not None:
                self.trans_fusion_module = TransFusionModule(trans_fusion_info[0], 10, model_num=trans_fusion_info[1])
            else:
                self.trans_fusion_module = TransFusionModule(256, 10)

        self.register_hook()

    def register_hook(self):
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

        # ========== 第一个残差连接（移除） ==========
        # 注意：不再保存 x1，也不进行第一个残差连接
        # 直接通过第一个GhostBottleneck和注意力模块
        x = self.features[0](x)
        x = self.match_channel_1(x)
        x = self.self_attention_1(x)
        # 注意：虽然计算了cross_attention，但不用于残差连接
        # 为了保持代码结构，我们仍然计算它，但不使用结果
        # x1 = x  # 不再需要保存
        # x1_copy = x1.clone().detach()
        # x1_attn = self.cross_attention_1(x1_copy, x)  # 计算但不使用
        # x += x1  # 移除：第一个残差连接

        # ========== 第二个残差连接（移除） ==========
        # 注意：不再保存 x2，也不进行第二个残差连接
        # 直接通过第二个GhostBottleneck和注意力模块
        x = self.features[1](x)
        x = self.match_channel_2(x)
        x = self.self_attention_2(x)
        # 注意：虽然计算了cross_attention，但不用于残差连接
        # x2 = x  # 不再需要保存
        # x2 = self.match_channel_3(x2)  # 不再需要
        # x2_copy = x2.clone().detach()
        # x2_attn = self.cross_attention_2(x2_copy, x)  # 计算但不使用
        # x2_downsampled = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=1)  # 不再需要
        # x += x2_downsampled  # 移除：第二个残差连接

        x = self.features[2](x)
        x = self.last_stage(x)

        if self.leader:
            trans_fusion_output = self.trans_fusion_module(x)

        x3 = self.global_avg_pool(x)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc(x3)

        if self.leader:
            return x3, trans_fusion_output
        else:
            return x3

# 作者：孙海滨
# 日期：2024/11/7

