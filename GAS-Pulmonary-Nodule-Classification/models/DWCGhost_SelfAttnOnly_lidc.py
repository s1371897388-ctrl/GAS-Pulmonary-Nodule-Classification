# 消融实验模型：仅使用Self-Attention（移除Cross-Attention）
# Ablation Model: Self-Attention Only (Cross-Attention removed)

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入基础组件
from models.DWCGhost_lidc import DepthwiseSeparableConv, GhostBottleneck

# 定义SelfAttentionBlock（从原始模型复制）
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

class DWCGhost_SelfAttnOnly(nn.Module):
    """
    消融实验模型：仅使用Self-Attention机制
    移除Cross-Attention机制，保留Self-Attention机制
    """
    def __init__(self, num_classes=2, leader=False, trans_fusion_info=None):
        super(DWCGhost_SelfAttnOnly, self).__init__()

        self.first_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        ghost_model_setting = [
            [32, 32, 32, 3, 1, False],
            [32, 64, 64, 3, 2, True],
            [64, 128, 64, 3, 1, True],
        ]

        layers = []
        for in_channel, mid_channel, out_channel, kernel_size, stride, use_se in ghost_model_setting:
            layers.append(GhostBottleneck(in_channel=in_channel, mid_channel=mid_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, use_se=use_se))
        
        self.total_feature_maps = {}
        self.features = nn.Sequential(*layers)

        self.match_channel_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False)
        self.match_channel_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False)
        self.match_channel_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, bias=False)

        # 只使用Self-Attention，不定义Cross-Attention
        self.self_attention_1 = SelfAttentionBlock(d_model=32, nhead=2)
        self.self_attention_2 = SelfAttentionBlock(d_model=64, nhead=2)

        self.last_stage = nn.Sequential(
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=5, stride=1),
            nn.ReLU6(inplace=True),
        )

        self.leader = leader

        if self.leader:
            from models.fusion_module import TransFusionModule
            if trans_fusion_info is not None:
                self.trans_fusion_module = TransFusionModule(trans_fusion_info[0], 10, model_num=trans_fusion_info[1])
            else:
                self.trans_fusion_module = TransFusionModule(256, 10)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

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

        x1 = x  # Save input for first residual connection
        x = self.features[0](x)
        
        # 使用Self-Attention（保留）
        x = self.match_channel_1(x)
        x = self.self_attention_1(x)
        # 注意：不使用Cross-Attention
        
        x += x1  # First residual connection

        x2 = x  # Save output for second residual connection
        x = self.features[1](x)
        
        # 使用Self-Attention（保留）
        x = self.match_channel_2(x)
        x = self.self_attention_2(x)
        # 注意：不使用Cross-Attention

        x2 = self.match_channel_3(x2)  # Match channel dimensions
        x2_downsampled = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=1)  # Downsample x2
        x += x2_downsampled  # Second residual connection

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

