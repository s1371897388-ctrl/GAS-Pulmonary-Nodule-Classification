import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.fusion_module import TransFusionModule
from models.self_distillation import SelfDistillationModel, SelfDistillationModule
class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes, se_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.se_ratio = se_ratio
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_planes, in_planes // se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // se_ratio, in_planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        out = self.squeeze(x).view(batch_size, channels)
        weights = self.excitation(out).view(batch_size, channels, 1, 1)
        return x * weights.expand_as(x)

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=1, device='cuda'):
        super(GhostModule, self).__init__()

        self.primary_channels = out_channels // ratio
        self.ghost_channels = out_channels - self.primary_channels

        self.primary_conv = nn.Conv2d(
            in_channels,
            self.primary_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        ).to(device)

        self.ghost_conv = nn.Conv2d(
            self.primary_channels,
            self.ghost_channels,
            dw_size,
            stride,
            padding,
            groups=self.primary_channels,
            bias=False
        ).to(device)
        self.squeeze_excitation = SqueezeExcitation(out_channels).to(device)

    def forward(self, x):
        primary_out = self.primary_conv(x)
        ghost_out = self.ghost_conv(primary_out)
        output = torch.cat([primary_out, ghost_out], dim=1)#concat
        output = output.to('cuda')  # 将输入张量转换为 CUDA 张量
        weights = self.squeeze_excitation(output)
        output1 = output * weights.expand_as(output)
        avg_output = F.adaptive_avg_pool2d(output, (1, 1))  # 全局平均池化
        weighted_output = torch.sigmoid(avg_output) * avg_output  # 使用 sigmoid 函数进行加权

        return weighted_output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # 调整输入通道数为输出通道数
        self.adjust_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.self_attention = nn.MultiheadAttention(out_channels, num_heads=1)
        self.cross_attention = nn.MultiheadAttention(out_channels, num_heads=1)
        self.embed_dim = out_channels

    def forward(self, x):
        # 调整输入通道数为输出通道数
        x = self.adjust_conv(x)

        out = self.conv1(x)
        out = self.relu(out)

        height, width = out.size(2), out.size(3)

        out = out.permute(0, 2, 3, 1)  # 将通道维度放到最后

        out = out.contiguous().view(out.size(0), -1, self.embed_dim)  # 将形状变换为 (batch_size, seq_len, embed_dim)

        # 自注意力
        out_self, _ = self.self_attention(out, out, out)

        # 交叉注意力
        out_cross, _ = self.cross_attention(out_self, out_self, out)

        out = out_cross.view(out.size(0), out.size(2), height, width)  # 将形状还原为 (batch_size, channels*2, height, width)
        return out

class TripleGhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=1, leader=False, trans_fusion_info=None, num_classes=2):
        super(TripleGhostModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)  # 将输出通道数设置为128，与后续的GhostModule输入通道数匹配

        self.ghost_module1 = GhostModule(out_channels, out_channels, kernel_size, ratio, dw_size, stride, padding)
        self.ghost_module2 = GhostModule(out_channels, out_channels, kernel_size, ratio, dw_size, stride, padding)
        self.ghost_module3 = GhostModule(out_channels, out_channels, kernel_size, ratio, dw_size, stride, padding)

        self.residual1 = ResidualBlock(out_channels, out_channels)
        self.residual2 = ResidualBlock(out_channels, out_channels)
        self.leader = leader
        if self.leader:
            if trans_fusion_info is not None:
                self.trans_fusion_module = TransFusionModule(trans_fusion_info[0], 8, model_num=trans_fusion_info[1])
            else:
                self.trans_fusion_module = TransFusionModule(64, 5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x1 = self.conv(x)

        x2 = self.ghost_module1(x1)

        x1_res = self.residual1(x1)

        x3 = self.ghost_module2(x2)

        x2_res = self.residual2(x2)

        x4 = self.ghost_module3(torch.add(x3, x1_res))

        x5 = torch.add(x4, x2_res)
        x5 = x5.to(device)
        if self.leader:
            trans_fusion_output = self.trans_fusion_module(x5)

        global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        x5 = global_avg_pool(x5)

        x5 = x5.view(x5.size(0), -1)

        fc_layer = nn.Linear(64, 2).to('cuda')  # 假设 num_classes 是分类任务的类别数量
        x5 = fc_layer(x5).to(device)
        if self.leader:
            return x5, trans_fusion_output
        else:
            return x5

# 组合所有模块的完整模型示例
# 组合所有模块的完整模型示例
class CombinedModel(nn.Module):
    def __init__(self, input_channel, layer_num, num_classes=2):
        super(CombinedModel, self).__init__()
        self.self_distillation = SelfDistillationModel(input_channel, layer_num)
        self.triple_ghost = TripleGhostModule(in_channels=4, out_channels=64, leader=True, num_classes=num_classes)

    def forward(self, x):
        x = self.self_distillation(x)
        x, trans_fusion_output = self.triple_ghost(x)
        return x, trans_fusion_output

# 使用示例
input_channel = 64
layer_num = 4
model = CombinedModel(input_channel=input_channel, layer_num=layer_num)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_tensor = torch.randn(1, 64, 64, 64).to(device)  # (batch_size, channels, height, width)
output, trans_fusion_output = model(input_tensor)

print("CombinedModel Output Shape:", output.shape)
print("TransFusionModule Output Shape:", trans_fusion_output.shape if trans_fusion_output is not None else "None")

# 打印模型结构
summary(model, input_size=(1, 64, 64))

# 作者：孙海滨
# 日期：2024/5/21
