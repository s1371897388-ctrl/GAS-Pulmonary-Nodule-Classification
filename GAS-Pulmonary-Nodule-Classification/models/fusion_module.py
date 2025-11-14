import torch.nn as nn

import math

from torchstat import stat


class FusionModule(nn.Module):
    def __init__(self, channel, numclass, sptial, model_num=2):#接受四个参数：channel表示输入特征图的通道数，numclass表示类别数，sptial表示平均池化的大小，model_num表示模型的数量student1/2，默认值为2。
        super(FusionModule, self).__init__()
        self.total_feature_maps = {}#存储特征图
        #2D卷积层conv1，输入通道数为channel * model_num，输出通道数为channel * model_num，卷积核大小为3x3，步长为1，填充为1，groups=channel*model_num表示按通道分组卷积
        self.conv1 = nn.Conv2d(channel * model_num, channel * model_num, kernel_size=3, stride=1, padding=1, groups=channel*model_num, bias=False)
        #创建一个2D批归一化层bn1，输入通道数为channel * model_num
        self.bn1 = nn.BatchNorm2d(channel * model_num)
        #创建一个ReLU激活函数relu1，True表示inplace激活，即直接在原始输入上进行操作。
        self.relu1 = nn.ReLU(True)
        #创建一个1x1的2D卷积层conv1_1，输入通道数为channel * model_num，输出通道数为channel，不分组卷积，不使用偏置项。
        self.conv1_1 = nn.Conv2d(channel * model_num, channel, kernel_size=1, groups=1, bias=False)
        #创建一个2D批归一化层bn1_1，输入通道数为channel。
        self.bn1_1 = nn.BatchNorm2d(channel)

        self.relu1_1 = nn.ReLU(True)
        #创建一个平均池化层avgpool2d，池化核大小为sptial，这里sptial是通过参数传入的平均池化大小。
        self.avgpool2d = nn.AvgPool2d(sptial)
        #创建一个全连接层fc2，输入大小为channel，输出大小为numclass，用于最终的类别预测。
        self.fc2 = nn.Linear(channel, numclass)

        #将传入的sptial值保存到实例变量sptial中。
        self.sptial = sptial
        #对模块中的每个子模块进行初始化。对于卷积层，采用标准正态分布初始化权重；对于批归一化层，将权重初始化为1，偏置项初始化为0。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #调用register_hook方法，注册一个钩子用于捕获中间特征图。
        self.register_hook()

    def forward(self, x):#定义了前向传播方法，接收输入x，代表输入的特征图。

        x = self.relu1(self.bn1((self.conv1(x))))#先对输入x进行卷积、批归一化、ReLU激活操作。

        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))#再对经过第一次处理的特征图进行卷积、批归一化、ReLU激活操作。

        x = self.avgpool2d(x)#处理后的特征图进行平均池化
        x = x.view(x.size(0), -1)#将特征图展平为一维向量。


        out = self.fc2(x)#将展平后的特征向量输入全连接层，得到最终的类别预测。

        return out

    def register_hook(self):#定义了register_hook方法，用于注册钩子。主要是用来捕获特定层的中间特征图，可以在训练或推断过程中用于进一步的分析或可视化。

        self.extract_layers = ['relu1_1']#定义了要提取特征图的层，这里是relu1_1。

#定义了一个内部函数get_activation，用于创建钩子函数。这个钩子函数将中间特征图存储到maps字典中。
        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name+str(output.device)] = output

            return get_output_hook
#定义了一个内部函数add_hook，用于为模型的指定层添加钩子。
        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)


class TransFusionModule(nn.Module):

    def __init__(self, channel, sptial, model_num=2):
        super(TransFusionModule, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(True)
        self.conv1_1 = nn.Conv2d(channel, channel * model_num, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel * model_num)
        self.relu1_1 = nn.ReLU(True)

        self.sptial = sptial

    def forward(self, input):#定义了前向传播方法，接收输入input，代表输入的特征图。

        x = self.relu1(self.bn1((self.conv1(input))))#先对输入input进行卷积、批归一化、ReLU激活操作
        out = self.relu1_1(self.bn1_1(self.conv1_1(x)))#再对经过第一次处理的特征图进行卷积、批归一化、ReLU激活操作。

        return out#输出特征图
