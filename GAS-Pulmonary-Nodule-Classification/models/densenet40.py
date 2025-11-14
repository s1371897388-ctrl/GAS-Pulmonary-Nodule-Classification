import torch
from torchsummary import summary
from models.densenet_cifar import densenet_40  # 假设你的模型定义在名为 your_module.py 的文件中
import sys
import os
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = densenet_40().to(device)
summary(model, (1, 32, 32))

# 将标准输出重定向到一个文件
# original_stdout = sys.stdout
# with open(os.path.join(current_dir, 'model_summary.txt'), 'w') as f:
#     sys.stdout = f  # 将标准输出重定向到文件
#     print(f"Model device: {device}")  # 打印设备信息到文件
#     summary(model, (1, 32, 32))  # 打印summary到文件中
#     sys.stdout = original_stdout  # 恢复标准输出
#
# # 打印成功保存的消息
# print("Model summary saved to 'model_summary.txt'")







#
# summary(model, (1, 32, 32))  # 输入图片的尺寸是 32x32，通道数是 1






# 作者：孙海滨
# 日期：2024/4/2
