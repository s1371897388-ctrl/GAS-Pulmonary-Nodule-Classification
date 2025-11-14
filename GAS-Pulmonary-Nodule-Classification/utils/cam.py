import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.DWCGhostattention_lidc import DWCGhost
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_num = 2

# 你的模型权重路径
model_weights_path = '../experiments/LIDC_IDRI_DWCGhost_div1e-5_sd1000_fusion10/checkpoints/modelleader_best.pth'


# 初始化DWCGhost模型
DWCGhost = DWCGhost(num_classes=2)

# 确保使用正确的设备加载模型权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型权重
ckpt = torch.load(model_weights_path, map_location=device)
# 假设'weight'键下是state_dict
model_state_dict = ckpt['weight']

model_state_dict = {k: v for k, v in model_state_dict.items() if k in DWCGhost.state_dict()}
DWCGhost.load_state_dict(model_state_dict, strict=True)

# 将模型设置为评估模式
DWCGhost.eval()
DWCGhost.to(device)
# print(DWCGhost)
num_ftrs = DWCGhost.fc.in_features
DWCGhost.fc = nn.Linear(num_ftrs, class_num)

model_features = nn.Sequential(*list(DWCGhost.children())[:-2])

fc_weights = DWCGhost.state_dict()['fc.weight'].cpu().numpy()  #[2,2048]  numpy数组取维度fc_weights[0].shape->(2048,)
print(fc_weights.shape)

# 定义图像预处理函数
preprocess = transforms.Compose([
    # transforms.Resize(32),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载和预处理图像
img_path = '../LIDC_IDRI/test/Malignant/LIDC-IDRI-00152.jpg'
img = Image.open(img_path).convert('RGB')
img_tensor = preprocess(img).unsqueeze(0).to(device)
class_ = {0:'Benign', 1:'Malignant'}
# 假设我们简单地通过取前16个通道来减少通道数

# 检查模型第一层期望的输入通道数
first_layer = list(DWCGhost.children())[0]
if isinstance(first_layer, nn.Sequential):
    first_sub_layer = list(first_layer.children())[0]
    print(f"First sub-layer expected input channels: {first_sub_layer.in_channels}")
else:
    print(f"First layer expected input channels: {first_layer.in_channels}")

# 检查输入图像的通道数
print(f"Image tensor channels: {img_tensor.shape[1]}")

# 调试模型特征部分的具体层
print("Model features structure:")
for name, layer in model_features.named_children():
    print(name, layer)







features = model_features(img_tensor).detach().cpu().numpy()  #最终的特征图集合[1,2048,7,7]
logit = DWCGhost(img_tensor)


logit  = DWCGhost(img_tensor)
print("logit ", logit )
h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  #每个类别对应概率([0.9981, 0.0019])
print("h_x", h_x)
probs, idx = h_x.sort(0, True)      #输出概率升序排列
probs = probs.cpu().numpy()  #[0.9981, 0.0019]
idx = idx.cpu().numpy()  #[1, 0]
for i in range(class_num):
    print('{:.3f} -> {}'.format(probs[i], class_[idx[i]]))  #打印预测结果
print(' output for the top1 prediction: %s' % class_[idx[0]])  # y预测第一

# def returnCAM(feature_conv, weight_softmax, class_idx):
#     b, c, h, w = feature_conv.shape        #1,2048,7,7
#     output_cam = []
#     for idx in class_idx:  #输出每个类别的预测效果
#         cam = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))
#         #(1, 2048) * (2048, 7*7) -> (1, 7*7)
#         cam = cam.reshape(h, w)
#         cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
#         cam_img = np.uint8(255 * cam_img)  #Format as CV_8UC1 (as applyColorMap required)
#         output_cam.append(cam_img)
#     return output_cam
#
# CAMs = returnCAM(features, fc_weights, [idx[0]])  #输出预测概率最大的特征图集对应的CAM



# # 获取模型的最后一个卷积层名称
# finalconv_name = 'last_stage'
# features = None
# grads = None
# # 获取最后一个卷积层的输出和梯度
# def get_features_hook(module, input, output):
#     global features
#     features = output
#     print(f"Features shape: {features.shape}")
#     print("get_features_hook called")
#
# def get_grads_hook(module, grad_input, grad_output):
#     global grads
#     grads = grad_output[0]
#     print(f"Grads shape: {grads.shape}")
#     print("get_grads_hook called")
# # 注册钩子
# layer = DWCGhost._modules.get(finalconv_name)
# if layer is None:
#     raise ValueError(f"Layer {finalconv_name} not found in model.")
# layer.register_forward_hook(get_features_hook)
# layer.register_backward_hook(get_grads_hook)
# layer = dict(DWCGhost.named_modules()).get(finalconv_name)
#
# # 前向传播
# output = DWCGhost(img_tensor)
# pred_class = output.argmax(dim=1).item()
# # 计算梯度
# DWCGhost.zero_grad()
# class_loss = output[0, pred_class]
# class_loss.backward()
#
# # 确保 features 和 grads 变量已正确赋值
# if features is None or grads is None:
#     raise ValueError("Features or grads not defined. Check if hooks are correctly registered and called.")
#
# # 获取卷积层特征图和梯度
# features = features[0].cpu().data.numpy()
# grads = grads.cpu().data.numpy()
#
# # 计算权重
# weights = np.mean(grads, axis=(1, 2))
#
# # 生成CAM
# cam = np.zeros(features.shape[1:], dtype=np.float32)
# for i, w in enumerate(weights):
#     cam += w * features[i]
#
# cam = np.maximum(cam, 0)
# cam = cv2.resize(cam, (img.size[0], img.size[1]))
# cam = cam - np.min(cam)
# cam = cam / np.max(cam)
#
# # 可视化CAM
# heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
# heatmap = np.float32(heatmap) / 255
# img = np.array(img)
# superimposed_img = heatmap * 0.4 + img
# superimposed_img = np.clip(superimposed_img, 0, 1)
#
# # 显示结果
# plt.imshow(superimposed_img)
# plt.axis('off')
# plt.show()
































# # 定义图像预处理函数
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# # 加载和预处理图像
# img_path = '../LIDC_IDRI/test/Malignant/CT-Training-LC008_CT_4-seg_23.jpg'
# img = Image.open(img_path).convert('RGB')
# img_t = preprocess(img)
# img_t = torch.unsqueeze(img_t, 0)  # Add batch dimension
# torch.cuda.memory_summary()
# # 确保使用正确的设备
# img_t = img_t.to(device)


# # 正向传播获取特征图
# with torch.no_grad():
#     model_output = DWCGhost(img_t)
#
# # 假设你知道你的模型中最后一个卷积层的名称
# # 例如 'features[24]'，这取决于你的模型定义
# # 你需要根据你的模型结构来调整这个名称
# # 这里以 'features' 为例，具体名称根据你的模型定义进行修改
# last_conv_layer_name = 'features.3'
# last_conv_layer = DWCGhost._modules[last_conv_layer_name]
#
# # 假设你的目标类别索引是386
# target_category = 1
#
# # 获取目标类别的激活
# target_activation = model_output[0][target_category]
# target_activation = torch.unsqueeze(target_activation, 0)  # 确保是2D tensor
#
# # 计算目标类别激活相对于最后一个卷积层输出的梯度
# target_activation.backward(retain_graph=True)
#
# # 获取最后一个卷积层的梯度
# last_conv_layer_grad = last_conv_layer.weight.grad.data.cpu()
#
# # 计算每个特征图通道的重要性
# channel_importance = last_conv_layer_grad.abs().sum(1).squeeze()
#
# # 选择最重要的特征图通道
# top_channels = channel_importance.argsort(0)[::-1][:256]
#
# # 应用这些通道的重要性到原始特征图上
# last_conv_layer_output = last_conv_layer_forward = last_conv_layer(img_t)
# for i in range(256):
#     last_conv_layer_output[:, i] *= (top_channels[i] / channel_importance.max())
#
# # 将特征图转换为热图
# heatmap = last_conv_layer_output.mean(dim=0).cpu().numpy()
#
# # 将热图归一化到[0, 1]
# heatmap = cv2.resize(heatmap, (224, 224))
# heatmap = np.maximum(heatmap, 0)
# heatmap /= np.max(heatmap)
#
# # 将热图应用到原始图像上
# img = cv2.imread(img_path)
# heatmap = cv2.applyColorMap(heatmap * 255, cv2.COLORMAP_JET)
# heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
# superimposed_img = heatmap * 0.4 + img
#
# # 保存结果图像
# save_path = '../LIDC_IDRI/cam.jpg'
# cv2.imwrite(save_path, superimposed_img)
#
# print(f"CAM image saved at: {save_path}")


# 作者：孙海滨
# 日期：2024/6/19
