import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载自己的网络
from modeltests.DWCGhosttest1_lidc import DWCGhost

class_num = 2
inplanes = 3  # 需要根据模型定义进行修改[                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        =
outplanes = 20  # 需要根据模型定义进行修改
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = DWCGhost(num_classes=class_num, inplanes=inplanes, outplanes=outplanes).to(device)

# 修改加载状态字典的代码，将"weight"映射到"fc_layer.weight"
state_dict = torch.load('../experiments/LIDC_IDRI_DWCGhost_div1e-5_sd1000_fusion10/checkpoints/modelleader_best.pth', map_locaon=device)
new_state_dict = {}
for key, value in state_dict.items():
    if key == 'fc_layer.weight':
        new_state_dict[key] = torch.Tensor(value)
    else:
        new_state_dict[key] = value
model_ft.load_state_dict(new_state_dict, strict=False)
if "cfg" in state_dict.keys():
    del state_dict["cfg"]
if "epoch" in state_dict.keys():
    del state_dict["epoch"]
if "index" in state_dict.keys():
    del state_dict["index"]
model_ft.load_state_dict(state_dict, strict=False)

model_ft.eval()

# Grad-CAM 类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x):
        output = self.model(x)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()
        weights = np.mean(gradients, axis=(2, 3))
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.size(-1), x.size(-2)))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
# def convert_to_color(heatmap):
#     heatmap = np.uint8(255 * heatmap)  # 将灰度热图转换为 0-255 范围的整数
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 应用彩色映射
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
#     return heatmap
#
# # 将灰度热图转换为彩色热图
# heatmap_color = convert_to_color(cam)

# def show_cam_on_image(img, mask):
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)
def apply_colormap_on_image(org_im, activation, colormap=cv2.COLORMAP_JET):
    """
    Apply heatmap on image
    Args:
        org_img (np.array): Original image
        activation_map (np.array): Activation map (heatmap)
        colormap: OpenCV Colormap to use for heatmap visualization
    Returns:
        (np.array): Original image overlaid with heatmap
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), colormap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(org_im)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# 加载输入图像并进行预处理
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    return img_tensor

# 示例图像路径
img_path = '../LIDC_IDRI/train/Benign/0003_MA000_slice000.png'

input_image = preprocess_image(img_path)
# print(input_image)
# 目标层，例如 DWCGhost 的某个层
target_layer = model_ft.ghost_module3.ghost_conv  # 需要根据实际情况修改

# 创建 GradCAM 对象
grad_cam = GradCAM(model_ft, target_layer)

# 生成 Grad-CAM 注意力热图
cam_mask = grad_cam(input_image)
# heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)

# 加载并转换原始图像
# 使用PIL库读取图像
img = Image.open(img_path)

# 转换为NumPy数组
# original_image = np.array(img)
original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

print(original_image.shape)
if original_image is None:
    print("Error: Could not read the image.")

original_image = cv2.resize(original_image, (224, 224))
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 将灰度热图转换为红蓝热图
colored_cam = apply_colormap_on_image(original_image, cam_mask, colormap=cv2.COLORMAP_JET)


# 显示结果
plt.imshow(original_image)
plt.axis('off')
plt.show()



# 作者：孙海滨
# 日期：2024/6/11
