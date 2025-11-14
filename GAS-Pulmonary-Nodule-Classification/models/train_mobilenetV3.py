# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
# import numpy as np
# import logging
# from datetime import datetime
# import os
# from models.mobilenetV2 import MobileNetV2
#
# # 创建日志目录
# log_dir = 'logs'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
#
# # 设置日志记录
# log_filename = os.path.join(log_dir, f'mobilenetV3_training1_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
# logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
#
# # 数据加载
# train_dataset = ImageFolder('../LIDC_IDRI/train', transform=transforms.ToTensor())
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#
# test_dataset = ImageFolder('../LIDC_IDRI/test', transform=transforms.ToTensor())
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
#
# # 训练
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MobileNetV2(num_classes=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
# # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08)
#
# criterion = nn.CrossEntropyLoss()
#
# # 保存初始参数
# logging.info(f'Model: MobileNetV3')
# logging.info(f'Optimizer: Adam')
# logging.info(f'Initial learning rate: 0.0001')
# logging.info(f'Loss function: CrossEntropyLoss')
#
# num_epochs = 96
# best_auc = 0.0
# best_model_path = 'mobilenetV2_best_model1.pth'
#
# for epoch in range(num_epochs):
#     model.train()
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     correct = 0
#     total = 0
#     predictions = []
#     targets = []
#     with torch.no_grad():
#         for imgs, lbls in test_loader:
#             imgs, lbls = imgs.to(device), lbls.to(device)
#             outputs = model(imgs)
#             _, predicted = torch.max(outputs, 1)
#             total += lbls.size(0)
#             correct += (predicted == lbls).sum().item()
#             predictions.extend(predicted.cpu().numpy())
#             targets.extend(lbls.cpu().numpy())
#
#     accuracy = 100 * correct / total
#     auc = roc_auc_score(np.array(targets), np.array(predictions))
#     f1 = f1_score(np.array(targets), np.array(predictions))
#     precision = precision_score(np.array(targets), np.array(predictions))
#     recall = recall_score(np.array(targets), np.array(predictions))
#     cm = confusion_matrix(np.array(targets), np.array(predictions))
#     specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#
#     logging.info(f'Epoch: {epoch+1}/{num_epochs}')
#     logging.info(f'Loss: {loss.item()}')
#     logging.info(f'Accuracy: {accuracy}%')
#     logging.info(f'AUC: {auc}')
#     logging.info(f'F1 Score: {f1}')
#     logging.info(f'Precision: {precision}')
#     logging.info(f'Recall (Sensitivity): {recall}')
#     logging.info(f'Specificity: {specificity}')
#
#     print(f'第 {epoch+1}/{num_epochs} 轮, 损失: {loss.item()}, 准确率: {accuracy}%, AUC: {auc}, F1: {f1}, 精度: {precision}, 敏感性: {recall}, 特异性: {specificity}')
#
#     # 保存当前最优模型
#     if auc > best_auc:
#         best_auc = auc
#         torch.save(model.state_dict(), best_model_path)
#         logging.info(f'Best model saved with AUC: {best_auc}')
#
# print(f'训练完成，最优模型保存在 {best_model_path}，AUC: {best_auc}')


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import logging
from datetime import datetime
import os
from models.mobilenetV3 import MobileNetV3_Large

# 创建日志目录
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置日志记录
log_filename = os.path.join(log_dir, f'mobilenetV3_training1_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

# 数据加载
train_dataset = ImageFolder('../LIDC_IDRI/train', transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = ImageFolder('../LIDC_IDRI/test', transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetV3_Large(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 保存初始参数
logging.info(f'Model: MobileNetV3')
logging.info(f'Optimizer: Adam')
logging.info(f'Initial learning rate: 0.0001')
logging.info(f'Loss function: CrossEntropyLoss')

num_epochs = 96
best_accuracy = 0.0
best_model_path = 'mobilenetV3_Large_best_model1.pth'
best_accuracy_info = None

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
            predictions.extend(predicted.cpu().numpy())
            targets.extend(lbls.cpu().numpy())

    accuracy = 100 * correct / total

    # 计算其他评估指标
    auc = roc_auc_score(np.array(targets), np.array(predictions))
    f1 = f1_score(np.array(targets), np.array(predictions))
    precision = precision_score(np.array(targets), np.array(predictions))
    recall = recall_score(np.array(targets), np.array(predictions))
    cm = confusion_matrix(np.array(targets), np.array(predictions))
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    logging.info(f'Epoch: {epoch+1}/{num_epochs}')
    logging.info(f'Loss: {loss.item()}')
    logging.info(f'Accuracy: {accuracy}%')
    logging.info(f'AUC: {auc}')
    logging.info(f'F1 Score: {f1}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall (Sensitivity): {recall}')
    logging.info(f'Specificity: {specificity}')

    print(f'第 {epoch+1}/{num_epochs} 轮, 损失: {loss.item()}, 准确率: {accuracy}%, AUC: {auc}, F1: {f1}, 精度: {precision}, 敏感性: {recall}, 特异性: {specificity}')

    # 保存当前最优模型（按准确率）
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_accuracy_info = {
            'epoch': epoch + 1,
            'loss': loss.item(),
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }
        torch.save(model.state_dict(), best_model_path)
        logging.info(f'Best model saved with Accuracy: {best_accuracy}%')

print(f'训练完成，最优模型保存在 {best_model_path}，最佳测试结果如下:')
print(best_accuracy_info)


# 作者：孙海滨
# 日期：2024/6/18
