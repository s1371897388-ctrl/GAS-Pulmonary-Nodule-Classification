import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.resnet_lidc import BasicBlock
from resnet_lidc import resnet56

# 创建模型实例
model = resnet56(num_classes=2)

# 在此处添加训练和测试代码

# 定义数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder('LIDC_IDRI/train', transform=data_transforms)
test_dataset = datasets.ImageFolder('LIDC_IDRI/test', transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义模型


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 64
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 每个epoch结束后在测试集上评估模型性能
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'resnet56.pth')
# 作者：孙海滨
# 日期：2024/6/18
