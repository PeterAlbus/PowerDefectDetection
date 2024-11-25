import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os

# 定义ResNet模型（你提供的代码）
from ResNet2_elec import getModel

# 数据集路径
# train_data_path = './data/train'
# val_data_path = './data/val'
train_data_path = './data/power_new_train'
val_data_path = './data/power_new_val'

# 超参数
batch_size = 32
learning_rate = 1e-3
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
print(train_dataset.class_to_idx)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 获取模型
model = getModel()
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 调整学习率函数
def adjust_learning_rate(optimizer, epoch):
    if epoch in [80, 120, 160]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    elif epoch == 180:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5


# 训练和验证循环
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # 验证模型
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Validation Acc: {val_acc:.4f}')

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model/resnet18/best_model.pt')
        torch.save(model.state_dict(), f'model/resnet18/last_model_Epoch_{epoch}_{epoch_acc:.4f}.pt')


# 开始训练
train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)
