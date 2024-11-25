import timm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models import create_model

import model

# 定义超参数
num_classes = 10
batch_size = 32
learning_rate = 0.001
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载和预处理数据
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.ImageFolder(root='/mnt/d/Code/Python/RobustnessEval/Data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = datasets.ImageFolder(root='/mnt/d/Code/Python/RobustnessEval/Data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model = create_model('repvit_m0_9', pretrained=False, num_classes=num_classes)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

        val_accuracy = correct.double() / total
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        torch.save(trained_model.state_dict(), f'model/rep/rep_epoch_{epoch}_acc_{val_accuracy:.4f}.pth')

    return model

# 开始训练
trained_model = train_model(model, criterion, optimizer, num_epochs)

# 保存模型
torch.save(trained_model.state_dict(), 'model/reprepvit_m0_9_10class.pth')
