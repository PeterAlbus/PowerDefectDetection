import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from tqdm import tqdm

# 定义ResNet模型（你提供的代码）
from AlexNet_elec import getModel

# Define the transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 数据集路径
# train_data_path = './data/train'
# val_data_path = './data/val'
train_data_path = './data/power_new_train'
val_data_path = './data/power_new_val'

# 超参数
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(train_dataset.class_to_idx)

# 获取模型
model = getModel()
model = model.to(device)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Print every 100 batches
            print(f'Epoch {epoch}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.6f}')
            running_loss = 0.0


def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return val_loss, accuracy


model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 50
best_acc = 0.0
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    loss, val_acc = validate(model, device, val_loader, criterion)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'model/alexnet/alexnet_cifar10_best.pt')
    torch.save(model.state_dict(), f'model/alexnet/last_model_Epoch_{epoch}_{val_acc:.4f}.pt')
