import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# 数据预处理
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
}

# 数据加载
data_dir = '/mnt/d/Code/Python/RobustnessEval/Data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Swin Transformer 模型
model_name = 'swin_tiny_patch4_window7_224'
model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
model = model.to(device)

# 加载 checkpoint
checkpoint_path = 'model/swin_tiny_patch4_window7_224.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

checkpoint_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
checkpoint_dict.pop('head.weight')
checkpoint_dict.pop('head.bias')
model.load_state_dict(checkpoint_dict, strict=False)

num_classes = 10
model.head = torch.nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


# 训练函数
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            torch.save(model.state_dict(), f'model/Swin/swin_transformer_model_{epoch}_{epoch_acc}.pth')

    return model


# 训练模型
model = train_model(model, criterion, optimizer, num_epochs=25)

# 保存模型
torch.save(model.state_dict(), 'swin_transformer_model.pth')

# 加载模型
model.load_state_dict(torch.load('swin_transformer_model.pth'))
model = model.to(device)

# 测试模型
model.eval()
test_accuracy = 0
test_samples = 0
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_accuracy += torch.sum(preds == labels.data)
        test_samples += len(labels)

test_accuracy = test_accuracy.double() / test_samples
print(f'Test Accuracy: {test_accuracy:.4f}')
