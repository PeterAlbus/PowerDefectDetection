import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 数据集路径
val_data_path = './data/power_new_train'

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# 加载数据集
val_dataset = ImageFolder(root=val_data_path, transform=transform)

# 随机选择30张图片
num_samples = 3000
indices = random.sample(range(len(val_dataset)), num_samples)

selected_images = []
selected_labels = []

for idx in indices:
    image, label = val_dataset[idx]
    selected_images.append(image.numpy())
    one_hot_label = np.zeros(len(val_dataset.classes))
    one_hot_label[label] = 1
    selected_labels.append(one_hot_label)

# 转换为numpy数组
selected_images = np.array(selected_images)
selected_labels = np.array(selected_labels)

# 保存为npy文件
np.save('images.npy', selected_images)
np.save('labels', selected_labels)

print(f'Saved {num_samples} images and labels to selected_images.npy and selected_labels.npy')
