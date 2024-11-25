import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import importlib

# 设置路径
image_path = '../Datasets/elec_test_cln/selected_images.npy'
label_path = '../Datasets/elec_test_cln/selected_labels.npy'
# model_path = '../Models/weights/last_model_Epoch_65_0.9937.pt'
model_path = '../Models/UserModel/model/resnet18/last_model_Epoch_30_0.9791.pt'
model_name = 'Models.UserModel.ResNet2_elec'

# 添加项目根路径到系统路径
sys.path.append("{}/../".format(os.path.dirname(os.path.realpath(__file__))))

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义的ResNet模型路径
module_user = importlib.import_module(model_name)
model = module_user.getModel()
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 加载数据
images = np.load(image_path)
one_hot_labels = np.load(label_path)

# 将one-hot编码转换为整数标签
labels = np.argmax(one_hot_labels, axis=1)
print(images.shape)
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])
images = images.transpose((0, 2, 3, 1))
print(images.shape)
image_input = []
for i in range(images.shape[0]):
    image_input.append(transform(images[i]))

# 创建数据集和数据加载器
tensor_x = torch.stack(image_input)  # 转换为张量
print(tensor_x.shape)
tensor_y = torch.Tensor(labels).long()  # 转换为长整型张量

dataset = TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# 测试模型并展示结果
results = []
true_labels = []
pred_labels = []

with torch.no_grad():
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # softmax
        outputs = nn.Softmax(dim=1)(outputs)
        # 转换为百分比，保留两位小数
        outputs = (outputs * 100).cpu().numpy().round(2)
        print(i)
        print(outputs)

        results.append((inputs.cpu().numpy(), targets.cpu().numpy(), preds.cpu().numpy()))
        true_labels.append(targets.cpu().numpy())
        pred_labels.append(preds.cpu().numpy())

        if i >= 29:  # 只展示30条测试结果
            break

# 计算准确率
accuracy = accuracy_score(np.concatenate(true_labels), np.concatenate(pred_labels))
print(f'Testing Accuracy: {accuracy * 100:.2f}%')

# 打印前30条测试结果
for i, (input_img, true_label, pred_label) in enumerate(results):
    print(f'Image {i + 1}: True Label = {true_label[0]}, Predicted Label = {pred_label[0]}, Correct = {true_label[0] == pred_label[0]}')
