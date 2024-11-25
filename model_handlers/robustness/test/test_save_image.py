import numpy as np
import os
from torchvision import transforms
import torch

# 假设你的标签是one-hot编码的，我们需要一个函数来转换为类别索引
def one_hot_to_index(one_hot_labels):
    return np.argmax(one_hot_labels, axis=1)

# 读取数据
imgs_test = np.load('../Datasets/elec_test_cln/images.npy')
labels_test = np.load('../Datasets/elec_test_cln/labels.npy')

# 转换标签为类别索引
labels_index = one_hot_to_index(labels_test)

# 反标准化操作
mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])
unnormalize = transforms.Normalize(
    mean=(-mean / std).tolist(),
    std=(1.0 / std).tolist()
)
to_pil = transforms.ToPILImage()

# 创建保存图片的函数
def save_images(imgs, labels_index, attack_num):
    for i in range(imgs.shape[0]):
        label_index = labels_index[i+attack_num-100]
        img = imgs[i]
        img = torch.tensor(img)
        img = unnormalize(img)
        img = to_pil(img)
        label_folder = f"generation/class_{label_index}"
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        file_name = f"{attack_num}_adv_{i}.png"
        img.save(os.path.join(label_folder, file_name))

# # 遍历不同的对抗样本文件
for attack_num in range(100, 3100, 100):
    attack_file = f'./Attack_generation/attack_param_RFGSM_rfgsm_01/RFGSM_{attack_num}_advs_3000.npy'
    imgs_test_fsgm = np.load(attack_file)
    save_images(imgs_test_fsgm, labels_index, attack_num)

# save_images(imgs_test, labels_index, "cln")


print("图片保存完成。")
