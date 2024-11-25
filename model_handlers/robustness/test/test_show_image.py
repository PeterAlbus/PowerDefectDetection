import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# imgs_test = np.load('../Datasets/elec_test_cln/images.npy')
print("111")
imgs_test_fsgm_01 = np.load('./Attack_generation/attack_param_FGSM_fgsm_01/FGSM_100_adv_preds_labels_3000.npy.npy')
imgs_test_BIM_01 = np.load('./Attack_generation/attack_param_BIM_bim_01/BIM_30_advs.npy')
imgs_test_fsgm_03 = np.load('./Attack_generation/attack_param_PGD_pgd_06/pgd_30_advs.npy')
imgs_test_rfsgm_02 = np.load('./Attack_generation/attack_param_RFGSM_rfgsm_02/RFGSM_30_advs.npy')
imgs_test_EAD_01 = np.load('./Attack_generation/attack_param_EAD_ead_01/EAD_30_advs.npy')
img_test_UAP_01 = np.load('./Attack_generation/attack_param_UAP_uap_01/UAP_30_advs.npy')
img_test_OM_01 = np.load('./Attack_generation/attack_param_OM_om_01/OM_30_advs.npy')
# imgs_test_fsgm_01 = np.load('./Attack_generation/attack_param_FGSM_fgsm_04/FGSM_30_advs.npy')
# imgs_test_fsgm_02 = np.load('./Attack_generation/attack_param_FGSM_fgsm_05/FGSM_30_advs.npy')
# imgs_test_fsgm_03 = np.load('./Attack_generation/attack_param_FGSM_fgsm_06/FGSM_30_advs.npy')
# print(imgs_test.shape)
labels_test = np.load('../Datasets/elec_test_cln/labels.npy')
print(labels_test.shape)
# print(labels_test[:10])

# 2. 反标准化操作
print("222")
mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])
unnormalize = transforms.Normalize(
    mean=(-mean / std).tolist(),
    std=(1.0 / std).tolist()
)
to_pil = transforms.ToPILImage()


def get_variable_name(variable):
    for name, value in locals().items():
        if value is variable:
            return name
    return None


def show_image_compare(image_npy_1, image_npy_2, image_npy_3, image_npy_4, label_npy, image_index):
    print("333")
    image_npy_1 = torch.tensor(image_npy_1[image_index])
    image_npy_1 = unnormalize(image_npy_1)
    image_npy_1 = to_pil(image_npy_1)
    # image_npy_1 = np.transpose(image_npy_1, (1, 2, 0))
    image_npy_2 = torch.tensor(image_npy_2[image_index])
    image_npy_2 = unnormalize(image_npy_2)
    image_npy_2 = to_pil(image_npy_2)
    # image_npy_2 = np.transpose(image_npy_2, (1, 2, 0))
    image_npy_3 = torch.tensor(image_npy_3[image_index])
    image_npy_3 = unnormalize(image_npy_3)
    image_npy_3 = to_pil(image_npy_3)
    # image_npy_3 = np.image_npy_3(image_npy_3, (1, 2, 0))
    image_npy_4 = torch.tensor(image_npy_4[image_index])
    image_npy_4 = unnormalize(image_npy_4)
    image_npy_4 = to_pil(image_npy_4)
    # image_npy_4 = np.transpose(image_npy_4, (1, 2, 0))

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(image_npy_1)
    plt.title('origin image')
    plt.subplot(2, 2, 2)
    plt.imshow(image_npy_2)
    plt.title('UAP image')
    plt.subplot(2, 2, 3)
    plt.imshow(image_npy_3)
    plt.title('FSGM image')
    plt.subplot(2, 2, 4)
    plt.imshow(image_npy_4)
    plt.title('EAD image')
    plt.show()


show_image_compare(imgs_test_BIM_01, img_test_UAP_01, imgs_test_fsgm_01, imgs_test_EAD_01, labels_test, 2)

# 显示每个攻击的图片对比
# image_index = 2
# image_origin = torch.tensor(imgs_test[image_index])
# image_origin = unnormalize(image_origin)
# image_origin = to_pil(image_origin)
# # image_origin = np.transpose(image_origin, (1, 2, 0))
# imgs_test_BIM_01 = torch.tensor(imgs_test_BIM_01[image_index])
# imgs_test_BIM_01 = unnormalize(imgs_test_BIM_01)
# imgs_test_BIM_01 = to_pil(imgs_test_BIM_01)
# # image_fsgm_01 = np.transpose(image_fsgm_01, (1, 2, 0))
# image_fsgm_02 = torch.tensor(imgs_test_rfsgm_02[image_index])
# image_fsgm_02 = unnormalize(image_fsgm_02)
# image_fsgm_02 = to_pil(image_fsgm_02)
# # image_fsgm_02 = np.transpose(image_fsgm_02, (1, 2, 0))
# image_fsgm_03 = torch.tensor(imgs_test_fsgm_03[image_index])
# image_fsgm_03 = unnormalize(image_fsgm_03)
# image_fsgm_03 = to_pil(image_fsgm_03)
# # image_fsgm_03 = np.transpose(image_fsgm_03, (1, 2, 0))
#
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.imshow(image_origin)
# plt.title('origin image')
# plt.subplot(2, 2, 2)
# plt.imshow(imgs_test_BIM_01)
# plt.title('BIM image')
# plt.subplot(2, 2, 3)
# plt.imshow(image_fsgm_02)
# plt.title('rfgsm_02 image')
# plt.subplot(2, 2, 4)
# plt.imshow(image_fsgm_03)
# plt.title('pgd_01 image')
# plt.show()
