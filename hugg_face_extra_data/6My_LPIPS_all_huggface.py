import os.path

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import lpips

from skimage.metrics import structural_similarity as ssim
from skimage import io


def my_LPIPS(ori_imag, pre_imag):
    # 加载两幅图像
    image1 = Image.open(ori_imag).convert('RGB')
    image2 = Image.open(pre_imag).convert('RGB')


    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"使用设备: {device}")

    # 加载预训练的LPIPS模型
    # lpips_model = models.lpips.LPIPS(net='alex')
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)

    # 对图像进行预处理
    preprocess = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image1 = preprocess(image1).unsqueeze(0).to(device)
    image2 = preprocess(image2).unsqueeze(0).to(device)

    image1 = torch.tensor(image1)
    image2 = torch.tensor(image2)

    # 使用LPIPS模型计算相似性
    similarity_score = lpips_model(image1, image2)

    print(f"LPIPS Similarity: {similarity_score.item()}")
    return similarity_score


def my_ssim(gen_img, real_img):
    # 读取两幅图像
    image1 = io.imread(gen_img, as_gray=True)
    image2 = io.imread(real_img, as_gray=True)
    # 将image2调整为512x512
    # image2 = transform.resize(image2, (512, 512), preserve_range=True).astype(image1.dtype)
    # 计算SSIM
    ssim_score = ssim(image1, image2, data_range=1.0)
    return ssim_score

if __name__ == '__main__':

    num = '1'
    na = '龙珠'

    # tar_imag = f'./{na}/our_{num}'
    tar_imag = f'./{na}/normal_{num}'
    nor_imag = f"./{na}/cnt_{num}"

    path_file = ['checkpoint-' + str(i) for i in range(1000, 20001, 1000)]
    epoch_list = []
    lpips_list = []
    ssim_list = []

    for file in path_file:
        tar_imag2 = tar_imag + '/' + file

        eva_value = 0
        eva_ssmi_value = 0
        # for i in range(1, 5):
        epoch_list.append(file)
        for i in range(1, len(os.listdir(tar_imag2))+1):

            ori_imag = f'{nor_imag}/result{i}.png'
            pre_imag = f'{tar_imag2}/pre_result{i}.png'
            print(pre_imag)

            ####  计算LPIP
            value = my_LPIPS(ori_imag, pre_imag)
            value = value.detach().item()
            eva_value = eva_value + value
            ####  计算SSIM
            ssim_value = my_ssim(ori_imag, pre_imag)
            eva_ssmi_value = eva_ssmi_value + ssim_value

            # lr_list.append(value)
        lpips_list.append(eva_value/len(os.listdir(tar_imag2)))
        ssim_list.append(eva_ssmi_value/len(os.listdir(tar_imag2)))

    import pandas as pd
    record = dict()
    record['epoch'] = epoch_list
    record['lpips'] = lpips_list
    record['ssim'] = ssim_list
    record = pd.DataFrame(record)
    import time

    # record_name = time.strftime("%Y_%m_%d_%H.csv", time.localtime())
    record_name = f"{os.path.split(tar_imag)[-1]}.csv"
    # record.to_csv(r'%s/%s' % (os.path.split(tar_imag)[0], record_name), index=False)
    record.to_csv(r'%s/%s' % (os.path.split(tar_imag)[0], record_name), index=False)
