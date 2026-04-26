import os.path

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import lpips


def my_LPIPS(ori_imag, pre_imag):
    # 加载两幅图像
    image1 = Image.open(ori_imag).convert('RGB')
    image2 = Image.open(pre_imag).convert('RGB')


    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载预训练的LPIPS模型
    # lpips_model = models.lpips.LPIPS(net='alex')
    lpips_model = lpips.LPIPS(net='alex').to(device)

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

if __name__ == '__main__':

    tar_imag = f'../evalu/火影/our'

    path_file = ['checkpoint-' + str(i) for i in range(1000, 20001, 1000)]
    epoch_list = []
    lr_list = []
    loss_list = []
    for file in path_file:
        tar_imag2 = tar_imag + '/' + file

        eva_value = 0
        for i in range(1, 5):
            epoch_list.append(file)

            ori_imag = f'../evalu/cnt/result{i}.png'
            pre_imag = f'{tar_imag2}/pre_result{i}.png'
            print(pre_imag)

            value = my_LPIPS(ori_imag, pre_imag)
            value = value.detach().item()
            eva_value = eva_value + value
            lr_list.append(value)
            loss_list.append(' ')
        loss_list.insert(-1, eva_value/4)
        loss_list.pop()

    import pandas as pd
    record = dict()
    record['epoch'] = epoch_list
    record['lpip'] = lr_list
    record['lpips'] = loss_list
    record = pd.DataFrame(record)
    import time

    record_name = time.strftime("%Y_%m_%d_%H.csv", time.localtime())
    # record.to_csv(r'%s/%s' % (os.path.split(tar_imag)[0], record_name), index=False)
    record.to_csv(r'%s/%s' % (tar_imag, record_name), index=False)
