import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import lpips

# 自动检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载预训练的LPIPS模型
# lpips_model = models.lpips.LPIPS(net='alex')
lpips_model = lpips.LPIPS(net='alex').to(device)

# 加载两幅图像
image1 = Image.open('../evalu/checkpoint-10000/pre_result1.png').convert('RGB')
image2 = Image.open('../evalu/checkpoint-10000/pre_result1.png').convert('RGB')
# image2 = Image.open('../evalu/cnt/result1.png').convert('RGB')

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