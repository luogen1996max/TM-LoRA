# import pandas as pd
# import os
# from PIL import Image
# import io
# import base64
#
# df = pd.read_parquet('./data/rick_and_morty_image_and_text/data/train-00000-of-00001-1f95078297b4c0ef.parquet')

import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ==================== 配置 ====================
DATASET_NAME = "./data/rick_and_morty_image_and_text/data/"  # 数据集名称
OUTPUT_DIR = "tar_style"

# ==================== 创建输出文件夹 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 使用 datasets 库加载 ====================
print("正在加载数据集...")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"数据集大小: {len(dataset)} 个样本")
print(f"数据集特征: {dataset.features}")

# ==================== 提取并保存图像 ====================
for idx, sample in enumerate(tqdm(dataset, desc="保存图像")):
    try:
        # 获取图像（datasets 库会自动加载图像）
        image = sample["image"]

        # 保存图像
        image_path = os.path.join(OUTPUT_DIR, f"image_{idx:06d}.png")
        image.save(image_path)

    except Exception as e:
        print(f"保存第 {idx} 张图像时出错: {e}")

print(f"✅ 图像已保存到: {OUTPUT_DIR}")