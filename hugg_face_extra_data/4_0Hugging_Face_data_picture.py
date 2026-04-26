# import pandas as pd
# import os
# from PIL import Image
# import io
# import base64
#
# df = pd.read_parquet('./data/rick_and_morty_image_and_text/data/train-00000-of-00001-1f95078297b4c0ef.parquet')
import io
import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ==================== 配置 ====================


num = '1'
na = "龙珠"
DATASET_NAME = "../data/DragonBall-SS-Captions/data/"  # 数据集名称

"""
naruto-blip-captions
doodles-captions-BLIP
onepiece-blip-captions
rick_and_morty_image_and_text
Dataset({
    features: ['image', 'text'],
    num_rows: 856
})
Dataset({
    features: ['caption', 'image'],
    num_rows: 146
})
"""

OUTPUT_DIR = f"./{na}/sty_{num}"
name = f"./{na}/tar_style_texts_{num}"
imag_save = f"./{na}/cnt_{num}"
file_name = [ i.split('_')[-1].split('.')[0] for i in os.listdir(name)]

# ==================== 创建输出文件夹 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 使用 datasets 库加载 ====================
print("正在加载数据集...")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"总数据集大小: {len(dataset)} 个样本")
print(f"提取的数据集大小: {len(file_name)} 个样本")
print(f"数据集特征: {dataset.features}")


# print(dataset)
# print(qwe)

# ==================== 提取并保存图像 ====================
for i, j in enumerate(file_name, 1):
    idx = int(j)
    image = dataset["image"][idx]

    # print(image)

    # 保存图像
    image_path = os.path.join(OUTPUT_DIR, f"image{i}.png")
    # image_path = os.path.join(OUTPUT_DIR, f"image_{idx:06d}.png")

    image.save(image_path)
    # img = Image.open(io.BytesIO(image['bytes']))
    # img.save(image_path)


print(f"✅ 图像已保存到: {OUTPUT_DIR}")