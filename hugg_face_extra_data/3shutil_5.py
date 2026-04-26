import os
import random
import shutil

# prompts = ["Bill Gates is a man", "John Oliver with sunshine style", "Lebron James with a hat", "a sunshine man"]
# file_name = [ './tar_style_texts/' + i for i in os.listdir('./tar_style_texts')]
name = "./龙珠/tar_style_texts"
os.makedirs(name, exist_ok=True)
for i in range(1,6):
    file_name = [f'{name}/' + i for i in random.sample(os.listdir(name), 20)]
    file_name = sorted(file_name)
    for src in (file_name):
        os.makedirs(f'{name}_{i}', exist_ok=True)
        shutil.copy(src, f"{name}_{i}/{src.split('/')[-1]}")

print(f"✅ 共划分 5 文件夹")
print(f"✅ 每文件 20 个样本")

