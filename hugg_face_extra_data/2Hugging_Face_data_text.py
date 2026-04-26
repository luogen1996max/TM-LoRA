# import pandas as pd
# import os
# from PIL import Image
# import io
# import base64
#

import os


import pandas as pd
from tqdm import tqdm
import json

# ==================== 配置 ====================
PARQUET_PATH = "../data/DragonBall-SS-Captions/data/train-00000-of-00001.parquet"
OUTPUT_FILE = "extracted_texts.json"
TXT_DIR = "./龙珠/tar_style_texts"

caption_column = 'caption'
# caption_column = 'text'

# ==================== 创建输出文件夹 ====================
os.makedirs(TXT_DIR, exist_ok=True)

# ==================== 读取 parquet 文件 ====================
df = pd.read_parquet(PARQUET_PATH)
print(f"列名: {df.columns.tolist()}")

# ==================== 提取文本 ====================
texts = []
for idx, row in df.iterrows():
    text = row.get(f'{caption_column}', '')
    if text:
        texts.append({
            "id": idx,
            f"{caption_column}": text
        })

        # 可选：单独保存每个文本
        with open(os.path.join(TXT_DIR, f"text_{idx:06d}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)

# # 保存为 JSON 文件
# with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#     json.dump(texts, f, ensure_ascii=False, indent=2)

print(f"✅ 共提取 {len(texts)} 条文本")
print(f"✅ 文本已保存到: {OUTPUT_FILE}")
print(f"✅ 单条文本保存到: {TXT_DIR}")

# 打印前几条文本示例
print("\n文本示例:")
for i in range(min(3, len(texts))):
    print(f"  [{i}] {texts[i][f'{caption_column}'][:100]}...")