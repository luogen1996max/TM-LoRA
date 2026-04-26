import shutil

from PIL import Image
import glob, os
import re

# def batch_resize(src_folder, dst_folder, size=(512,512)):
def batch_resize(src_folder, dst_folder, size=(55,55)):
    os.makedirs(dst_folder, exist_ok=True)
    for fn in glob.glob(f"{src_folder}/*.*"):
        img = Image.open(fn).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        basename = os.path.basename(fn)

        spl = fn.split('/')[1].split('\\')[0][:-2]

        if spl == "cnt":
            qufen = '1'
        elif spl == "our":
            qufen = '2'
        elif spl == "normal":
            qufen = '3'
        else:
            print(spl)
            print("出问题了")
            break

        imag_num = re.findall(r'\d+', basename)[0]
        imag_name = f'img_{imag_num}_{qufen}.png'
        # print(basename)
        # print(qwe)
        img.save(os.path.join(dst_folder, imag_name))

num = 1
# src = f"./our_{num}/checkpoint-20000"
san = ['cnt', 'our',  'normal']
dst = "./512"

os.makedirs(dst, exist_ok=True)
for i in san:
    if i == 'cnt':
        src = f"./{i}_{num}"
        path = './新建文件夹'
        os.makedirs(path, exist_ok=True)
        for j in [i for i in os.listdir(src) if '.txt' in i]:
            shutil.move(src+'/'+j, './新建文件夹')
    else:
        src = f"./{i}_{num}/checkpoint-20000"
# batch_resize("./data/imagenet_val_1000flat","./data/imagenet_val_1000flat_resized")
    batch_resize(src, dst)
