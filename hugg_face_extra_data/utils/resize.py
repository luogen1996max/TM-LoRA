import shutil

from PIL import Image
import glob, os
import re

# def batch_resize(src_folder, dst_folder, size=(512,512)):
def batch_resize(fn, size=(240,150)):
    img = Image.open(fn).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    img.save('./resize.png')

src = './line_chart_smooth.png'


batch_resize(src)
