import torch
# import tomesd

import tomesd.patch as tomesd
from diffusers import StableDiffusionPipeline
import torch


# def print_gpu_usage():
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         # 当前 GPU 显存占用（MB）
#         allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
#         # 当前 GPU 显存缓存（MB）
#         cached = torch.cuda.memory_reserved(device) / (1024 ** 2)
#         # GPU 总显存（MB）
#         total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
#         print(f"GPU 显存占用: {allocated:.2f} MB / {total:.2f} MB")
#         print(f"GPU 显存缓存: {cached:.2f} MB")
#     else:
#         print("CUDA 不可用，未检测到 GPU")


# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5-me", torch_dtype=torch.float16).to("cuda")

## 这是跳过安全检查的代码
pipe.safety_checker = None

# Apply ToMe with a 50% merging ratio
tomesd.apply_patch(pipe, ratio=0.5) # Can also use pipe.unet in place of pipe here
# tomesd.apply_patch(pipe, ratio=0.5, merge_crossattn=True, merge_mlp=True) # Can also use pipe.unet in place of pipe here

para_total = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
s = f"{para_total}" if para_total < 100_000_000 else "_".join(f"{para_total:,}".replace(",","")[i:i+4] for i in range(0,len(str(para_total)),4))
print('='*10, f'参数量为：{s}', '='*10)

# # 调用函数
# print("开始生成图片...")
# print_gpu_usage()

import time
start = time.time()

prompt="a photo of an astronaut riding a horse on mars"

# 生成图片时显式指定seed
generator = torch.Generator("cuda").manual_seed(42)  # ⚠️ 必须用CUDA Generator

# image = pipe(prompt, 1024, 1024, num_inference_steps=999, generator=generator).images[0]
# image = pipe(prompt, 1024, 1024, generator=generator).images[0]
image = pipe(prompt, 512, 512, generator=generator).images[0]
# image = pipe(prompt, 256, 256, generator=generator).images[0]
print("Tome完成！")
# print(qwe)


# print("图片生成完成！")
# print_gpu_usage()

image.save("astronaut.png")
print('*=*=*=   ' + "\033[31m用时为:%f S \033[0m" % int(time.time() - start))