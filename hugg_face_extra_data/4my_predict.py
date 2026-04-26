from diffusers import StableDiffusionPipeline
import torch
import os
import tqdm
import random
import shutil


# prompts = ["Bill Gates is a man", "John Oliver with sunshine style", "Lebron James with a hat", "a sunshine man"]
num = '1'
na = "龙珠"


name = f"./{na}/tar_style_texts_{num}"
imag_save = f"./{na}/cnt_{num}"
file_name = [ f'{name}/' + i for i in os.listdir(name)]
# file_name = [ f'{name}/' + i for i in  random.sample(os.listdir(f'{os.path.split(name)}/tar_style_texts'), 3)]
file_name = sorted(file_name)


model_id = "../stable-diffusion-v1-5-normal"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
## 如果模型是LoRA微调的，加载LoRA权重
# pipe.load_lora_weights('../checkpoint-20000/')
pipe = pipe.to("cuda")
## 这是跳过安全检查的代码
pipe.safety_checker = None
# 生成图片时显式指定seed
generator = torch.Generator("cuda").manual_seed(42)  # ⚠️ 必须用CUDA Generator

prompts = []
for i in file_name:
    with open(i, "r+") as f:
        prompts.append(f.readline())

os.makedirs(imag_save, exist_ok=True)

progress_bar = tqdm.tqdm(
        range(0, len(prompts)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
    )

pipe.set_progress_bar_config(disable=True)
for i, prompt in enumerate(prompts, 1):
    image = pipe(prompt, 512, 512, num_inference_steps=50, generator=generator).images[0]
    image.save(f"{imag_save}/result{i}.png")

    # print(prompt)
    shutil.copy(file_name[i-1], f"{imag_save}/{file_name[i-1].split('/')[-1]}")

    progress_bar.update(1)
    logs = {"#####  ": i + 1, "#####  ": prompt}
    progress_bar.set_postfix(**logs)
