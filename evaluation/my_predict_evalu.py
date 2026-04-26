from diffusers import StableDiffusionPipeline
import torch
import os
import gc

import tqdm

from pytorch.six_Net import generator
#
model_id = "../stable-diffusion-v1-5-normal"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
## 如果模型是LoRA微调的，加载LoRA权重

## 权重文件位置
path_name = '../train/海贼/parameter/解合并后归一2'
## 生成文件存放位置
tar_file = '../evalu/our'


# path_file = os.listdir(path_name)
# path_file = sorted(path_file)
path_file = [ 'checkpoint-' + str(i)  for i in range(1000, 20001, 1000) ]

progress_bar = tqdm.tqdm(
        range(0, len(path_file)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
    )

for idx, i in enumerate(path_file):
    if i == 'pytorch_lora_weights.safetensors':
        continue
    parameter = path_name + '/' + i

    tar_file_pic = tar_file + '/' + i
    if not os.path.exists(tar_file_pic):
        os.makedirs(tar_file_pic, exist_ok=True)

    # pipe.load_lora_weights('../checkpoint-20000')
    pipe.load_lora_weights(parameter)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)


    # print(f"\n[{idx + 1}/{len(path_file)}] 加载 LoRA: {i}")

    ## 这是跳过安全检查的代码
    pipe.safety_checker = None

    # 生成图片时显式指定seed
    generator = torch.Generator("cuda").manual_seed(42)  # ⚠️ 必须用CUDA Generator

    prompts = ["Bill Gates is a man", "John Oliver with sunshine style", "Lebron James with a hat", "a sunshine man"]
    for i, prompt in enumerate(prompts, 1):
        image = pipe(prompt, 512, 512, num_inference_steps=50, generator=generator).images[0]
        image.save(f"{tar_file_pic}/pre_result{i}.png")
    progress_bar.update(1)
    logs = {"#####  ": idx + 1}
    progress_bar.set_postfix(**logs)
        # print(qwe)
    # 3. ⚠️ 关键：卸载 LoRA 权重，释放显存
    pipe.unload_lora_weights()

    # 4. 清理 CUDA 缓存
    torch.cuda.empty_cache()
    gc.collect()
