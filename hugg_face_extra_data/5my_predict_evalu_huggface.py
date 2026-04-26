from diffusers import StableDiffusionPipeline
import torch
import os
import gc
import tqdm


num = '1'
na = "龙珠"
# na_na = '113944'
# na_na = '解合并后归一'
# our_or_normal = "our_"
#
na_na = '正常1'
our_or_normal = "normal_"

## 权重文件位置
path_name = f'../train/{na}/parameter/{na_na}'
## 生成文件存放位置
tar_file = f'../hugg_face_extra_data/{na}/{our_or_normal}{num}'

name = f"../hugg_face_extra_data/{na}/tar_style_texts_{num}"
imag_save = f"../hugg_face_extra_data/{na}/cnt_{num}"
file_name = [ f'{name}/' + i for i in os.listdir(name)]
# file_name = [ f'{name}/' + i for i in  random.sample(os.listdir(f'{os.path.split(name)}/tar_style_texts'), 3)]
file_name = sorted(file_name)

# file_name = ['a woman with a pink shirt and a brown hair', 'a cartoon character with a blue hair and a white shirt']pr
prompts = []
for i in file_name:
    with open(i, "r+") as f:
        prompts.append(f.readline())
# print(qwe)

path_file = [ 'checkpoint-' + str(i)  for i in range(1000, 20001, 1000) ]
progress_bar = tqdm.tqdm(
        range(0, len(path_file)),
        initial=0,
        desc="Steps",
    )
model_id = "../stable-diffusion-v1-5-normal"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
## 如果模型是LoRA微调的，加载LoRA权重
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

    ## 这是跳过安全检查的代码
    pipe.safety_checker = None

    # 生成图片时显式指定seed
    generator = torch.Generator("cuda").manual_seed(42)  # ⚠️ 必须用CUDA Generator

    # prompts = file_name
    for i, prompt in enumerate(prompts, 1):
        image = pipe(prompt, 512, 512, num_inference_steps=50, generator=generator).images[0]
        image.save(f"{tar_file_pic}/pre_result{i}.png")
        logs = {"#####  ": i, " || 20  ##### ": prompt}
        progress_bar.set_postfix(**logs)
    progress_bar.update(1)
        # print(qwe)
    # 3. ⚠️ 关键：卸载 LoRA 权重，释放显存
    pipe.unload_lora_weights()

    # 4. 清理 CUDA 缓存
    torch.cuda.empty_cache()
    gc.collect()


contain_txt = f"推理结束： {path_name}"
from send_email import send_email
send_email(contain_txt)