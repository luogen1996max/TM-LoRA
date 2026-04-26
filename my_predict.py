from diffusers import StableDiffusionPipeline
import torch

from pytorch.six_Net import generator

# model_id = "./sd-naruto-model"
model_id = "./stable-diffusion-v1-5-normal"
# model_id = "./checkpoint-5000"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
## 如果模型是LoRA微调的，加载LoRA权重
pipe.load_lora_weights('./checkpoint-5000/')
pipe = pipe.to("cuda")

## 这是跳过安全检查的代码
pipe.safety_checker = None

# 生成图片时显式指定seed
generator = torch.Generator("cuda").manual_seed(42)  # ⚠️ 必须用CUDA Generator
# print(generator)
# print(qwe)

# prompt = "Lebron James with a hat"
# prompt = "a man with glasses holds wired device black and wearing black framed glasses, black and brown spliced coat"
prompts = ["Bill Gates is a man", "John Oliver with sunshine style", "Lebron James with a hat", "a sunshine man"]
# prompt = "Lebron James with a hat is a basketball player who plays for the Los Angeles Lakers in the NBA. He is known for his incredible athleticism, basketball IQ, and versatility on the court. James has won multiple NBA championships and is often considered one of the greatest basketball players of all time."
# prompt = "sexy girl in a bikini, pron, 1girl, solo, looking at viewer, 18yo, swimsuit, bikini, beach, summer, sunbathing, cute face, beautiful eyes, perfect body, sexy pose"
for i, prompt in enumerate(prompts, 1):
    image = pipe(prompt, 512, 512, num_inference_steps=50, generator=generator).images[0]
    # image = pipe(prompt, 256, 256, num_inference_steps=999).images[0]
    # image = pipe(prompt, 256, 256, generator=generator).images[0]
    image.save(f"result{i}.png")
