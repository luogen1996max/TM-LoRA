import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from torchaudio.prototype.models import hifigan_vocoder

from . import merge
from .utils import isinstance_str, init_generator

import torch.nn.functional as F

def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    # print(x.shape)   ## torch.Size([2, 4096, 320])
    original_h, original_w = tome_info["size"]  ## 64, 64
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))       ## 1 math.ceil向上取整

    args = tome_info["args"]    ## 里面存储了各种参数

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))     ## 64
        h = int(math.ceil(original_h / downsample))     ## 64
        r = int(x.shape[1] * args["ratio"])             ## 2048

        if args["generator"] is None:
            ## 这里是根据输入 x 的设备类型初始化一个随机数生成器
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        # 如果 batch size 是奇数，就强制关闭 use_rand， 以避免提示和非提示图像在同一批次中出现伪影
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]   ## ture
        ## 调用 merge 模块中的 bipartite_soft_matching_random2d 函数，计算合并和展开函数
        ## x[2, 4096, 320] w=64 h=64 sx=2 sy=2 r=2048 no_rand=False generator=args["generator"]
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r,
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)
    ## 这里的a，c，m分别代表是attention、cross attention、MLP

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good





def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x

    return ToMeBlock

import torch.nn as nn
def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 在这里添加卷积层
            self.k_c = nn.Sequential(
                nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(4)  # 注意：应该是 BatchNorm2d，不是 BatchNorm3d
            )

        ## 这个前向传播仅在图片生成时调用
        ## 这也是整个算法最核心开始的地方
        def forward(
            self,
            hidden_states,                          ## [1, 256, 320]
            attention_mask=None,
            encoder_hidden_states=None,             ## [2, 77, 768]
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_q, m_k, m_v, u_q, u_k, u_v = compute_merge(hidden_states, self._tome_info)    ## 这里是算出随机取舍的位置，4个选其一，以及合并和解合并函数,注意这里只是生成了模板并未真正执行合并和解合并操作，真正的合并和解合并操作是在后续的代码中调用 m_a, m_c, m_m, u_a, u_c, u_m 这6个函数时才会执行的
            # weight_layers = [
            #     (name, layer) for name, layer in self.named_modules()
            #     if hasattr(layer, "weight")
            # ]
            # print(len(weight_layers))
            # print(weight_layers)
            # print('!!!',hidden_states.shape)

            residual = hidden_states
            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            hidden_states = m_q(hidden_states)
            query = self.to_q(hidden_states)
            # query = m_q(query)

            # # 原始基础层的输出（冻结的原始权重）
            # base_output = self.to_q.base_layer(hidden_states)
            # dropout_output = self.to_q.lora_dropout(base_output)
            # # LoRA A 矩阵（下投影）
            # lora_a_output = self.to_q.lora_A(dropout_output)
            # lora_ab_output = self.k_c(lora_a_output)
            # # LoRA B 矩阵（上投影）
            # lora_b_output = self.to_q.lora_B(lora_ab_output)
            # # LoRA Embedding A（可选，用于 embedding 层）
            # lora_embedding_a_output = self.to_q.lora_embedding_A(hidden_states)
            # # LoRA Embedding B（可选）
            # lora_embedding_b_output = self.to_q.lora_embedding_B(lora_embedding_a_output)
            # # 最终输出 = 原始输出 + LoRA 贡献
            # query = base_output + (self.to_q.scaling * lora_b_output) + lora_embedding_b_output


            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            # key = m_k(key)
            value = self.to_v(encoder_hidden_states)
            # value = m_v(value)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = u_q(hidden_states)
            hidden_states = hidden_states / self.rescale_output_factor

            # hidden_states_q = u_q(hidden_states)
            # hidden_states_k = u_k(hidden_states)
            # hidden_states_v = u_v(hidden_states)
            # hidden_states = hidden_states_q + hidden_states_k + hidden_states_v

            return hidden_states

    return ToMeBlock

def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    ## 这里是弄了个钩子函数，获取输入的图像尺寸
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])    ## 2, 4, 128, 128
        return None
    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))

def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = True,
        merge_mlp: bool = True):

    remove_patch(model)

    # 这是在判断 model 是不是 diffusers 的模型
    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:    ## 若不是 diffusers 模型
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,                         ## 合并比例 0.5
            "max_downsample": max_downsample,       ##  最大下采样 1
            "sx": sx, "sy": sy,
            "use_rand": use_rand,                   ##  使用随机 True
            "generator": None,
            "merge_attn": merge_attn,               ##  合并注意力 True
            "merge_crossattn": merge_crossattn,     ##  合并交叉注意力 False
            "merge_mlp": merge_mlp                  ##  合并MLP False
        }
    }

    ## 给 diffusion_model 添加钩子函数, 这个钩子的功能是获取输入图像尺寸，并存储在 diffusion_model._tome_info["size"] 中
    hook_tome_model(diffusion_model)

    for name, submodule in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        ## 过滤整个模型，只筛选BasicTransformerBlock模块进来修改

        if isinstance_str(submodule, "BasicTransformerBlock"):
            for _, module in submodule.named_modules():
                if isinstance_str(module, "Attention"):

                    make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
                    module.__class__ = make_tome_block_fn(module.__class__) ## <class 'diffusers.models.attention.BasicTransformerBlock'>
                    module._tome_info = diffusion_model._tome_info

                    # Something introduced in SD 2.0 (LDM only)
                    ## 这里是给 module 添加 disable_self_attn 属性，默认值为 False
                    if not hasattr(module, "disable_self_attn") and not is_diffusers:
                        module.disable_self_attn = False

                    if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                        module.use_ada_layer_norm = False
                        module.use_ada_layer_norm_zero = False

    return model




## 这段代码是用于移除 ToMe 补丁的函数
def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():

        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    return model
