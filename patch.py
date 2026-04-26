import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from torchaudio.prototype.models import hifigan_vocoder

from . import merge
from .utils import isinstance_str, init_generator



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

        ## x.shape[1] 是 token 数量 4096  ratio 是合并比例 0.5
        # Re-init the generator if it hasn't already been initialized or device has changed.
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






def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        ## 这个前向传播仅在图片生成时调用
        ## 这也是整个算法最核心开始的地方
        def forward(
            self,
            hidden_states,                          ## [2, 16384, 320]
            attention_mask=None,
            encoder_hidden_states=None,             ## [2, 77, 768]
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            ### 这里就是BasicTransformerBlock模块的forward函数开始
            ### 就是模型train中调用的 forward 函数
            ### 这里是hook插入的地方是模型的transformer_blocks中的forward函数
            ### 总共有13层计算模块，比如线性层、卷积层、归一化层（目前这个模型用好像没有卷积层，池化层是不算来参数量的）
            # weight_layers = [
            #     (name, layer) for name, layer in self.named_modules()
            #     if hasattr(layer, "weight")
            # ]
            # print(len(weight_layers))
            # print('!!!',hidden_states.shape)
            # print(qwe)
            ## 此刻的hidden_states的形状是 torch.Size([2, 4096, 320]),
            ## 这个形状的计算规则是 batch_size=2，token数量=512/8 * 512/8 = 4096，hidden_size=320
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)    ## 这里是算出随机取舍的位置，4个选其一，以及合并和解合并函数,注意这里只是生成了模板并未真正执行合并和解合并操作，真正的合并和解合并操作是在后续的代码中调用 m_a, m_c, m_m, u_a, u_c, u_m 这6个函数时才会执行的

            ## 这里norm1
            if self.use_ada_layer_norm:     ## False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:      ## False
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)      ## [2, 4096, 320]

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)        ## 合并之后[2, 2048, 320]

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            '''
            合并之后的norm_hidden_states[2, 2份， 320]输入attn1模块里
            attn1模块里是简单的kqv计算，都是线性层
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=320, out_features=320, bias=False)
            (to_v): Linear(in_features=320, out_features=320, bias=False)
            这就是DIffusion中Transformer加速的重要原因了，因为少了将近一半的Token计算量
            '''
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )   ## [2, 2份, 320], [2, 2048, 320]
            if self.use_ada_layer_norm_zero:    ## False
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            ## 解合并，将原先的token数量恢复回来  [2, 4份, 320]，[2, 4096, 320]，注意此hidden_states是原始输入的未做任何处理的
            hidden_states = u_a(attn_output) + hidden_states        ## 这个结构有点像ResNet的结构


            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                ## 送入attention里前的一次合并，但是这里的交叉注意力是False，所以并不执行合并，什么也不会改变
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                ## 输出Cross-Attention后的一次解合并，但是这里的交叉注意力是False，所以并不执行解合并，什么也不会改变
                hidden_states = u_c(attn_output) + hidden_states


            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:    ## False
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)        ## MLP层不做合并

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:    ## False
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states
            # print("norm_hidden_states", norm_hidden_states.shape)
            # print(qwe)

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
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.

    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    # 确认模型没有被 ToMe 补丁修改过
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

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        ## 过滤整个模型，只筛选BasicTransformerBlock模块进来修改
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            ## 重点  这里是把 module 的类改成 ToMeBlock 类，从而实现对 forward 方法的重写，对Diffusion的BaseTransformerBlock进行改写
            ## 原本是<class 'diffusers.models.attention.BasicTransformerBlock'>
            ## 经过make_tome_block_fn函数之后变成了<class 'tomesd.patch.make_diffusers_tome_block.<locals>.ToMeBlock'>
            module.__class__ = make_tome_block_fn(module.__class__) ## <class 'diffusers.models.attention.BasicTransformerBlock'>
            module._tome_info = diffusion_model._tome_info

            # Something introduced in SD 2.0 (LDM only)
            ## 这里是给 module 添加 disable_self_attn 属性，默认值为 False
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            ## 这里是给 module 添加 use_ada_layer_norm 和 use_ada_layer_norm_zero 属性，默认值为 False
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    '''
    总结： 这段代码的主要功能是将 ToMe（Token Merging）补丁应用到一个稳定扩散模型（Stable Diffusion Model）中。
    通过修改模型的基本变换块（BasicTransformerBlock）的类定义，使其在前向传播过程中能够执行
    令牌合并操作，从而提高模型的计算效率。补丁应用后，模型在处理输入时会根据设定的参数动态调整令牌数量，
    反向传播时的功能在不在这段函数里体现，而是在函数？？？make_diffusers_tome_block？？？（猜测）
    -----------------外部的 ToMeBlock 类的 forward 方法中实现。
    '''
    # print("*-*-"*30)
    # print(qwe)

    return model




## 这段代码是用于移除 ToMe 补丁的函数
def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    # 判断 model 是否为 diffusers 模型
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():

        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    return model
