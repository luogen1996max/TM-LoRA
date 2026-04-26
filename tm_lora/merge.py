import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape  ## [2, 4096, 320]

    if r <= 0:
        return do_nothing, do_nothing

    ## 这里是为了兼容MPS设备，因为MPS设备在某些情况下可能不支持torch.gather，所以提供了一个替代函数mps_gather_workaround
    ## torch.gather函数是根据输入的索引从输入张量中收集元素的函数
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx     ## 64//2=32, 64//2=32

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        ## 创建一个随机索引，这个索引将为后续的scatter操作提供位置
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            ## 创建一个模版索引【32, 32, 1】索引内容为0-3对应2*2的模板里的位置
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        ## idx_buffer_view的形状为[32, 32, 4] 都是零, 这些零一共4096个
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        ## idx_buffer_view的形状为[32, 32, 4], 里面的值为0或-1，且-1是根据rand_idx随机分布的
        ## 32行32列，每个行有4个值，这4个值中有3个是0，1个是-1，且-1的位置是根据rand_idx随机分布的
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))   ## 合并索引，[64,64,,4]值为0/-1,且-1是根据rand_idx随机分布的
        ## 此刻的idx_buffer_view的形状为torch.Size([64, 64])，
        ## 此刻里面的值为0或-1，且-1是根据rand_idx随机分布的，每个2*2里有3个0和1个-1
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)    ## [128,128]生成的随机模版，里面是值是每2*2里有3个0和1个-1


        # Image is not divisible by sx or sy so we need to move it into a new buffer
        ## 图片是无法被sx或sy整除的，所以我们需要将其移动到一个新的缓冲区中
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        '''
        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        ## 又将此拉伸成一维[1, 64*64, 1],然后排序，得到dst|src的索引顺序,该索引是按列从小到大的，得到的ran_idx实际上这些排序后值的原始索引位置
        ## 比如idx_buffer拉伸成一维之后是[0, 0, -1, 0,
        ##                            -1, 0, 0, 0,
        ##                            0, -1, -1, 0,
        ##                            0, 0, 0, 0]
        ## 经过argsort排序后得到的是，[ 2,  4,  9, 10,  0,  1,  3,  5,  6,  7,  8, 11, 12, 13, 14, 15]
        ## 其中前4个索引位置是-1的位置，也就是dst的位置，后面8个索引位置是0的位置，也就是src的位置
        ## 所以得到的rand_idx总数为4096，前1024个是dst -1 的索引位置，后3072个是src 0 的索引位置
        '''
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)


        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx     ## 32*32=1024
        a_idx = rand_idx[:, num_dst:, :] # src      ## 这个其实就是0的索引位置     [1, 3072, 1]
        b_idx = rand_idx[:, :num_dst, :] # dst      ## 这个其实就是-1的索引位置    [1, 1024, 1]

        def split(x):
            C = x.shape[-1]     ## 320
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))       ## [2, 3072, 320]
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))           ## [2, 1024, 320]
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        ## 拆分src和dst
        a, b = split(metric)
        ## 计算相似度分数
        ## a的形状为[2, 3072, 320]，b的形状为[2, 1024, 320]，
        ## b.transpose(-1, -2)的形状为torch.Size([2, 320, 1024])
        ## 所以a @ b.transpose(-1, -2)的形状为[2, 3072, 1024]，每个元素表示src中的一个token与dst中的一个token的相似度分数
        scores = a @ b.transpose(-1, -2)    ## [2, 12288, 4096]

        r = min(a.shape[1], r)      ## 2048

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]


        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens   ## 相似度靠后面的值     ## 2, 1024, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens     ## 相似度靠前面的值     ## 2, 2048, 1
        ## 根据src_idx从node_idx中获取对应的dst_idx，也就是从12288中获取最大值到8192个对应的索引位置，也就是相似度最高的8192个dst位置
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)    ## 2, 2048, 1


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)         ## 划分4份，3份src，1份dst
        n, t1, c = src.shape        ## n=2, t1=3072, c=320

        print('src',src.shape)
        print('dst', dst.shape)

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))   ## 4格里最不像的一份    unm=[2, 1024, 320]
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))        ## 4格里最像的两份(关于这两份怎么来的，其实是总4份，然后乘以要合并的系数得到的)     src=[2, 2048, 320]
        ## 以dst为模板，然后根据相似度最高的dst_idx索引将src合并到dst上，这个合并操作是根据mode来决定的，这里是mean
        dst = dst.scatter_reduce(-2, index=dst_idx.expand(n, r, c), src=src, reduce=mode)   ## 将src合并到dst上, dst=[2, 1024, 320]

        return torch.cat([unm, dst], dim=1)         ## 2份torch.Size([2, 2048, 320])

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        ## x为[2, 2份， 320]
        unm_len = unm_idx.shape[1]      ## 1份
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]     ## unm=1份， dst=1份，将merge的结果拆分开来
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))    ## [2, 2份, 320]，根据dst_idx索引从dst中获取对应的src

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out      ## [2, 16384, 320], [2, 4份， 320]

    return merge, unmerge
