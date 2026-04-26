
# 首先安装: pip install pytorch-fid
from pytorch_fid import fid_score

gen_dir = '../hugg_face_extra_data/dood1/our_1/checkpoint-1000'
real_dir = '../hugg_face_extra_data/dood1/sty_1'


eval_fid_value = 0
# for i in range(1, 21):
#     gen_img = gen_dir + '/' + f'pre_result{i}.png'
#     real_img = real_dir + '/' + f'image{i}.png'

# 计算 FID
fid_value = fid_score.calculate_fid_given_paths(
    [gen_dir, real_dir],
    batch_size=1,
    device='cuda',
    dims=2048,
    num_workers=0
)
eval_fid_value = eval_fid_value + fid_value
print(f"FID: {fid_value:.4f}")

print("平均FID：", eval_fid_value/20)