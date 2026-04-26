from skimage.metrics import structural_similarity as ssim
from skimage import io, transform


def myssim(gen_img, real_img):
    # 读取两幅图像
    image1 = io.imread(gen_img, as_gray=True)
    image2 = io.imread(real_img, as_gray=True)
    # 将image2调整为512x512
    # image2 = transform.resize(image2, (512, 512), preserve_range=True).astype(image1.dtype)
    # 计算SSIM
    ssim_score = ssim(image1, image2, data_range=1.0)
    return ssim_score

if __name__ == '__main__':

    gen_dir = '../hugg_face_extra_data/dood1/our_1/checkpoint-10000'
    # real_dir = '../hugg_face_extra_data/dood1/sty_1'
    real_dir = '../hugg_face_extra_data/dood1/cnt_1'

    eval_ssim_score = 0
    for i in range(1, 21):
        gen_img = gen_dir + '/' + f'pre_result{i}.png'
        real_img = real_dir + '/' + f'result{i}.png'
        # real_img = real_dir + '/' + f'image{i}.png'
        ssim_score= myssim(gen_img, real_img)
        eval_ssim_score = eval_ssim_score + ssim_score
    print(f"SSIM: {ssim_score}")

print("平均：" , eval_ssim_score / 20)