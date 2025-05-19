import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
from piq import DISTS

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x.to(device)  # 直接将数据送到GPU
])

transform_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 转换到[-1, 1]
    lambda x: x.to(device)
])


def CalcuPSNR_torch(img1, img2, max_val=255.0):
    # 确保输入在GPU上且为浮点类型
    img1 = torch.clamp(img1, 0, 1).to(device)
    img2 = torch.clamp(img2, 0, 1).to(device)

    # 转换为0-255范围
    img1 = img1 * max_val
    img2 = img2 * max_val

    # 计算MSE和PSNR
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr


def calculate_ssim(img1, img2):
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
    return ssim_module(img1, img2).unsqueeze(-1)


def calculate_lpips(imglip_1, imglip_2, net_type='alex'):
    loss_fn = lpips.LPIPS(net=net_type, verbose=False).to(device)
    loss_fn.eval()
    with torch.no_grad():
        distance = loss_fn(imglip_1, imglip_2)
    return distance.view(-1)


def calculate_dists(img1_tensor, img2_tensor):
    model = DISTS().to(device)
    model.eval()
    with torch.no_grad():
        score = model(img1_tensor, img2_tensor)
    return score.view(-1)


def load_images_to_tensor(img_paths, transform, batch_indices):
    """加载指定批次的图片到Tensor"""
    batch_tensors = []
    for idx in batch_indices:
        with Image.open(img_paths[idx]).convert('RGB') as img:
            batch_tensors.append(transform(img))
    return torch.stack(batch_tensors).to(device)


def eval_calculate(coeff_dir, original_dir, avg_eval_dir, batch_size, idx, sum_nums):
    coeff_paths = [os.path.join(coeff_dir, f"frame_{i}.png") for i in range(sum_nums)]
    original_paths = [os.path.join(original_dir, f"{i:06d}.png") for i in range(sum_nums)]

    # 初始化统计结果
    all_psnr, all_ssim, all_lpips, all_dists = [], [], [], []

    # 分批次处理
    for batch_start in range(0, sum_nums, batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, sum_nums))

        # 加载当前批次
        img_coeff = load_images_to_tensor(coeff_paths, transform, batch_indices)
        img_original = load_images_to_tensor(original_paths, transform, batch_indices)
        img_coeff_2 = load_images_to_tensor(coeff_paths, transform_2, batch_indices)
        img_original_2 = load_images_to_tensor(original_paths, transform_2, batch_indices)

        # 计算指标
        psnr_batch = CalcuPSNR_torch(img_coeff, img_original)
        ssim_batch = calculate_ssim(img_coeff, img_original)
        lpips_batch = calculate_lpips(img_coeff_2, img_original_2)
        dists_batch = calculate_dists(img_coeff_2, img_original_2)

        # 汇总结果
        all_psnr.extend(psnr_batch.cpu().numpy())
        all_ssim.extend(ssim_batch.cpu().numpy())
        all_lpips.extend(lpips_batch.cpu().numpy())
        all_dists.extend(dists_batch.cpu().numpy())

        print(f"Processed batch {batch_start // batch_size + 1}/{(sum_nums + batch_size - 1) // batch_size}")

    # 计算全局平均值
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_lpips = np.mean(all_lpips)
    avg_dists = np.mean(all_dists)

    print("\nFinal Averages:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    print(f"DISTS: {avg_dists:.4f}")

    result_line = f"{'H265' if idx == 0 else 'NTC'}平均PSNR为: {avg_psnr:.4f}; 平均SSIM为: {avg_ssim:.4f}; 平均LPIPS为: {avg_lpips:.4f}; 平均DISTS为: {avg_dists:.4f}"
    with open(avg_eval_dir, 'a', encoding='utf-8') as f:
        f.write(result_line + '\n')
    print(result_line)


def eval_data(sum_poc, BATCH_SIZE, NTC_dir, H265_dir, original_dir, out_eval, index_H265, index_NTC):

    os.makedirs(out_eval, exist_ok=True)

    avg_PSNR_dir = os.path.join(out_eval, "avg_eval.txt")

    if index_H265 == 1:
        eval_calculate(H265_dir, original_dir, avg_PSNR_dir, BATCH_SIZE, 0, sum_poc)

    if index_NTC == 1:
        eval_calculate(NTC_dir, original_dir, avg_PSNR_dir, BATCH_SIZE, 1, sum_poc)


if __name__ == '__main__':

    # 参数配置
    final_nums = 10           # 总共测试帧
    BATCH_SIZE = 10            # 每批评价计算的样本数
    original_dir = r'D:\original_data'     # 原始图像地址
    NTC_dir = r'D:\Projects\DL_zuoye\JSCC\POC\restore\imagedata\H265-last1000'         # 重建图像目录
    H265_dir = '/media/D/yangxuzhi/test3/cmake-build-debug/decode_uvg/NTC/wang-test-10.5'   # H265重建图像目录
    out_eval = "eval/y-uv-snr10.9-yufa"                     # 评价指标地址
    index_H265 = 0                                          # 判断H265PSNR是否输出
    index_NTC = 1                                           # 判断NTCPSNR是否输出


    eval_data(final_nums, BATCH_SIZE,NTC_dir, H265_dir, original_dir, out_eval, index_H265, index_NTC)
