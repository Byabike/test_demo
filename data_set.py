import os
import tarfile
from tqdm import tqdm
import cv2
import re
import numpy as np


def is_color_image(img):
    """通过YCbCr颜色空间检测彩色图像"""
    UV_THRESHOLD = 1  # 色度通道波动阈值
    try:
        # 转换到YUV颜色空间
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)

        # 获取色度通道极值
        cb_min, cb_max = u.min(), u.max()
        cr_min, cr_max = v.min(), v.max()

        return not (abs(cb_min - 128) <= UV_THRESHOLD and
                    abs(cb_max - 128) <= UV_THRESHOLD and
                    abs(cr_min - 128) <= UV_THRESHOLD and
                    abs(cr_max - 128) <= UV_THRESHOLD)
    except Exception as e:
        return False


def process_image(img_path, output_path, TARGET_SIZE):
    """处理并保存彩色图像"""
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if not is_color_image(img):
            return False

        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        cv2.imencode('.png', img)[1].tofile(output_path)
        return True
    except Exception as e:
        print(f"处理失败 {img_path}: {str(e)}")
        return False


def extract_tar_if_needed(TAR_PATH, TEMP_DIR):
    """条件解压：仅在目录不存在时解压"""
    if os.path.exists(TEMP_DIR):
        print(f"检测到现有解压目录 {TEMP_DIR}，跳过解压步骤")
        return

    print("开始解压tar文件...")
    os.makedirs(TEMP_DIR, exist_ok=True)
    with tarfile.open(TAR_PATH, "r") as tar:
        tar.extractall(path=TEMP_DIR)
    print(f"解压完成 → {TEMP_DIR}")


def get_sorted_image_paths(TEMP_DIR):
    """获取排序后的图像文件路径列表"""
    return sorted(
        [os.path.join(root, f)
         for root, _, files in os.walk(TEMP_DIR)
         for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: os.path.basename(x)
    )


def process_images(file_list, SELECT_NUM, TARGET_SIZE, output_dir):
    """处理图像并返回计数结果"""
    success_count = 0
    processed_count = 0

    with tqdm(total=SELECT_NUM, desc="处理进度") as pbar:
        for img_path in file_list:
            if success_count >= SELECT_NUM:
                break

            output_path = os.path.join(output_dir, f"{success_count:06d}.png")
            if process_image(img_path, output_path, TARGET_SIZE):
                success_count += 1
                pbar.update(1)

            processed_count += 1

    return success_count, processed_count


def report_results(success_count, processed_count, SELECT_NUM):

    if success_count < SELECT_NUM:
        print(f"不足{SELECT_NUM}张非灰度图（实际找到：{success_count}张）")
    else:
        print(f"成功收集{SELECT_NUM}张非灰度图")

    print(f"共扫描图像：{processed_count}张")


def data_set256(tar_path, temp_dir, output_dir, select_num, get_size):
    # 准备输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 执行处理流程
    extract_tar_if_needed(tar_path, temp_dir)
    image_files = get_sorted_image_paths(temp_dir)
    success, total = process_images(image_files, select_num, get_size, output_dir)

    # 输出结果报告
    report_results(success, total, select_num)

def convert_images_to_yuv420(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    for filename in files:
        output_name = os.path.splitext(filename)[0] + '.yuv'
        output_path = os.path.join(output_dir, output_name)

        img_path = os.path.join(input_dir, filename)


        try:
            bgr = cv2.imread(img_path)

        except Exception as e:
            print(f"图像处理失败 {filename}: {str(e)}")
            continue

        # 转换为YUV420格式
        yuv420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)

        # # 打印YUV文件维度
        # print(yuv420.shape)

        with open(output_path, 'wb') as yuv_file:
            yuv_file.write(yuv420.tobytes())

def set_yuv(input_dir, output_dir):
    convert_images_to_yuv420(input_dir, output_dir)
    print(f"转换完成！所有YUV文件保存在 {output_dir}")


if __name__ == "__main__":
    # 配置参数
    TAR_PATH = "D:\ILSVRC2012_img_val.tar"  # 输入原始数据集的压缩文件路径(若有)
    TEMP_DIR = "D:\emperdata"  # 原始数据集（或者原始数据集的解压路径）
    TARGET_SIZE = (256, 256)  # 目标分辨率
    SELECT_NUM = 100  # 输出图片数量
    OUTPUT_DIR = "D:\original_data"  # 构建数据集目录
    outyuv_path = "D:\POC\yuv_output"  # YUV文件输出路径

    # 构建所需图片数据集
    data_set256(TAR_PATH, TEMP_DIR, OUTPUT_DIR, SELECT_NUM, TARGET_SIZE)

    # 构建yuv（4:2:0）格式数据集
    set_yuv(OUTPUT_DIR, outyuv_path)