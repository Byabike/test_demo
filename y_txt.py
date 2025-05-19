import os
import cv2
import re
import numpy as np


def process_images_to_txt(input_dir, output_file):

    # 获取文件列表并按5位数字排序
    files = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):

            match = re.search(r'\d+', fname)
            if match:
                files.append((int(match.group()), fname))

    # 按提取的5位数字排序
    files.sort(key=lambda x: x[0])

    with open(output_file, 'w') as f_out:
        for idx, (num, fname) in enumerate(files, start=1):
            img_path = os.path.join(input_dir, fname)

            # 读取图像并转换YUV
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：跳过无法读取的文件 {fname}")
                continue

            # 转换为YUV并提取Y通道
            img_float = img.astype(np.float32)
            yuv = cv2.cvtColor(img_float, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]

            # 生成行前缀（5位序号 + 空格）
            line_prefix = f"{idx - 1:05d} "

            # 格式化像素数据
            pixel_lines = []
            for row in y_channel:
                formatted_row = [f"{pixel:.6f} " for pixel in row]
                pixel_lines.append("".join(formatted_row))

            # 合并所有行像素数据
            full_pixel_str = "".join(pixel_lines)

            # 写入文件
            f_out.write(f"{line_prefix}{full_pixel_str}\n")

    print(f"处理完成！共输出 {len(files)} 张图片数据")


if __name__ == "__main__":

    input_dir = r'D:\original_data'  # 输入目录路径
    output_file = 'y_testH265_1000.txt'  # 输出文件路径
    # 将灰度图像的Y通道像素，按照行优先的规则存储到TXT文件中
    process_images_to_txt(input_dir, output_file)