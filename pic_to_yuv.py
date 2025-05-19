import os
import cv2
import re
from PIL import Image
import numpy as np


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
            pil_img = Image.open(img_path)

            # 转换为OpenCV需要的BGR格式
            bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
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
    # 参数配置
    input_path = "D:\pocdata"  # 图片输入路径
    output_path = "D:\POC\yuv_output"  # YUV文件输出路径

    set_yuv(input_path, output_path)