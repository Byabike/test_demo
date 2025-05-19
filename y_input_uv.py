import os
import cv2
import numpy as np
import json
from itertools import islice


def read_specific_frame(jsonl_path, frame_number=1):
    """读取JSONL文件中指定行号的数据(行号从1开始)"""
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if line_num == frame_number:
                return json.loads(line.strip())
        raise ValueError(f"文件不足{frame_number}行")


def create_y_matrices(data):
    y_matrix = np.array(data["y"]).reshape(256, 256)

    return y_matrix


def replace_y_channel(image_path, y_matrix, out_pic_dir, i):
    assert y_matrix.shape == (256, 256), "Y矩阵必须是256x256"
    assert os.path.exists(image_path), f"输入图片不存在: {image_path}"

    # 创建输出目录
    os.makedirs(out_pic_dir, exist_ok=True)

    # 读取图像并转换为 float32 类型 (范围 0-255)
    img_bgr = cv2.imread(image_path).astype(np.float32)

    # 转换为 YUV 并保持浮点运算
    yuv_float = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

    # 替换 Y 通道
    yuv_float[:, :, 0] = y_matrix

    # 转回 BGR 浮点图像
    reconstructed_bgr_float = cv2.cvtColor(yuv_float, cv2.COLOR_YUV2BGR)

    # 四舍五入
    reconstructed_bgr = np.clip(np.round(reconstructed_bgr_float), 0, 255).astype(np.uint8)

    # 保存图片
    output_path = os.path.join(out_pic_dir, f"{i:%06d}.png")
    cv2.imwrite(output_path, reconstructed_bgr)


def ydata_input(final_frame, ydata_dir, uv_dir, out_pic):

    os.makedirs(out_pic, exist_ok=True)

    for line_number in range(0, final_frame):
        frame_data = read_specific_frame(ydata_dir, line_number + 1)

        y_mat = create_y_matrices(frame_data)
        poc_dir = os.path.join(uv_dir, f"frame_{line_number}.png")

        replace_y_channel(poc_dir, y_mat, out_pic, line_number)

if __name__ == "__main__":
    # 参数配置
    sun_poc = 1000                  # 总帧数
    ydata_dir = "/media/D/dutanlong/NTSCC/final_ntc/result/loss/test/ydata/snr109.jsonl"     # y通道重建数据地址
    uvimg_dir = "/media/D/yangxuzhi/test3/cmake-build-debug/decode_uvg/NTC/uv-10.9"          # uv通道重建图片地址
    out_pic = "imagedata/y-uv-snr10.9-np"                                                    # 方案最终重建结果图
    ydata_input(sun_poc, ydata_dir, uvimg_dir, out_pic)