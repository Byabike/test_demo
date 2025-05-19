import json
import os
import numpy as np
import torch
from torch_dct import dct_2d
from itertools import islice

# 设置GPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_specific_frame(jsonl_path, frame_number=1):
    """读取指定帧的JSONL数据"""
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num == frame_number:
                return json.loads(line.strip())
        raise ValueError(f"文件不足{frame_number}行")


def validate_coefficients(data):
    """验证系数矩阵尺寸"""
    assert len(data["y"]) == 256 * 256, "Y系数数量错误"
    assert len(data["u"]) == 128 * 128, "U系数数量错误"
    assert len(data["v"]) == 128 * 128, "V系数数量错误"


def create_yuv_matrices(data):
    """创建YUV矩阵"""
    return (
        np.array(data["y"]).reshape(256, 256),
        np.array(data["u"]).reshape(128, 128),
        np.array(data["v"]).reshape(128, 128)
    )


def parse_tu_file(file_path, line_number=1):
    """解析TU划分数据"""
    tu_data = []
    try:
        with open(file_path, 'r') as f:
            line = next(islice(f, line_number - 1, line_number), None)
            if not line:
                raise ValueError(f"文件不足{line_number}行")

            parts = line.strip().split()
            frame = int(parts[0])
            for tu_str in parts[1:]:
                channel, y, x, size = tu_str.split('_')
                tu_data.append({
                    'channel': channel,
                    'position': (int(y), int(x)),
                    'tu_size': int(size)
                })
    except Exception as e:
        print(f"解析TU文件错误: {str(e)}")
    return tu_data


def process_block(src_mat, target_mat, orig_pos, size, a):
    """处理单个TU块的核心逻辑"""
    # 计算源矩阵起始位置
    start_y = orig_pos[0] // a
    start_x = orig_pos[1] // a

    # 源矩阵边界检查
    max_src_y = src_mat.shape[0] - size
    max_src_x = src_mat.shape[1] - size
    start_y = max(0, min(start_y, max_src_y))
    start_x = max(0, min(start_x, max_src_x))

    # 目标矩阵边界计算
    target_start_y = orig_pos[0]
    target_start_x = orig_pos[1]
    target_end_y = target_start_y + size
    target_end_x = target_start_x + size

    # 实际处理尺寸
    actual_size = min(
        size,
        256 - target_start_y,
        256 - target_start_x
    )

    if actual_size <= 0:
        return  # 跳过无效块

    # 从源矩阵提取数据
    src_block = src_mat[start_y:start_y + actual_size, start_x:start_x + actual_size]
    tensor = torch.from_numpy(src_block).float().to(device)

    # DCT变换和取整
    dct_coeffs = dct_2d(tensor, 'ortho')
    rounded = torch.round(dct_coeffs).int()

    # 处理不同通道的写入方式
    if a == 1:
        # Y通道直接写入
        target_mat[target_start_y:target_start_y + actual_size,
        target_start_x:target_start_x + actual_size] = rounded
    else:
        # UV通道写入
        for i in range(rounded.shape[0]):
            for j in range(rounded.shape[1]):
                target_y = target_start_y + i
                target_x = target_start_x + j
                if target_y < 256 and target_x < 256:
                    target_mat[target_y, target_x] = rounded[i, j]

def res_uvdatas(sum_poc, coeff_path, tu_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for frame_idx in range(sum_poc):
        try:
            # 数据读取与验证
            frame_data = read_specific_frame(coeff_path, frame_idx + 1)
            validate_coefficients(frame_data)
            y_mat, u_mat, v_mat = create_yuv_matrices(frame_data)

            # TU数据解析
            tu_blocks = parse_tu_file(tu_path, frame_idx + 1)

            matY = torch.zeros((256, 256), dtype=torch.int32, device=device)
            matU = torch.zeros((256, 256), dtype=torch.int32, device=device)
            matV = torch.zeros((256, 256), dtype=torch.int32, device=device)

            # 处理每个TU块
            for block in tu_blocks:
                channel = block['channel']
                orig_pos = block['position']
                size = block['tu_size']

                if channel == 'y':
                    process_block(y_mat, matY, orig_pos, size, a=1)
                elif channel == 'u':
                    process_block(u_mat, matU, orig_pos, size, a=2)
                elif channel == 'v':
                    process_block(v_mat, matV, orig_pos, size, a=2)

            # 将结果写回CPU并保存
            for channel, mat in [('y', matY), ('u', matU), ('v', matV)]:
                np_mat = mat.cpu().numpy()
                frame_dir = os.path.join(output_dir, f"{frame_idx:06d}")
                os.makedirs(frame_dir, exist_ok=True)

                with open(os.path.join(frame_dir, f"{channel}.txt"), 'wb') as f:  # 使用二进制模式
                    flattened = [num for row in np_mat for num in row]

                    # 按每256个元素分块处理
                    for i in range(0, len(flattened), 256):
                        chunk = flattened[i:i + 256]
                        # 每个元素严格占用5字节（4位数字+1空格），末尾保留空格
                        formatted_line = ''.join(f"{num:4d} " for num in chunk)
                        # 编码为字节并添加换行符
                        line_bytes = formatted_line.encode('ascii') + b'\n'
                        f.write(line_bytes)

        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {str(e)}")
            continue

if __name__ == '__main__':

    # 配置路径参数
    sum_poc = 1000
    coeff_path = "/media/D/dutanlong/NTSCC/final_ntc/result/loss/test/uvdata/snr107.jsonl"
    # TU规则地址（TXT）
    tu_path = "/media/D/yangxuzhi/test3/cmake-build-debug/POC/tu_rules-last1000.txt"
    # 重建系数主目录
    output_dir = "RES-pic-coeff/uv-10.7-5-10"

    res_uvdatas(sum_poc, coeff_path, tu_path, output_dir)