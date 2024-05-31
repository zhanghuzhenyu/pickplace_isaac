import h5py
from PIL import Image
import numpy as np
import io

def read_and_show_image_from_hdf5(file_path, image_key, length_key, time_step, image_shape=(256, 256)):
    """
    从HDF5文件中读取并显示指定时间步的图像，使用存储的二进制长度。

    :param file_path: HDF5文件路径
    :param image_key: 存储图像数据的键名
    :param length_key: 存储每个时间步图像数据长度的键名
    :param time_step: 时间步的索引
    :param image_shape: 图像的尺寸，以(height, width)形式提供
    """
    with h5py.File(file_path, 'r') as file:
        # 确认键和时间步存在
        print(file.keys())

        image_dataset = file[image_key]
        length_dataset = file[length_key]
        if time_step < image_dataset.shape[0] and time_step < length_dataset.shape[0]:
            # 读取指定时间步的图像数据长度
            actual_data_length = length_dataset[time_step]
            # 读取指定时间步的图像数据
            image_data = image_dataset[time_step, :int(actual_data_length)]

            # 将图像数据转换为图像
            # image = Image.frombytes('RGB', image_shape, image_data.tobytes())
            buffer = io.BytesIO(image_data)
            image = Image.open(buffer)  # 这里使用Image.open处理压缩的JPEG数据
            
            # 显示图像
            image.show()
            # 显示图像
            # image.show()
        else:
            print("时间步超出范围")


# 示例用法
file_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/DATA/logs/pick_husky_control_2024-05-01_00-14-29_a6d3242b-b6f3-4e60-bc97-51d0393c0f49_.h5'  # 替换为你的HDF5文件路径

# image_key = 'rbg_image_fix_camera'
# length_key = 'rbg_image_fix_binary_length'
image_key = 'rbg_image_ur_front_left'
length_key = 'rbg_image_hand_binary_length'
time_step = 50  # 选择一个具体的时间步
image_shape = (256, 256)  # 假设图像尺寸为256x256

read_and_show_image_from_hdf5(file_path, image_key, length_key, time_step, image_shape)

