import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_image_montage_from_hdf5(file_path, montage_path, step_interval=10):
    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as file:
        # 读取图像数据
        images_data = file['rbg_image_fix_camera'][:]
    
    # 假设每帧图像的尺寸是固定的
    height, width = 256, 256  # 根据实际图像尺寸调整这些值

    # 准备一个大图的画布
    frames_count = len(images_data)
    selected_frames = [np.frombuffer(images_data[i], dtype=np.uint8) for i in range(0, frames_count, step_interval)]
    images = [cv2.imdecode(frame, cv2.IMREAD_COLOR) for frame in selected_frames]
    images = [cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR) for img in images]

    # 计算需要的网格大小
    grid_size = int(np.ceil(np.sqrt(len(images))))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # 将图片放到画布上
    for i, img in enumerate(images):
        ax = axs[i // grid_size, i % grid_size]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    
    # 剩下的子图隐藏
    for j in range(i + 1, grid_size * grid_size):
        axs[j // grid_size, j % grid_size].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(montage_path)
    plt.show()

# 指定 HDF5 文件路径和图像拼接保存路径
file_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/4-28-2.h5'
montage_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/img_4-28-2.png'

# 创建图像拼接
create_image_montage_from_hdf5(file_path, montage_path)
print("图像拼接已保存至", montage_path)
