import h5py
import cv2
import numpy as np

def create_video_from_hdf5(file_path, video_path):
    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as file:
        # 读取图像数据
        images_data = file['rbg_image_fix_camera'][:]
        # images_data = file['rbg_image_ur_front_left'][:]
        
    
    # 确定视频的分辨率和编码
    # 假设每帧图像的尺寸是固定的，例如 640x480
    height, width = 480, 640  # 根据实际图像尺寸调整这些值
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
    video = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
    
    for img_data in images_data:
        # 解码图像数据
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # 检查图像尺寸，并调整到视频的分辨率
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        # 写入视频帧
        video.write(img)
    
    # 释放视频文件
    video.release()

# 指定 HDF5 文件路径和视频文件保存路径
file_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/DATA/logs/place/episode_1.hdf5'
video_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/5-6/place_fix.mp4'

# 创建视频
create_video_from_hdf5(file_path, video_path)
print("视频已保存至", video_path)
