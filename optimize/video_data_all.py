import h5py
import cv2
import numpy as np
import os

def create_video_from_hdf5(file_path, video_path):
    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as file:
        # 读取图像数据
        images_data = file['rbg_image_ur_front_left'][:]  # 确保这个 key 是正确的
        # rbg_image_fix_camera
        # rbg_image_ur_front_left
    
    # 确定视频的分辨率和编码
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

def create_videos_for_all_episodes(folder_path, output_folder):
    # 遍历目录下的所有 HDF5 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.hdf5'):
            episode_number = filename.split('_')[1].split('.')[0]  # 从文件名中提取编号
            video_path = os.path.join(output_folder, f'episode_{episode_number}_mp4.mp4')  # 构造视频文件路径
            file_path = os.path.join(folder_path, filename)
            create_video_from_hdf5(file_path, video_path)
            print("视频已保存至", video_path)

# 指定 HDF5 文件夹路径和视频文件保存文件夹路径
folder_path = '/home/zhang/act_dataset/pick'
output_folder = '/home/zhang/act_dataset/video_pick_hand'

# 生成视频
create_videos_for_all_episodes(folder_path, output_folder)
