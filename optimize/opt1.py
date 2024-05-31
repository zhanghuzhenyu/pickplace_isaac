import h5py
import numpy as np
from scipy.signal import savgol_filter

def smooth_trajectory(data, window_length, polyorder):
    # 仅当窗口长度小于数据长度时应用滤波器
    if len(data) > window_length:
        return savgol_filter(data, window_length, polyorder)
    return data

def optimize_trajectory(file_path):
    # 打开原始 HDF5 文件
    with h5py.File(file_path, 'r+') as file:
        # 读取 UR5 机械臂的目标关节位置
        ur5_positions = file['ur_5_applied_joint_positions'][:]
        
        # 应用平滑滤波器优化 UR5 关节位置
        window_length = 15  # 窗口长度，需要根据数据长度可能需要调整
        polyorder = 3  # 多项式阶数
        optimized_positions = np.apply_along_axis(smooth_trajectory, 0, ur5_positions, window_length, polyorder)
        
        # 替换原始数据集中的数据
        del file['ur_5_applied_joint_positions']  # 删除原始数据集
        file.create_dataset('ur_5_applied_joint_positions', data=optimized_positions)  # 创建新的优化数据集

        # 如果有必要，可以添加对 Husky 速度的优化
        # 例如平滑速度曲线或重新计算速度以匹配优化的位置

    print("轨迹优化完成，数据已更新")

# 文件路径
file_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/episode_0.hdf5'

# 执行优化
optimize_trajectory(file_path)
