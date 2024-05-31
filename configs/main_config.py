from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, Optional, Tuple, Union

import numpy as np
@dataclass
class MainConfig: 


    # assets paths TO-DO: make paths relative 
    husky_usd_path: str =  '/home/zhang/asset/pipeline_asset/husky_with_sensors.usd'
    husky_stage_path: str = '/World/husky'
    ur5_usd_path: str = '/home/zhang/asset/pipeline_asset/ceshi.usd'
    ur5_stage_path: str = '/World/ur5'
    # end_effector_stage_path: str = '/World/ur5/Gripper/robotiq_arg2f_base_link'
    end_effector_stage_path: str = '/World/ur5/robotiq_85_base_link'  # 在程序中先使用直接路径 不从config调用
    
    
    # setup task params: 
    log_camera_data: bool = True
    log_lidar_data: bool = True

    # 
    # ur5_relative_pose: Tuple[float, ...] = (0.3312, 0, 0.407)
    ur5_relative_pose: Tuple[float, ...] = (0.3312, 0, 0.257)
    
    # ur5_init_pose: Tuple[float, ...] = (0, -0.5, -2, 0, 1.570, 0)
    ur5_init_pose: Tuple[float, ...] = (3.1415927, -2.871760, 2.799204, -3.072348, -1.581982, -0.000120)
    joints_default_positions: Tuple[float, ...] = (3.1415927, -2.871760, 2.799204, -3.072348, -1.581982, -0.000120)
    # joints_default_positions: Tuple[float, ...] = (-0.000120, -1.581982, -3.072348, 2.799204, -2.871760, 3.1415927)
    end_effector_initial_height: float = 0.2

    # scene setup 
    env_usd_path: str = '/home/zhang/asset/DATA_asset/warehouse_7.usd'
    env_name: str = 'warehouse_7'
    env_obj_name: str = 'Warehouse'
    env_prim_path: str = '/World/' + env_name

    # object setup
    # object_usd_path: str =  '/home/zhang/asset/DATA_asset/RubixCube.usd' # for import custom usd model
    object_usd_path: str =  '/home/zhang/asset/DATA_asset/cube.usd' # for import custom usd model
    object_name: str = 'RubixCube'
    object_prim_path: str = '/World/' + object_name
    # object_init_position: Tuple[float, ...] = (2.31, 2.324, 0.39)
    ################################################################

    def get_random_position():
        # 使用numpy生成随机坐标
        x = np.random.uniform(0.7, 0.9)
        y = np.random.uniform(-0.25, 0.25)
        z = 0.45  # z坐标固定为0.45
        return (x, y, z)
    object_init_position = get_random_position()
    # object_init_position: Tuple[float, ...] = (0.8, 0, 0.45) 
    table_init_position: Tuple[float, ...] = (0.9, 0, 0)
    ################################################################
    object_scale: Tuple[float, ...] = (0.05, 0.05, 0.05)
    object_init_orientation: Tuple[float, ...] = (0.70711, -0.70711, 0, 0)

    # husky setup
    # husky_init_pose: Tuple[float, ...] = (0, 0, 0.168)
    husky_init_pose: Tuple[float, ...] = (0, 0, 0.168)
    def get_random_position_place():
        # 使用numpy生成随机坐标
        x = np.random.uniform(0.7, 0.9)
        y = np.random.uniform(-0.25, 0.25)
        z = 0.45  # z坐标固定为0.45
        return (x, y, z)
    # target_position: Tuple[float, ...] = (0.8, 0, 0.45)
    target_position = get_random_position_place()

    # controllers_setup:
    wheel_radius: float = 0.3
    wheel_base: float =  0.5

    lateral_velocity: float = 1.3
    yaw_velocity: float = 1.2
    position_tol: float = 0.3

    # pick_place controll 
    events_dt: Tuple[float, ...] = (0.016, 0.02, 0.1, 0.02, 0.0150, 0.01, 0.012, 0.016, 0.008, 0.08)
    # events_dt: Tuple[float, ...] = (0.0039, 0.0030, 0.09, 0.0027, 0.0031, 0.003, 0.0025, 1, 0.008, 0.08)#(0.2, 0.2, 1, 0.05, 0.01, 0.05, 0.008, 0.1, 0.008, 0.08) #
    # end_effector_offset: Tuple[float, ...] = (-0.003, -0.003, +0.035)
    end_effector_offset: Tuple[float, ...] = (0,0,0)
    gripper_joint_closed_positions: Tuple[float, ...] = (0.42, -0.42)
    # trans_pint_relative_path: str = 'ur5_ee_link'
    trans_pint_relative_path: str = 'base_link_inertia/trans_point'

    # camera setup

    # relative path to camera, absolute path start from husky and ur5 like = /World/Husky,World/ur5
    # cameras: Tuple[str, ...] = ('fence_link/zed/husky_front_right','fence_link/zed/husky_front_left','ur5_ee_link/realsense/ur_front_right', 'ur5_ee_link/realsense/ur_front_left')
    
    # don't forget comma in tuple if you use obly one camera!!!
    # cameras: Tuple[str, ...] = ('fence_link/zed/husky_front_right', )
    # camera_prim_path: str = '/World/Husky/fence_link/zed/husky_front_right'

    cameras: Tuple[str, ...] = ('fence_link/zed/husky_front_right', )
    camera_prim_path: str = '/World/ur5/realsense/ur_front_left' # /World/ur5/realsense/ur_front_right

    depth_img_size: Tuple[float, ...] = (128, 128)
    img_size: Tuple[float, ...] = (128, 128, 3)

    # lidar setup - so heavy to render

    # relative to husky prim path
    
    lidar_relative_path: str = '/fence_link/fence_link_small/VLP_16/vlp16/lidar'

    # lidar configs, for more data check ./exts/omni.isaac.sensor/data/lidar_configs/ or https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_based_lidar.html#rtx-lidar-config-library
    lidar_pos: Tuple[float, ...] = (0., 0., 0.0147)
    min_range: float = 0.1
    max_range: float = 100.0
    draw_points: bool = True
    draw_lines: bool = False
    horizontal_fov: float = 360.0
    vertical_fov: float = 60.0
    horizontal_resolution: float = 0.4
    vertical_resolution: float = 0.4
    rotation_rate: float = 0.0
    high_lod: bool = True
    yaw_offset: float = 0.0
    enable_semantics: bool = True

    # logging setup 
    log_folder_path: str = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/DATA/logs/'
    log_name: str = 'husky_control'
