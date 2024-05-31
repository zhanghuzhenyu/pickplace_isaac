# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Callable, List, Dict
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.types import DataFrame

from configs.main_config import MainConfig

from datetime import datetime
import pkg_resources
import subprocess
import numpy as np
import importlib
import json
import uuid
import os
import sys

import os
import re
# from mpi4py import MPI
import h5py





class DataLogger:
    """ Parallel trajectories Logger using HDF5: https://docs.h5py.org
    """
    def __init__(self, 
                config: MainConfig,
                task: None
                ) -> None:
        self.config = config
        self._pause = True
        self._data_frames = []
        self.task = task
        self.configure_paths()


    def configure_paths(self) -> None:
        """Configure paths for logging, filename format: init_log_name + curr_datetime + uid
        """
        # uid = str(uuid.uuid4())

        # curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # log_name = '_'.join([self.task, self.config.log_name, curr_datetime, uid, '.h5'])
        # self.log_path = os.path.join(self.config.log_folder_path, log_name)


        # 列出所有的 .hdf5 文件
        log_folder_path_with_task = os.path.join(self.config.log_folder_path, self.task)
        files = [f for f in os.listdir(log_folder_path_with_task) if f.endswith('.hdf5')]

        # 初始化最大索引为0
        max_index = 0

        # 正则表达式来找到数字部分
        pattern = re.compile(r'episode_(\d+)\.hdf5')

        for file in files:
            match = pattern.search(file)
            if match:
                # 将找到的数字转换为整数并更新最大索引
                index = int(match.group(1))
                if index > max_index:
                    max_index = index

        # 确定下一个文件的索引
        next_index = max_index + 1
        log_name = f'episode_{next_index}.hdf5'
        self.log_path = os.path.join(log_folder_path_with_task, log_name)

        return 



    def add_data(self, data: dict) -> None:
        """ Write data paraller to the log files 

        Args:
            data (dict): Dictionary representing the data to be logged at this time index.

        """
        # h5py.File(self.log_path, 'w', driver='mpio', comm=MPI.COMM_WORLD )for parallel data preprocessing

        img_len = 25000
        def prepare_binary_data(binary_data, target_length=img_len, fill_char=b'\x01'):
            """调整二进制数据到固定长度，不足补指定字符，超出截断。"""
            if len(binary_data) > target_length:
                print("Error\nError\nError\nError\nError\nError\nError\n")
                sys.exit()
                return binary_data[:target_length]  # 截断到固定长度
            elif len(binary_data) < target_length:
                return binary_data.ljust(target_length, fill_char)  # 不足部分补指定字符
            return binary_data

        
        if not os.path.isfile(self.log_path): # log does not exis 
            with h5py.File(self.log_path, 'w') as file:

                for key, value in data.items():
                    if isinstance(value, int) or isinstance(value, float) == int:
                        shape = (1,)
                        maxshape = (None,)
                    elif isinstance(value, np.ndarray):
                        shape = (1,) + value.shape
                        maxshape = (None,) + value.shape
                    elif isinstance(value, list) or isinstance(value, tuple):
                        shape =  (1,) + np.array(value).shape
                        maxshape = (None,) + value.shape
                    elif isinstance(value, bytes):
                        adjusted_value = prepare_binary_data(value)
                        file.create_dataset(name=key, shape=(1, img_len), maxshape=(None, img_len), dtype='uint8')
                        file[key][0, :] = np.frombuffer(adjusted_value, dtype='uint8')
                        continue

                    flag_type = isinstance(value, bytes)
                    file.create_dataset(name = key, shape = shape,  maxshape = maxshape)
                    file[key][0] = value # add fist frame

        else:
            with h5py.File(self.log_path, 'a') as file:
                for key, value in data.items():
                    
                    # dataset = file[key]

                    # resize for new element for element 
                    if isinstance(value, int) or isinstance(value, float):
                        dataset = file[key]
                        num_values = dataset.shape[0]
                        new_shape = (num_values + 1,)

                        dataset.resize(new_shape)
                    elif isinstance(value, np.ndarray):
                        dataset = file[key]
                        old_shape = dataset.shape
                        num_values = dataset.shape[0] 
                        new_shape = (num_values + 1,) + dataset.shape[1:]

                        dataset.resize(new_shape)
                    elif isinstance(value, bytes):
                        adjusted_value = prepare_binary_data(value)
                        dataset = file[key]
                        num_values = dataset.shape[0]
                        new_shape = (num_values + 1, img_len)
                        dataset.resize(new_shape)
                        dataset[num_values, :] = np.frombuffer(adjusted_value, dtype='uint8')
                        continue

                    dataset[num_values] = value

        return

    def get_num_of_data_frames(self) -> int:
        """

        Returns:
            int: the number of data frames collected/ retrieved in the data logger.
        """

        return len(self._data_frames)

    def pause(self) -> None:
        """Pauses data collection.
        """
        self._pause = True
        return

    def start(self) -> None:
        """Resumes/ starts data collection.
        """
        self._pause = False
        return

    def is_started(self) -> bool:
        """
        Returns:
            bool: True if data collection is started/ resumed. False otherwise.
        """
        return not self._pause

    def reset(self) -> None:
        """Clears the data in the logger.
        """
        self._pause = True

        return

    def load(self, log_path: str) -> None:
        """Loads data from dataset to read back a previous saved data or to resume recording data from another time step.

        Args:
            log_path (str): path of the hdf5 file to be used to load the data.
        """

        return
