import h5py
import numpy as np

# 测试数据
data = {
    'test_int': 123,
    'test_array': np.random.rand(10, 10),
    'test_bytes': b"example"
}

with h5py.File('test.h5', 'w') as file:
    for key, value in data.items():
        if isinstance(value, (int, float)):
            dtype = 'f' if isinstance(value, float) else 'i8'
            ds = file.create_dataset(name=key, shape=(1,), maxshape=(None,), dtype=dtype)
            ds[0] = value
        elif isinstance(value, np.ndarray):
            ds = file.create_dataset(name=key, shape=(1,) + value.shape, maxshape=(None,) + value.shape, dtype=value.dtype)
            ds[0] = value
        elif isinstance(value, bytes):
            dtype = h5py.special_dtype(vlen=bytes)
            ds = file.create_dataset(name=key, shape=(1,), maxshape=(None,), dtype=dtype)
            ds[0] = value
        print(f"数据 {key} 已写入HDF5文件")

            # with h5py.File(self.log_path, 'w') as file:
            #     for key, value in data.items():
            #         # 检查数据类型并设置相应的HDF5 dataset参数
            #         if isinstance(value, int) or isinstance(value, float):
            #             shape = (1,)
            #             maxshape = (None,)
            #             file.create_dataset(name = key, shape = shape,  maxshape = maxshape)
            #             file[key][0] = value # add fist frame
            #         elif isinstance(value, np.ndarray):
            #             shape = (1,) + value.shape
            #             maxshape = (None,) + value.shape
            #             file.create_dataset(name = key, shape = shape,  maxshape = maxshape)
            #             file[key][0] = value # add fist frame
            #         elif isinstance(value, list) or isinstance(value, tuple):
            #             # 转换list或tuple为numpy数组以获得shape
            #             value = np.array(value)
            #             shape = (1,) + value.shape
            #             maxshape = (None,) + value.shape
            #             file.create_dataset(name = key, shape = shape,  maxshape = maxshape)
            #             file[key][0] = value # add fist frame
            #         elif isinstance(value, bytes):  # 处理二进制数据，如JPEG图像数据
            #             shape = (1,)
            #             maxshape = (None,)
            #             dtype = h5py.special_dtype(vlen=bytes)  # 可变长度的二进制数据类型
            #             file.create_dataset(name=key, shape=shape, maxshape=maxshape, dtype=dtype)
            #             file[key][0] = value  # 添加第一帧数据
            #         else:
            #             continue  # 如果数据类型未知，跳过

                    # # 创建数据集
                    # ds = file.create_dataset(name=key, shape=shape, maxshape=maxshape, dtype=dtype)
                    # ds[key][0] = value  # 添加第一帧数据