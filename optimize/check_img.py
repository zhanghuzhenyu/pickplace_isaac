import numpy as np
from PIL import Image
import io
import h5py

dataset_path = "/home/zhang/act_dataset/pick/episode_70.hdf5"
start_ts = 100
with h5py.File(dataset_path, 'r') as root:
    image_dict = dict()


    image_binary = root['rbg_image_fix_camera']
    length_dataset = root['rbg_image_fix_binary_length']
    video_frames = []    
    for k in range(len(image_binary)):
        actual_data_length = length_dataset[k]
        i_b = image_binary[k, :int(actual_data_length)]
        img_buffer = io.BytesIO(i_b)
        image = Image.open(img_buffer)
        video_frames.append(image)
    # video_frames = np.array(video_frames)
    image_dict['fixed'] = video_frames[start_ts]

    image_binary_hand = root['rbg_image_ur_front_left']
    length_dataset_hand = root['rbg_image_hand_binary_length']
    video_frames_hand = []
    for k in range(len(image_binary_hand)):
        actual_data_length_hand = length_dataset_hand[k]
        i_b = image_binary_hand[k, :int(actual_data_length_hand)]
        img_buffer = io.BytesIO(i_b)
        image = Image.open(img_buffer)
        video_frames_hand.append(image)
    # video_frames = np.array(video_frames)
    image_dict['left_hand'] = video_frames_hand[start_ts]

image_arr_fix = np.array(image_dict['fixed'])
image_arr_hand = np.array(image_dict['left_hand'])

try:
    image_dict['fixed'].show()
    
except IOError:
    print("无法从扩展的数据中恢复图像")
