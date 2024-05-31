import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

# Load the h5 file
file_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/5-1/pick_example.h5'
with h5py.File(file_path, 'r') as file:
    end_effector_position = np.array(file['end_effector_position'])
    object_position = np.array(file['object_position'])
    timestep = np.array(file['current_time_step'])
    event_list = np.array(file['pickplace_event'])
    end_effector_orientation = np.array(file['end_effector_orientation'])
    object_orientation = np.array(file['object_orientation'])

# 转换四元数为方向向量，这里需要具体的转换逻辑
def quaternion_to_vector(quaternion):
    # 这里是一个简化的例子，具体需要根据四元数转换为向量的方法
    return quaternion[1:4]  # 假设仅返回虚部

# 方向向量
ee_direction_vectors = np.apply_along_axis(quaternion_to_vector, 1, end_effector_orientation)
object_direction_vectors = np.apply_along_axis(quaternion_to_vector, 1, object_orientation)


# Determine the indices where events change
change_indices = np.where(np.diff(event_list) != 0)[0] + 1  # Add 1 because diff shifts index by 1

phase_boundaries = [0, timestep[-1]]
# event = 0
# phase_boundaries = [0]
# for i in range(404):
#     if event_list[i] != event:
#         phase_boundaries.append(int(i))
#     event = event_list[i]

# Define phase boundaries and colors for visualization
# phase_boundaries = [0, 50, 100, 150, 200, 250, 350, 404]  # Adjust these as per your analysis
phase_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
phase_labels = [
    # 'Phase 0: Initialize',
    'Phase 1: Move above cube',
    'Phase 2: Lower to cube',
    'Phase 3: Inertia settle',
    'Phase 4: Close grip',
    'Phase 5: Lift block',
    'Phase 6: Return to init'
]

# Plotting the trajectory with object interaction in 3D space
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory of the end effector
# for i in range(len(phase_boundaries)-1):
#     start, end = phase_boundaries[i], phase_boundaries[i+1]
start = 0
end = int(timestep[-1])
i = 0
ax.plot(end_effector_position[start:end, 0], end_effector_position[start:end, 1], end_effector_position[start:end, 2],
        label=phase_labels[i], color=phase_colors[i])

# Plot the object position
# ax.scatter(object_position[:, 0], object_position[:, 1], object_position[:, 2], color='orange', s=30, label='Object Position')
ax.plot(object_position[:, 0], object_position[:, 1], object_position[:, 2], linestyle='--', color='orange', label='Object Position')


ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectory of UR5 End Effector with Object Interaction')
ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1))

# plt.show()
plt.savefig('/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/5-1/ee_traj_3d.png')
plt.close()
