import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import cv2

# Load the h5 file
file_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/5-2/place_husky_control_2024-05-02_18-54-33_1444b4da-9b74-47dc-be7a-ef200844aa3c_.h5'
with h5py.File(file_path, 'r') as file:
    end_effector_position = np.array(file['end_effector_position'])
    object_position = np.array(file['object_position'])
    end_effector_orientation = np.array(file['end_effector_orientation'])
    object_orientation = np.array(file['object_orientation'])
    event_list = np.array(file['pickplace_event'])

# Convert quaternion to vector
def quaternion_to_vector(quaternion):
    return quaternion[1:4]  # Adjust as necessary

# Convert orientations to vectors
ee_direction_vectors = np.apply_along_axis(quaternion_to_vector, 1, end_effector_orientation)
object_direction_vectors = np.apply_along_axis(quaternion_to_vector, 1, object_orientation)


# Initialize video settings
video_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/LLM/replay_data/optimize/5-2/ee_traj_video_place.mp4'  # Change to a valid output path
fps = 10  # Frames per second for video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
frame_size = (800, 600)  # Video frame size

# Create video writer
video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

# Generate frames
for i in range(len(end_effector_position)):
    # Create a new figure for each frame
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the end effector and object positions up to the current frame
    ax.plot(end_effector_position[:i, 0], end_effector_position[:i, 1], end_effector_position[:i, 2], color='blue', label='End Effector Position')
    ax.plot(object_position[:i, 0], object_position[:i, 1], object_position[:i, 2], linestyle='--', color='orange', label='Object Position')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title('3D Trajectory Visualization')

    # Convert the plot to an image and add it to the video
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape((frame_size[1], frame_size[0], 3))
    video_writer.write(frame)
    plt.close()

# Finalize and save the video
video_writer.release()

print(f"Video saved at {video_path}")
