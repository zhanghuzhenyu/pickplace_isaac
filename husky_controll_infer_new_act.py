import sys
sys.path.append('DATA/collect_srcipt_with_move/act')
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import IPython

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from act2.model.build_model import build, ACTPolicy
from act2.model.act_config import act_config

# from sim_env import BOX_POSE

eval = True


########################## LOAD CKPT #######################33

parser = argparse.ArgumentParser(description="Example of adding dict items to argparse.")
for key, value in act_config.items():
    parser.add_argument(f'--{key}', type=type(value), default=value, help=f'Set the {key}')
args = parser.parse_args()
state_dim = act_config["state_dim"]
policy = ACTPolicy(args)
policy.load_state_dict(torch.load('/home/zhang/act2/ckpt/5-8/policy_epoch_6000.ckpt'))
policy.cuda()
policy.eval()

ckpt_dir = "/home/zhang/act2/ckpt/5-8"
stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']
max_timesteps = 1000
num_queries = act_config['num_queries'] # 100
query_frequency = 1
############################ SET ACT Chunking ############################ 
temporal_agg = True
if temporal_agg:
    all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
image_list = [] # for visualization
qpos_list = []
target_qpos_list = []
rewards = []

################# Isaac 启动！ ########################
from omni.isaac.kit import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Set display options to show default grid
}

# simulation_app = SimulationApp(launch_config=CONFIG)
simulation_app = SimulationApp({"headless": False})


from omni.isaac.core import World
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage

from omni.isaac.core.utils import nucleus, stage
from pxr import UsdPhysics, Usd, Sdf, UsdLux
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import get_stage_units
from pxr import UsdPhysics
import os
import numpy as np

from tasks.pick_place import PickPlace
from configs.main_config import MainConfig
from controllers.husky_controller import HuskyController
import asyncio
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
import matplotlib.pyplot as plt

import IPython
from PIL import Image
import io
from omni.isaac.core.utils.types import ArticulationAction

############################ INIT ############################

config = MainConfig()


tasks = []
num_of_tasks = 1
husky_controllers = []

object_names = []
objects = []

my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()

############################ SETUP SCENE #############################
'''
    Add multiple tasks
'''

my_world.add_task(PickPlace(name="task_" + '0', world = my_world, config = config, offset=np.array([0, 0, 0])))

my_world.scene.add_default_ground_plane()
my_world.reset()

############################ SETUP TASKS #############################
'''
    Add info about tasks
'''

task = my_world.get_task(name="task_" + str(0))
tasks.append(task)

default_pose = np.zeros(12)
default_pose[:6] = list(config.joints_default_positions) 
task.manipulator.set_joints_default_state(positions=default_pose)    

controller = HuskyController(task, my_world, simulation_app)
husky_controllers.append(controller)    

my_world.reset()

stage = get_current_stage()
ur5_stage_path = "/World/ur5"
connection_joint_stage_path = "/World/connect_joint"
husky_stage_path = "/World/husky"
prim_root_joint = stage.GetPrimAtPath(f"{ur5_stage_path}/root_joint")
root_joint = UsdPhysics.Joint(prim_root_joint)
root_joint_enable = root_joint.GetJointEnabledAttr().Set(False)

fixed_joint = UsdPhysics.FixedJoint.Define(stage, connection_joint_stage_path)
fixed_joint.GetBody0Rel().SetTargets([f"{husky_stage_path}/put_ur5"]) 
fixed_joint.GetBody1Rel().SetTargets([f"{ur5_stage_path}/base_link"])
fixed_joint.GetExcludeFromArticulationAttr().Set(True)

# Adds a light to the scene

demoLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
demoLight.CreateIntensityAttr(500)
# distantLight.AddOrientOp().Set(Gf.Quatf(-0.3748, -0.42060, -0.0716, 0.823))

for i in range(50):
    my_world.step(render=True)
    print(i)

pick_success_infer = False
############################ START SIMULATION ############################ 
# while simulation_app.is_running() and not simulation_app.is_exiting() and (tasks[0].data_step < max_timesteps):  # 这一行用于Inference
while simulation_app.is_running() and not simulation_app.is_exiting() and (not pick_success_infer):  # 这一行用于Inference
    my_world.step(render=True)

    if my_world.is_playing():

        for i in range(num_of_tasks):
            
            phys_time = my_world.current_time_step_index
            print(f"phys_time:{phys_time}")
            current_observations = tasks[i].get_observations()
            # print(current_observations)

            if eval:                
                # data = tasks[i].data_frame_logging_func()
                data = tasks[i].data_frame_logging_func(task_type = "eval",)
                with torch.inference_mode():
                    policy.eval()
                    
                    qpos_arm = data["ur_5_joint_positions"]
                    qpos_gripper = data["gripper_joint_positions"]
                    qpos = np.concatenate((qpos_arm, qpos_gripper), axis=0)
                    qpos = pre_process(qpos)
                    qpos = torch.from_numpy(qpos).float()
                    qpos = qpos[None, :]
                    qpos = qpos.cuda()
                    
                    all_cam_images = []
                    binary_image_fix = data["rbg_image_fix_camera"] 
                    img_buffer_fix = io.BytesIO(binary_image_fix)
                    image_data_fix = Image.open(img_buffer_fix)
                    all_cam_images.append(image_data_fix)

                    binary_image_hand = data["rbg_image_ur_front_left"] 
                    img_buffer_hand = io.BytesIO(binary_image_hand)
                    image_data_hand = Image.open(img_buffer_hand)
                    all_cam_images.append(image_data_hand)
                    
                    all_cam_images = np.stack(all_cam_images, axis=0)                
                    image_data = torch.from_numpy(all_cam_images)
                    image_data = torch.einsum('k h w c -> k c h w', image_data)
                    image_data = image_data / 255.0                
                    curr_image = image_data
                    curr_image = curr_image[None, :]
                    curr_image = curr_image.cuda()

                    t = tasks[i].data_step
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]


                    # raw_action = raw_action[:,0]
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)

                    target_joint_position = np.zeros(12)
                    target_joint_position[0:6] = action[0:6]
                    target_joint_position[6] = action[6]
                    target_joint_position[9] = action[7]
                    action = ArticulationAction(joint_positions=target_joint_position)
                    controller.apply_action(action=action)

                    pick_success_infer = controller.pick_success_infer()
                    print(f"pick_success_infer: {pick_success_infer}")
            else:
                target_position = current_observations[tasks[i]._object.name]['target_position']
                object_position = current_observations[tasks[i]._object.name]['position']
                goal_position = tuple(object_position)
                if current_observations[tasks[i].name + "_event"] == 0:
                    husky_controllers[i].move_to_location(goal_position = object_position, min_distance = 1.0)
                elif current_observations[tasks[i].name + "_event"] == 1:
                    object_position = current_observations[tasks[i]._object.name]['position']
                    husky_controllers[i].pickup_object()
                # elif current_observations[tasks[i].name + "_event"] == 2:
                #     husky_controllers[i].move_to_location(goal_position = target_position, min_distance = 1.0)
                # elif current_observations[tasks[i].name + "_event"] == 3:
                #     husky_controllers[i].put_object(object_position)
                # elif current_observations[tasks[i].name + "_event"] == 4:
                #     print('TRAJECTORY DONE!!!')

                # # log data 
                # if current_observations[tasks[i].name + "_event"] == 1:
                #     data = tasks[i].data_frame_logging_func(event = controller._pick_place_controller._event)
                #     tasks[i]._data_logger.add_data(data)

                    # imgplot = plt.imshow(tasks[i].camera.get_rgba()[:, :, :3])
                    # plt.show()
                    # print(tasks[i].camera.get_current_frame()["motion_vectors"])

    simulation_app.update()
simulation_app.close() 
