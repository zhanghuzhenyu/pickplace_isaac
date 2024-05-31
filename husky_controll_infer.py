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

# from sim_env import BOX_POSE

eval = True

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    # ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/DATA/collect_srcipt_17/act/husky_ckpt/policy_best.ckpt'
    # ckpt_path = '/home/zhang/act2/ckpt/policy_epoch_100.ckpt'
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    # stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    stats_path = '/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/DATA/collect_srcipt_17/act/husky_ckpt/dataset_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    # post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    # TODO:
    return policy, stats
    
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy
########################## LOAD CKPT #######################33

state_dim = 8
lr_backbone = 1e-5
backbone = 'resnet18'
policy_class = 'ACT'
enc_layers = 4
dec_layers = 7
policy_config = {'backbone': 'resnet18', 'camera_names': ['fixed'], 'dec_layers': 7, 'dim_feedforward': 3200, 'enc_layers': 4, 'hidden_dim': 512, 'kl_weight': 10, 'lr': 1e-05, 'lr_backbone': 1e-05, 'nheads': 8, 'num_queries': 100}
config = {'num_epochs': 2000, 'ckpt_dir': 'husky_ckpt', 'episode_len': 500, 'state_dim': 8, 'lr': 1e-05, 'policy_class': 'ACT', 'onscreen_render': False, 'policy_config': {'lr': 1e-05, 'num_queries': 100, 'kl_weight': 10, 'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-05, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8, 'camera_names': ['fixed']}, 'task_name': 'sim_husky', 'seed': 0, 'temporal_agg': False, 'camera_names': ['fixed'], 'real_robot': False}

ckpt_names = [f'policy_best.ckpt']
results = []
if eval:
    for ckpt_name in ckpt_names:
        policy,stats = eval_bc(config, ckpt_name, save_episode=True)
        # success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
        # results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    print()


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

# scene_phys = PhysicsContext()
# scene_phys.set_broadphase_type("GPU")
# scene_phys.enable_gpu_dynamics(flag=True)
# scene_phys.set_solver_type("PGS")

############################ SETUP SCENE #############################
'''
    Add multiple tasks
'''
for i in range(num_of_tasks):
    print(i)
    my_world.add_task(PickPlace(name="task_" + str(i), world = my_world, config = config, offset=np.array([0, (i * 30), 0])))
# my_world.add_task(PickPlace(name="task", world = my_world, config = config, offset=np.array([0, 0, 0])))
my_world.scene.add_default_ground_plane()
my_world.reset()

############################ SETUP TASKS #############################
'''
    Add info about tasks
'''

for i in range(num_of_tasks):
    task = my_world.get_task(name="task_" + str(i))
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

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']
max_timesteps = 400
num_queries = policy_config['num_queries'] # 100
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

############################ START SIMULATION ############################ 
exit_flag = False


# while simulation_app.is_running() and not simulation_app.is_exiting() and (tasks[0].data_step < max_timesteps):  # 这一行用于Inference
while simulation_app.is_running() and not simulation_app.is_exiting() and (not controller._pick_done):    # 这一行用于收集数据
# while simulation_app.is_running() and not simulation_app.is_exiting():
    my_world.step(render=True)

    if my_world.is_playing():

        for i in range(num_of_tasks):
            
            phys_time = my_world.current_time_step_index
            print(f"phys_time:{phys_time}")
            current_observations = tasks[i].get_observations()
            # print(current_observations)

            if eval:                
                data = tasks[i].data_frame_logging_func()
                with torch.inference_mode():
                    policy.eval()
                    
                    qpos_arm = data["ur_5_joint_positions"]
                    qpos_gripper = data["gripper_joint_positions"]
                    qpos = np.concatenate((qpos_arm, qpos_gripper), axis=0)
                    qpos = pre_process(qpos)
                    qpos = torch.from_numpy(qpos).float()
                    qpos = qpos[None, :]
                    qpos = qpos.cuda()

                    binary_image = data["rbg_image_fix_camera"] 
                    img_buffer = io.BytesIO(binary_image)
                    image_data = Image.open(img_buffer)
                    all_cam_images = []
                    all_cam_images.append(image_data)
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
                if current_observations[tasks[i].name + "_event"] == 1:
                    data = tasks[i].data_frame_logging_func(event = controller._pick_place_controller._event)
                    tasks[i]._data_logger.add_data(data)

                    # imgplot = plt.imshow(tasks[i].camera.get_rgba()[:, :, :3])
                    # plt.show()
                    # print(tasks[i].camera.get_current_frame()["motion_vectors"])

    simulation_app.update()
simulation_app.close() 
