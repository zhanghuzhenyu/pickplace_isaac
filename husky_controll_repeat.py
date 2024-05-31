import sys
sys.path.append('DATA/collect_srcipt_with_move/act')

import numpy as np

# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from einops import rearrange
# import IPython

################# Isaac 启动！ ########################
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})


from omni.isaac.core import World
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils import stage
from pxr import UsdPhysics,  Sdf, UsdLux
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics
import numpy as np
from tasks.pick_place import PickPlace
from configs.main_config import MainConfig
from controllers.husky_controller import HuskyController
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
import matplotlib.pyplot as plt
import IPython

# import asyncio
# from PIL import Image
# import io
# from omni.isaac.core.utils.types import ArticulationAction

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


############################ START SIMULATION ############################ 

# while simulation_app.is_running() and not simulation_app.is_exiting() and (tasks[0].data_step < max_timesteps):  # 这一行用于Inference
# while simulation_app.is_running() and not simulation_app.is_exiting() and (not controller._pick_done): 
while simulation_app.is_running() and not simulation_app.is_exiting() and (not controller._place_done):   

    my_world.step(render=True)

    if my_world.is_playing():

        for i in range(num_of_tasks):
            
            phys_time = my_world.current_time_step_index
            # print(f"phys_time:{phys_time}")
            current_observations = tasks[i].get_observations()
            # print(current_observations)

            print(f"tasks[i]._target_position: {tasks[i]._target_position}")

            target_position = current_observations[tasks[i]._object.name]['target_position']
            object_position = current_observations[tasks[i]._object.name]['position']
            goal_position = tuple(object_position)
            if current_observations[tasks[i].name + "_event"] == 0:
                husky_controllers[i].move_to_location(goal_position = object_position, min_distance = 1.0)
            elif current_observations[tasks[i].name + "_event"] == 1:
                object_position = current_observations[tasks[i]._object.name]['position']
                husky_controllers[i].pickup_object()
            elif current_observations[tasks[i].name + "_event"] == 2:
                husky_controllers[i].move_to_location(goal_position = target_position, min_distance = 1.0)
            elif current_observations[tasks[i].name + "_event"] == 3:
                # husky_controllers[i].put_object(object_position)
                husky_controllers[i].put_object(tasks[i]._target_position)
            elif current_observations[tasks[i].name + "_event"] == 4:
                print('TRAJECTORY DONE!!!')

            # # log data 
            if current_observations[tasks[i].name + "_event"] == 1:
                data = tasks[i].data_frame_logging_func(event = controller._pick_place_controller._event,
                                                        task_type = "pick",
                                                        success = controller.pick_success())
                tasks[i]._data_logger_pick.add_data(data)
            elif current_observations[tasks[i].name + "_event"] == 3:
                data = tasks[i].data_frame_logging_func(event = controller._pick_place_controller._event,
                                                        task_type = "place",
                                                        success = controller.place_success())
                tasks[i]._data_logger_place.add_data(data)

                # imgplot = plt.imshow(tasks[i].camera.get_rgba()[:, :, :3])
                # plt.show()
                # print(tasks[i].camera.get_current_frame()["motion_vectors"])

    # simulation_app.update()

simulation_app.close() 
