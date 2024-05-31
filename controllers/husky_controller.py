import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.kit import SimulationApp
from omni.isaac.motion_generation import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from omni.usd import get_world_transform_matrix

# from src.actions.base_server import ActionServer
# from src.config import Config
# from src.controllers.pick_place import PickPlaceController
# from src.sensors.cameras import setup_cameras
# from src.sensors.imu import setup_imu_graph
# from src.sensors.lidar import setup_lidar_graph
# from src.sensors.tf import setup_tf_graph
# from src.tasks.pick_place import PickPlace

from controllers.pick_place import PickPlaceController
from configs.main_config import MainConfig
from tasks.pick_place import PickPlace
from typing import List, Optional, Tuple
import asyncio


from omni.isaac.core import World
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.kit import SimulationApp
from omni.isaac.motion_generation import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from omni.usd import get_world_transform_matrix
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics, Usd, Sdf


class HuskyController:
    def __init__(self, task:PickPlace, world: World, simulation_app: SimulationApp) -> None:
        """
        A class that initializes and controls the Husky robot in the simulated environment.

        Args:
            cfg (Config): The configuration object for the simulation.
            world (World): The Isaac simulation world object.
            simulation_app (SimulationApp): The Isaac simulation application object.
        """
        self._task = task
        self._world = world
        self._simulation_app = simulation_app
        self._prim_trans_point = self._task._robots["trans_point"]

        self._husky_controller = WheelBasePoseController(
            name="husky_controller",
            open_loop_wheel_controller=DifferentialController(
                name="simple_control", wheel_radius=self._task._config.wheel_radius, wheel_base=self._task._config.wheel_base
            ),
            is_holonomic=False,
        )
        self._pick_place_controller = None
        # self._pick_place_controller = PickPlaceController(
        #     name="ur5_controller",
        #     robot_articulation=self._task._ur5,
        #     gripper=self._task._ur5.gripper,
        #     config=self._task._config,
        # )
        
        self._articulation_controller = self._task._ur5.get_articulation_controller()
        self._pick_done = False
        self._place_done = False
        self._pick_t = 0

    def move_to_location(self, goal_position: Tuple[float, ...], min_distance: float = 1.6) -> bool:
        """
        Moves the Husky robot to a specified location.

        Args:
            task: The task object containing the location to move to.
            action_server: The action server object for the move to location action.
        """

        # distance = 100
        goal_position = list(goal_position)

        #while not (distance < min_distance):
            #print(self._task._husky.get_applied_action())

        distance = np.sum((self._task._husky.get_world_pose()[0][:2] - goal_position[:2]) ** 2)  # Compute distance between husky and target

        if distance > min_distance:
            position, orientation = self._task._husky.get_world_pose()
            wheel_actions = self._husky_controller.forward(start_position=position,
                                                    start_orientation=orientation,
                                                    goal_position=goal_position,
                                                    lateral_velocity=self._task._config.lateral_velocity,
                                                    yaw_velocity=self._task._config.yaw_velocity,
                                                    position_tol=self._task._config.position_tol)
            wheel_actions.joint_velocities = np.tile(wheel_actions.joint_velocities, 2)
            print(f"wheel_actions:{wheel_actions}")
            self._task._husky.apply_action(wheel_actions)        
        
        else: # if done moving 
            print('DONE MOVING!!!')
            for i in range(50):
                self._world.step(render=True)

            self._task._task_event += 1
            # end_position, end_orientation = self._task._husky.get_world_pose()
            # wheel_actions = self._husky_controller.forward(start_position=end_position,
            #                                         start_orientation=end_orientation,
            #                                         goal_position=end_position,
            #                                         lateral_velocity=self._task._config.lateral_velocity,
            #                                         yaw_velocity=self._task._config.yaw_velocity,
            #                                         position_tol=self._task._config.position_tol)
            # wheel_actions.joint_velocities = np.tile(wheel_actions.joint_velocities, 2)
            # self._task._husky.apply_action(wheel_actions)
            return False


        #     # self._world.step(render=True)

        #     return False

        return True

    def pickup_object(self) -> bool:
        """
        Picks up an object with the UR5 manipulator.

        Args:
            task: The task object containing the object to pick up.
            action_server: The action server, responsible for this task types.
                            Needed to send feedback and result back to planner.
        """
        stage = get_current_stage()

        #position, orientation = self._task._husky.get_world_pose()
        # joints_state = self._task..manipulator.get_joints_state()
        joints_state = self._task.manipulator.get_joints_state()
        #end_effector_position, _ = husky.manipulator.end_effector.get_local_pose()
        joint_positions = joints_state.positions


        if self._pick_place_controller is None: 
            self._pick_place_controller = PickPlaceController(
                name="ur5_controller",
                robot_articulation=self._task._ur5,
                gripper=self._task._ur5.gripper,
                config=self._task._config,
            )

            self._pick_place_controller._cspace_controller.reset()
            print(self._pick_place_controller.__dict__)
            self._world.step(render=True)


        # self._world.step(render=True)
        observations = self._task.get_observations()
        picking_position = observations[self._task._object.name]['position']
        target_position = observations[self._task._object.name]['target_position']

        actions = self._pick_place_controller.forward(
            picking_position=picking_position,
            placing_position=target_position, 
            current_joint_positions=joint_positions,#observations[self._task._husky.name]["husky_position"],
            end_effector_offset=list(self._task._config.end_effector_offset),
            end_effector_orientation=euler_angles_to_quat(np.array([0.5*np.pi, -0.5*np.pi, 0])),
            prim_trans_point=self._prim_trans_point,
        )
        # print(self._pick_place_controller._event)
        self._articulation_controller.apply_action(actions)

        if self._pick_place_controller.pick_done():
            self._task._task_event += 1
            self._pick_place_controller.pause()
            print("PCIK_DONE!")
            self._pick_done = True
            return False

        return True
    
    def pick_success(self) -> bool:  # 这里是在收集数据的时候判断是否抓取成功
        if self._pick_place_controller._event in [2, 3]:
            object_position =  self._task._object.get_world_pose()[0]
            end_effector_position = self._task.manipulator.end_effector.get_world_pose()[0]
            distance_ee_obj = np.linalg.norm(object_position - end_effector_position)
            print(f"distance_ee_obj: {distance_ee_obj}")
            if distance_ee_obj > 0.16:
                pick_success = False
            else:
                pick_success = True
        else:
                pick_success = False
        return pick_success

    def pick_success_infer(self) -> bool:  # 在Infer时判断是否抓取成功

        object_position =  self._task._object.get_world_pose()[0]
        end_effector_position = self._task.manipulator.end_effector.get_world_pose()[0]
        distance_ee_obj = np.linalg.norm(object_position - end_effector_position)
        print(f"distance_ee_obj: {distance_ee_obj}")

        default_joints_positions = self._task._config.joints_default_positions
        current_joints_positions = self._task.manipulator.get_joints_state().positions
        current_joints_positions = current_joints_positions[:6]
        joint_pose_error = default_joints_positions-current_joints_positions
        joint_pose_error_max = max(joint_pose_error)
        print(f"joint_pose_error_max: {joint_pose_error_max}\n")

        if (distance_ee_obj < 0.16) and (joint_pose_error_max<0.1):
            pick_success = True
        else:
            pick_success = False

        return pick_success

    def put_object(self, object_position) -> bool:
        
        self.object_position = object_position

        stage = get_current_stage()

        joints_state = self._task.manipulator.get_joints_state()
        joint_positions = joints_state.positions

        if self._pick_place_controller.is_paused:
            #print(f'IS PAUSED: {self._pick_place_controller.is_paused()}')
            self._pick_place_controller._cspace_controller.reset()
            self._pick_place_controller.resume()
            # print('RESUMED _pick_place_controller')

        # self._world.step(render=True)

        observations = self._task.get_observations()
        picking_position = observations[self._task._object.name]['position']
        target_position = observations[self._task._object.name]['target_position']
        end_effector_position = observations[self._task.manipulator.name]['end_effector_position']


        # end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0.8 * np.pi]))
        actions = self._pick_place_controller.forward(
            picking_position=picking_position,
            # placing_position=target_position,
            placing_position=self._task._target_position,
            current_joint_positions=joint_positions,
            end_effector_offset=np.array(self._task._config.end_effector_offset),
            end_effector_orientation=euler_angles_to_quat(np.array([0.5*np.pi, -0.5*np.pi, 0])),
            prim_trans_point=self._prim_trans_point,
            object_position = object_position,
        )
        self._articulation_controller.apply_action(actions)
        print(self._pick_place_controller._event)
        if self._pick_place_controller.is_done():

            self._task._task_event += 1
            self._pick_place_controller.reset()
            self._place_done = True

            print("PLACING_DONE!")
            return False

        return True
    
    def place_success(self) -> bool:
        # if self._pick_place_controller._event in [2, 3]:
        #     object_position =  self._task._object.get_world_pose()[0]
        #     end_effector_position = self._task.manipulator.end_effector.get_world_pose()[0]
        #     distance_ee_obj = np.linalg.norm(object_position - end_effector_position)
        # TODO:
        return True

    def apply_action(self, action):

        self._articulation_controller.apply_action(action)

        pass
