import numpy as np
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics, Usd, Sdf, UsdLux

# Init World
world = World(stage_units_in_meters=1.0)

# Set init and target pose
object_path = "/World/cube"
init_position = np.array([0.8, 0, 0.45])  
init_orientation = (0.70711, -0.70711, 0, 0)  
target_position = np.array([0.8, 0, 0.45])  # target placement position (x, y, z)
target_orientation = Gf.Quatf(1.0, 0.0, 0.0, 0.0)  # Target placement orientation (quaternion)
# target_orientation = (1.0, 0.0, 0.0, 0.0)  

# Get the position and orientation of object
def get_pose(prim_path):
    prim = get_prim_at_path(prim_path)
    position = np.array(prim.GetAttribute("xformOp:translate").Get())
    orientation = prim.GetAttribute("xformOp:orient").Get()
    return position, orientation

# Calculate position error and orientation error
def compute_errors(target_pos, target_ori, actual_pos, actual_ori):
    position_error = np.linalg.norm(actual_pos - target_pos)
    orientation_error = np.arccos(np.clip(np.dot(actual_ori.GetReal(), target_ori.GetReal()), -1.0, 1.0)) * 2.0
    return position_error, orientation_error

# Check place accuracy
def check_placement_accuracy():
    actual_position, actual_orientation = get_pose(object_path)
    pos_error, ori_error = compute_errors(target_position, target_orientation, actual_position, actual_orientation)
    print(f"Position Error: {pos_error:.4f} meters")
    print(f"Orientation Error: {ori_error:.4f} radians")

def add_objectes():
    # Add Cube
    object = world.scene.add(
        DynamicCuboid(
            name="cube",
            position=init_position,
            orientation=init_orientation,
            prim_path="/World/cube",
            scale=(0.05, 0.05, 0.05),
            size=1.0,
            color=np.array([0, 0, 1]),
        )
    )
    # Add Table
    add_reference_to_stage(usd_path="/home/zhang/asset/DATA_asset/table_place.usd",  # table.usd is without place red point
                            prim_path="/World/table")
    table_init_position = (0.9, 0, 0)
    world.scene.add(XFormPrim(prim_path="/World/table", name="table", position = table_init_position, scale = np.array([1.2, 0.9, 0.65]),
                orientation = np.array([0.70711, 0, 0, 0.70711])))

    # # Add red place point
    # place_prim = XFormPrim("/World/table/urdf_table/place_pos", name="place_point")
    # place_prim.set_world_pose(position=np.array([0, 0, 0.416]))

def add_light():
    stage = get_current_stage()
    demoLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    demoLight.CreateIntensityAttr(500)

add_objectes()
add_light()

# run the simulation
world.reset()
while simulation_app.app.is_running():
    world.step(render=True)
    check_placement_accuracy()
    

simulation_app.close()
