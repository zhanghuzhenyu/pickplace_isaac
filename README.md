# Data collect for Pick and Place

## Success Rate

The calculation of success rate in the data collection process and the inference process is different. Taking pick as an example, we use the distance between the end effector and the object at each time step as the criterion. If the distance exceeds the threshold after the gripper is closed, it means that the object slipped during the grasping process, that is, the grasping failed.

But during the inference process, how do we know whether the gripper is closed? So we also added another condition, that is, whether each joint has returned to its initial position.

Success rate function is located in ./controllers/husky_controller.py
