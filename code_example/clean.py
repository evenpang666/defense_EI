# Primitives: move_to(pos, quat_wxyz, num_steps), move_ee(dx,dy,dz,droll,dpitch,dyaw,steps),
#            gripper_control(value, delay), ee_pose() -> (pos, quat_wxyz), np
# Quaternion for move_to / ee_pose is (w, x, y, z).
# Pre-registered composite APIs (evoma_atomic_ops.py — call directly, do not redefine):
#   pick_and_place, push, pull, press, open, close, pour,
#   move_x, move_y, move_z, rotate_x, rotate_y, rotate_z,
#   get_object_abs_pose(object_poses, object_name),
#   recover_grasp_pose_from_offset(object_pos_xyz, object_quat_wxyz, offset_pos_xyz, offset_rpy_deg)
gripper_control(0)
move_to(pos=[0.4, 0.18, 1.3], num_steps=300)
move_to(pos=[0.4, 0.18, 0.95], num_steps=300)
gripper_control(255, 100)
move_to(pos=[0.4, 0.18, 1.3], num_steps=300)
move_to(pos=[0.4, 0., 1.3], num_steps=300)
move_to(pos=[0.37, -0.015, 1.3], num_steps=300)
move_to(pos=[0.37, -0.015, 1], num_steps=300)
gripper_control(0)

move_to(pos=[0.37, -0.02, 1.3], num_steps=300)
move_to(pos=[0.37, -0.1, 1.3], num_steps=300)
move_to(pos=[0.37, -0.1, 0.98], num_steps=300)
gripper_control(255, 100)
move_to(pos=[0.37, -0.13, 1.4], num_steps=300)
move_to(pos=[-0.12, 0.38, 1.4], num_steps=300)
move_to(pos=[-0.12, 0.38, 1.2], num_steps=300)
gripper_control(0)
move_to(pos=[-0.12, 0.38, 1.4], num_steps=300)

move_to(pos=[-0.12, 0.33, 1.2], quat=[0,1,0,0], num_steps=300)
move_to(pos=[-0.12, 0.33, 1.13], quat=[0,1,0,0], num_steps=300)
gripper_control(255, 100)
move_to(pos=[-0.12, 0.33, 1.4], quat=[0,1,0,0], num_steps=500)
move_to(pos=[0.37, -0.05, 1.4], quat=[0,1,0,0], num_steps=500)
move_to(pos=[0.37, -0.05, 1.], quat=[0,1,0,0], num_steps=500)
gripper_control(0, 100)