# Primitives: move_to(pos, quat_wxyz, num_steps), move_ee(dx,dy,dz,droll,dpitch,dyaw,steps),
#            gripper_control(value, delay), ee_pose() -> (pos, quat_wxyz), np
# Quaternion for move_to / ee_pose is (w, x, y, z).
# Pre-registered composite APIs (evoma_atomic_ops.py — call directly, do not redefine):
#   pick_and_place, push, pull, press, open, close, pour,
#   move_x, move_y, move_z, rotate_x, rotate_y, rotate_z,
#   get_object_abs_pose(object_poses, object_name),
#   recover_grasp_pose_from_offset(object_pos_xyz, object_quat_wxyz, offset_pos_xyz, offset_rpy_deg)

gripper_control(0)
move_to(pos=[0.037, -0.51, 1.3], num_steps=300)
move_ee(0,0,0,0,0,-90,steps=300)
move_to(pos=[0.037, -0.51, 1.18], num_steps=300)
gripper_control(255)
move_to(pos=[0.037, -0.51, 1.23], num_steps=300)
move_ee(0,0,0,0,45,0,steps=300)